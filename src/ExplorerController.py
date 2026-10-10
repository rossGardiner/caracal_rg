from functools import partial

from PySide6.QtCore import QObject, Signal, Slot

from src.AudioBufferLoader import AudioBufferLoader
from src.AudioReader import AudioReader
from src.CaracalStreamer import CaracalStreamer
from src.EmbeddingCacheLoader import EmbeddingCacheLoader
from src.EmbeddingRequest import EmbeddingRequest
from src.EmbeddingsCreator import EmbeddingsCreator
from src.EmbeddingCorpus import EmbeddingRef, EmbeddingSpaceKey
from src.ExplorerModel import ExplorerModel
from src.LatestJobRunner import LatestJobRunner
from src.SpectrogramData import calculate_spectrogram


class ExplorerController(QObject):
    """Coordinate interactive explorer work between model and view.

    The controller owns asynchronous job policy and stale-result protection.
    It deliberately does not own Qt window layout or audio-device playback.
    """

    status_changed = Signal(str)
    stop_browsing_playback_requested = Signal()
    stop_processed_playback_requested = Signal()
    embedding_selection_changed = Signal(object, object)

    def __init__(
        self,
        model: ExplorerModel,
        view,
        parent=None,
    ):
        super().__init__(parent)

        if not isinstance(model, ExplorerModel):
            raise TypeError("model must be an ExplorerModel")

        self.model = model
        self.view = view

        self.view.recording_selected.connect(
            self.recording_selected
        )
        self.view.chunk_selected.connect(
            self.chunk_selected
        )
        self.view.processed_buffer_selected.connect(
            self.processed_buffer_selected
        )
        self.view.processed_buffers_received.connect(
            self.update_processed_buffers
        )

        self.status_changed.connect(
            self.view.show_status
        )
        self.stop_browsing_playback_requested.connect(
            self.view.stop_browsing_audio
        )
        self.stop_processed_playback_requested.connect(
            self.view.stop_processed_audio
        )

        self._is_shutdown = False

        self._browsing_request_id = 0
        self._processed_spectrogram_request_id = 0
        self._embedding_cache_request_id = 0
        self._interactive_embedding_request_id = 0

        self.embedding_request = EmbeddingRequest(
            cache=self.model.embedding_cache,
            pipeline=self.model.pipeline,
        )

        self.embedding_cache_jobs = LatestJobRunner(
            name="embedding-cache-jobs",
            parent=self,
        )
        self.embedding_cache_jobs.ready.connect(
            self._cached_embeddings_loaded
        )
        self.embedding_cache_jobs.failed.connect(
            self._cached_embeddings_failed
        )

        self.embedding_request_jobs = LatestJobRunner(
            name="embedding-request-jobs",
            parent=self,
        )
        self.embedding_request_jobs.ready.connect(
            self._interactive_embedding_ready
        )
        self.embedding_request_jobs.failed.connect(
            self._interactive_embedding_failed
        )

        self.browsing_audio_jobs = LatestJobRunner(
            name="browsing-audio-jobs",
            parent=self,
        )
        self.browsing_audio_jobs.ready.connect(
            self._browsing_audio_loaded
        )
        self.browsing_audio_jobs.failed.connect(
            self._browsing_audio_failed
        )

        self.browsing_spectrogram_jobs = LatestJobRunner(
            name="browsing-spectrogram-jobs",
            parent=self,
        )
        self.browsing_spectrogram_jobs.ready.connect(
            self._browsing_spectrogram_ready
        )
        self.browsing_spectrogram_jobs.failed.connect(
            self._browsing_spectrogram_failed
        )

        self.processed_spectrogram_jobs = LatestJobRunner(
            name="processed-spectrogram-jobs",
            parent=self,
        )
        self.processed_spectrogram_jobs.ready.connect(
            self._processed_spectrogram_ready
        )
        self.processed_spectrogram_jobs.failed.connect(
            self._processed_spectrogram_failed
        )

    @Slot(str, object)
    def activate_pipeline(self, pipeline_name: str, pipeline):
        """Make one runtime pipeline the Explorer's active configuration."""

        source = pipeline.get_link(CaracalStreamer)
        loader = pipeline.get_link(AudioBufferLoader)
        cache_loader = pipeline.get_link(EmbeddingCacheLoader)
        embeddings_creator = pipeline.get_link(EmbeddingsCreator)

        # EmbeddingRequest currently mirrors the interactive subset of the
        # configured callback graph, so fail activation early if that subset
        # is not available rather than leaving the Explorer half-switched.
        candidate_request = EmbeddingRequest(
            cache=cache_loader.cache,
            pipeline=pipeline,
        )

        previous_recording_id = self.model.selected_recording_id

        self.stop_browsing_playback_requested.emit()
        self.stop_processed_playback_requested.emit()

        # Invalidate results already running for the previous pipeline.
        self._browsing_request_id += 1
        self._processed_spectrogram_request_id += 1
        self._embedding_cache_request_id += 1
        self._interactive_embedding_request_id += 1

        self.model.set_pipeline_context(
            pipeline=pipeline,
            pipeline_name=pipeline_name,
            audio_packet_source=source,
            audio_reader=AudioReader(is_caracal=loader.is_caracal),
            chunk_grid=loader.chunk_grid,
            embedding_cache=cache_loader.cache,
            embedding_name=embeddings_creator.embedding_name,
        )
        self.embedding_request = candidate_request

        self.model.browsing_buffers = []
        self.model.processed_buffers = []
        self.model.current_processed_buffer_index = 0
        self.model.interactive_embeddings_by_key.clear()

        selected_packet_index = None
        if previous_recording_id is not None:
            for index, packet in enumerate(self.model.audio_packets):
                if packet.recording_id == previous_recording_id:
                    selected_packet_index = index
                    break

        if selected_packet_index is None and self.model.audio_packets:
            selected_packet_index = 0

        self.view.set_active_pipeline(
            self.model.pipeline_name,
            self.model.pipeline_hash,
        )
        self.view.set_embedding_name(self.model.embedding_name)
        self.view.set_audio_packets(
            self.model.audio_packets,
            selected_packet_index=selected_packet_index,
        )
        self.view.clear_browsing_spectrogram()
        self._clear_processed_audio()
        self.view.clear_recording_embeddings(
            f"Loading cached {self.model.embedding_name!r} embeddings..."
        )

        if selected_packet_index is None:
            self._clear_recording_selection()
        else:
            self.recording_selected(selected_packet_index)

        self.status_changed.emit(
            f"Explorer now using {self.model.pipeline_name} "
            f"({self.model.pipeline_hash})"
        )

        return self.model.pipeline_hash

    def initialize(self, packet_index=None):
        """Initialise the explorer selection without doing work in a constructor."""

        self.view.set_browsing_controls_enabled(False)

        if packet_index is None:
            self.view.set_no_recordings()
            return

        self.recording_selected(packet_index)

    # ======================================================
    # Optional processed-pipeline observation
    # ======================================================

    @Slot(object)
    def update_processed_buffers(self, buffers):
        """Receive a processed batch from an optional GuiPipelineLink."""

        buffers = list(buffers)

        if not buffers:
            return

        self.stop_processed_playback_requested.emit()

        self.model.processed_buffers = buffers
        self.model.current_processed_buffer_index = 0

        selected_recording_id = self.model.selected_recording_id
        matching_buffers = [
            buffer
            for buffer in self.model.processed_buffers
            if buffer.recording_id == selected_recording_id
        ]

        self.view.add_processed_buffers(
            matching_buffers
        )

        self.processed_buffer_selected(0)
        self._update_pipeline_status()

    # ======================================================
    # Processed audio inspection
    # ======================================================

    @Slot(int)
    def processed_buffer_selected(self, buffer_index: int):
        """Select a processed buffer and calculate its spectrogram off-thread."""

        if not self.model.processed_buffers:
            self.view.clear_processed_audio()
            return

        if not (
            0
            <= buffer_index
            < len(self.model.processed_buffers)
        ):
            return

        self.stop_processed_playback_requested.emit()

        self.model.current_processed_buffer_index = buffer_index
        audio_buffer = self.model.current_processed_buffer

        self.view.set_processed_buffer_selection(
            index=buffer_index,
            buffer_count=len(self.model.processed_buffers),
            audio_buffer=audio_buffer,
        )

        self.view.clear_processed_spectrogram()

        self._processed_spectrogram_request_id += 1
        request_id = self._processed_spectrogram_request_id

        self.processed_spectrogram_jobs.submit(
            request_id,
            partial(
                calculate_spectrogram,
                waveform=audio_buffer.waveform,
                sample_rate=audio_buffer.sample_rate,
            ),
        )

    @Slot(int, object)
    def _processed_spectrogram_ready(self, request_id: int, result):
        if request_id != self._processed_spectrogram_request_id:
            return

        self.view.set_processed_spectrogram_result(
            result
        )

    @Slot(int, str)
    def _processed_spectrogram_failed(self, request_id: int, message: str):
        if request_id != self._processed_spectrogram_request_id:
            return

        self.view.clear_processed_spectrogram()
        self.status_changed.emit(
            "Could not calculate processed spectrogram: "
            f"{message}"
        )

    # ======================================================
    # Recording / chunk selection
    # ======================================================

    @Slot(int)
    def recording_selected(self, packet_index: int):
        """Select a logical recording and request its first canonical chunk."""

        self._select_recording(
            packet_index,
            chunk_index=0,
            update_view_selection=False,
        )

    @Slot(object)
    def navigate_to_embedding(self, ref):
        """Navigate Explorer to one recording/chunk reference.

        The composition root uses this as the bridge from Active Learning.
        Explorer remains responsible for translating the stable embedding ref
        into its own packet/chunk selection and loading the corresponding audio.
        """

        if not isinstance(ref, EmbeddingRef):
            self.status_changed.emit("Could not open Active Learning candidate: invalid ref")
            return False

        packet_index = None
        for index, packet in enumerate(self.model.audio_packets):
            if str(packet.recording_id) == ref.recording_id:
                packet_index = index
                break

        if packet_index is None:
            self.status_changed.emit(
                f"Could not open Active Learning candidate: recording "
                f"{ref.recording_id!r} is not available in Explorer"
            )
            return False

        packet = self.model.audio_packets[packet_index]
        if packet.duration is None:
            self.status_changed.emit(
                "Could not open Active Learning candidate: recording has no duration"
            )
            return False

        chunk_count = self.model.chunk_grid.chunk_count(packet.duration)
        if not 0 <= ref.chunk_index < chunk_count:
            self.status_changed.emit(
                f"Could not open Active Learning candidate: chunk "
                f"{ref.chunk_index} is outside the recording"
            )
            return False

        self._select_recording(
            packet_index,
            chunk_index=ref.chunk_index,
            update_view_selection=True,
        )
        return True

    def current_embedding_selection(self):
        """Return the Explorer's current embedding space and selected chunk."""

        space = EmbeddingSpaceKey(
            pipeline_hash=self.model.pipeline_hash,
            embedding_name=self.model.embedding_name,
        )

        if self.model.selected_recording_id is None:
            return space, None

        return space, EmbeddingRef(
            recording_id=self.model.selected_recording_id,
            chunk_index=self.model.current_chunk_index,
        )

    def _select_recording(
        self,
        packet_index: int,
        *,
        chunk_index: int,
        update_view_selection: bool,
    ):
        if (
            packet_index < 0
            or packet_index >= len(self.model.audio_packets)
        ):
            self._clear_recording_selection()
            return

        packet = self.model.audio_packets[packet_index]

        if packet.duration is None:
            raise ValueError(
                "Selected AudioPacket has no duration"
            )

        chunk_count = self.model.chunk_grid.chunk_count(packet.duration)
        if chunk_count <= 0:
            raise ValueError(
                "Selected AudioPacket has no canonical chunks"
            )

        chunk_index = int(chunk_index)
        if not 0 <= chunk_index < chunk_count:
            raise ValueError(
                f"Chunk index {chunk_index} is outside 0..{chunk_count - 1}"
            )

        self.model.current_audio_packet = packet
        self.model.current_chunk_index = chunk_index
        self.model.current_num_chunks = chunk_count
        self.model.interactive_embeddings_by_key.clear()

        self.view.set_selected_embedding_status(
            "Selected chunk embedding: waiting for source audio..."
        )

        self.model.set_recording_navigation(
            duration_s=packet.duration,
            cursor_time_s=0.0,
        )

        if update_view_selection:
            self.view.set_recording_selection(packet_index)

        self.view.set_chunk_range(self.model.current_num_chunks)
        self.view.set_chunk_selection(chunk_index)

        self._request_cached_embeddings()
        self._update_chunk_label()
        self._emit_embedding_selection()
        self._request_selected_chunk()

    def _clear_recording_selection(self):
        self.model.current_audio_packet = None
        self.model.current_chunk_index = 0
        self.model.current_num_chunks = 0
        self.model.clear_recording_navigation()
        self.model.browsing_buffers = []
        self._clear_processed_audio()

        # Invalidate any result still running for the previous selection.
        self._browsing_request_id += 1
        self._embedding_cache_request_id += 1
        self._interactive_embedding_request_id += 1
        self.model.interactive_embeddings_by_key.clear()

        self.view.clear_recording_embeddings(
            "No recording selected"
        )
        self.view.set_selected_embedding_status(
            "Selected chunk embedding: no recording selected"
        )

        self.view.set_browsing_controls_enabled(
            False
        )
        self.view.set_chunk_range(
            0
        )
        self.view.set_chunk_label(
            "No recording selected"
        )
        self.view.clear_browsing_spectrogram()
        self._emit_embedding_selection()

    @Slot(int)
    def chunk_selected(self, chunk_index: int):
        if self.model.current_audio_packet is None:
            return

        chunk_index = int(chunk_index)
        if not 0 <= chunk_index < self.model.current_num_chunks:
            return

        self.model.current_chunk_index = chunk_index

        self._update_chunk_label()
        self._emit_embedding_selection()
        self._request_selected_chunk()

    def _request_selected_chunk(self):
        if self.model.current_audio_packet is None:
            return

        self.stop_browsing_playback_requested.emit()
        self._clear_processed_audio()

        self.model.browsing_buffers = []
        self.view.clear_browsing_spectrogram()

        start_s, end_s = self.model.chunk_grid.chunk_bounds(
            self.model.current_chunk_index,
            total_duration_s=self.model.current_audio_packet.duration,
        )

        self.model.set_loaded_window(
            start_s=start_s,
            duration_s=(end_s - start_s),
            cursor_time_s=start_s + ((end_s - start_s) / 2.0),
        )

        self._browsing_request_id += 1
        self._interactive_embedding_request_id += 1
        request_id = self._browsing_request_id

        self.view.set_selected_embedding_status(
            f"Selected chunk embedding: waiting for chunk "
            f"{self.model.current_chunk_index} audio..."
        )

        self.view.set_browsing_controls_enabled(
            False
        )

        self.status_changed.emit(
            f"Loading chunk {self.model.current_chunk_index}..."
        )

        self.browsing_audio_jobs.submit(
            request_id,
            partial(
                self.model.audio_reader.read_buffer,
                packet=self.model.current_audio_packet,
                chunk_index=self.model.current_chunk_index,
                start_s=start_s,
                duration_s=(end_s - start_s),
            ),
        )

    def _update_chunk_label(self):
        if self.model.current_audio_packet is None:
            self.view.set_chunk_label(
                "No recording selected"
            )
            self.view.clear_browsing_spectrogram()
            return

        start_s, end_s = self.model.chunk_grid.chunk_bounds(
            self.model.current_chunk_index,
            total_duration_s=self.model.current_audio_packet.duration,
        )

        self.view.set_chunk_label(
            f"{self.model.current_chunk_index} "
            f"({start_s:.1f}s - {end_s:.1f}s)"
        )

    # ======================================================
    # Cached / interactive embeddings
    # ======================================================

    def _request_cached_embeddings(self):
        recording_id = self.model.selected_recording_id

        self._embedding_cache_request_id += 1
        request_id = self._embedding_cache_request_id

        if recording_id is None:
            self.view.clear_recording_embeddings(
                "Selected recording has no recording_id"
            )
            return

        self.view.clear_recording_embeddings(
            f"Loading cached {self.model.embedding_name!r} embeddings..."
        )

        self.embedding_cache_jobs.submit(
            request_id,
            partial(
                self.model.embedding_cache.list_embeddings,
                recording_id=recording_id,
                pipeline_hash=self.model.pipeline_hash,
                embedding_name=self.model.embedding_name,
            ),
        )

    @Slot(int, object)
    def _cached_embeddings_loaded(self, request_id: int, embeddings):
        if request_id != self._embedding_cache_request_id:
            return

        self.view.set_recording_embeddings(
            embeddings
        )

        if self.model.interactive_embeddings_by_key:
            self.view.add_embeddings(
                self.model.interactive_embeddings_by_key.values()
            )

        selected_recording_id = self.model.selected_recording_id
        matching_buffers = [
            buffer
            for buffer in self.model.processed_buffers
            if buffer.recording_id == selected_recording_id
        ]

        if matching_buffers:
            self.view.add_processed_buffers(
                matching_buffers
            )

    @Slot(int, str)
    def _cached_embeddings_failed(self, request_id: int, message: str):
        if request_id != self._embedding_cache_request_id:
            return

        self.view.clear_recording_embeddings(
            "Could not load cached embeddings"
        )
        self.status_changed.emit(
            f"Could not load cached embeddings: {message}"
        )

    @Slot(int, object)
    def _interactive_embedding_ready(
        self,
        request_id: int,
        processed_buffer,
    ):
        if request_id != self._interactive_embedding_request_id:
            return

        if processed_buffer.recording_id != self.model.selected_recording_id:
            return

        if processed_buffer.chunk_index != self.model.current_chunk_index:
            return

        embedding = processed_buffer.embeddings.get(
            self.model.embedding_name
        )
        if embedding is None:
            self._interactive_embedding_failed(
                request_id,
                f"Processed buffer has no {self.model.embedding_name!r} embedding",
            )
            return

        embedding_record = {
            "recording_id": processed_buffer.recording_id,
            "chunk_index": processed_buffer.chunk_index,
            "values": embedding["values"],
            "pipeline_hash": embedding["pipeline_hash"],
            "embedding_name": self.model.embedding_name,
            "metadata": embedding.get("metadata", {}),
            "loaded_from_cache": bool(
                embedding.get("loaded_from_cache", False)
            ),
        }

        key = (
            str(processed_buffer.recording_id),
            int(processed_buffer.chunk_index),
        )
        self.model.interactive_embeddings_by_key[key] = embedding_record

        self.view.add_embeddings(
            [embedding_record]
        )

        # The interactive request has now produced the exact processed audio
        # representation for the selected canonical chunk.  Make that the
        # processed pane's current buffer and calculate its spectrogram using
        # the same existing processed-audio path used by observed pipelines.
        self.stop_processed_playback_requested.emit()
        self.model.processed_buffers = [
            processed_buffer
        ]
        self.model.current_processed_buffer_index = 0
        self.processed_buffer_selected(0)
        self._update_pipeline_status()

        if embedding_record.get("loaded_from_cache", False):
            source_text = "loaded from cache"
        else:
            source_text = "computed and cached"

        self.view.set_selected_embedding_status(
            f"Selected chunk embedding: chunk {processed_buffer.chunk_index} "
            f"{source_text}"
        )

    @Slot(int, str)
    def _interactive_embedding_failed(self, request_id: int, message: str):
        if request_id != self._interactive_embedding_request_id:
            return

        self.view.set_selected_embedding_status(
            "Selected chunk embedding failed: "
            f"{message}"
        )

    # ======================================================
    # Browsing audio / spectrogram results
    # ======================================================

    @Slot(int, object)
    def _browsing_audio_loaded(self, request_id: int, audio_buffer):
        if request_id != self._browsing_request_id:
            return

        self.model.browsing_buffers = [
            audio_buffer
        ]

        self.view.set_browsing_controls_enabled(
            True
        )

        self.status_changed.emit(
            f"Loaded chunk {audio_buffer.chunk_index}; "
            "calculating spectrogram..."
        )

        self.browsing_spectrogram_jobs.submit(
            request_id,
            partial(
                calculate_spectrogram,
                waveform=audio_buffer.waveform,
                sample_rate=audio_buffer.sample_rate,
            ),
        )

        embedding_request_id = self._interactive_embedding_request_id

        self.view.set_selected_embedding_status(
            f"Selected chunk embedding: checking chunk "
            f"{audio_buffer.chunk_index}..."
        )

        self.embedding_request_jobs.submit(
            embedding_request_id,
            partial(
                self.embedding_request.run,
                audio_buffer,
            ),
        )

    @Slot(int, object)
    def _browsing_spectrogram_ready(self, request_id: int, result):
        if request_id != self._browsing_request_id:
            return

        self.view.set_browsing_spectrogram_result(
            result
        )
        self.status_changed.emit(
            f"Loaded chunk {self.model.current_chunk_index}"
        )

    @Slot(int, str)
    def _browsing_spectrogram_failed(self, request_id: int, message: str):
        if request_id != self._browsing_request_id:
            return

        self.view.clear_browsing_spectrogram()
        self.status_changed.emit(
            f"Audio loaded, but spectrogram failed: {message}"
        )

    @Slot(int, str)
    def _browsing_audio_failed(self, request_id: int, message: str):
        if request_id != self._browsing_request_id:
            return

        self.model.browsing_buffers = []
        self._clear_processed_audio()
        self.view.clear_browsing_spectrogram()
        self.view.set_browsing_controls_enabled(
            False
        )
        self.view.set_selected_embedding_status(
            "Selected chunk embedding: source audio unavailable"
        )
        self.status_changed.emit(
            f"Could not load chunk: {message}"
        )

    def _clear_processed_audio(self):
        """Clear processed audio and invalidate any outstanding spectrogram."""

        self.stop_processed_playback_requested.emit()
        self.model.processed_buffers = []
        self.model.current_processed_buffer_index = 0

        self._processed_spectrogram_request_id += 1

        self.view.clear_processed_audio()
        self._update_pipeline_status()

    # ======================================================
    # Processed status
    # ======================================================

    def _update_pipeline_status(self):
        if not self.model.processed_buffers:
            self.view.set_processed_status(
                "Waiting for processed audio..."
            )
            return

        first_buffer = self.model.processed_buffers[0]
        sample_rate = first_buffer.sample_rate
        first_waveform = first_buffer.waveform

        if first_waveform.ndim == 1:
            channel_count = 1
        else:
            channel_count = first_waveform.shape[1]

        total_samples = sum(
            len(buffer.waveform)
            for buffer in self.model.processed_buffers
        )
        duration = total_samples / sample_rate

        total_embeddings = self.view.embedding_count()

        if len(self.model.processed_buffers) == 1:
            buffer = self.model.processed_buffers[0]
            if buffer.chunk_index is None:
                source_text = "Processed selection"
            else:
                source_text = f"Processed chunk {buffer.chunk_index}"
        else:
            source_text = (
                f"Observed batch: {len(self.model.processed_buffers)} buffers"
            )

        self.view.set_processed_status(
            f"{source_text}"
            f" | {duration:.2f}s"
            f" | {sample_rate} Hz"
            f" | {channel_count} channel(s)"
            f" | {total_embeddings} selected-recording embeddings"
        )

    # ======================================================
    # Cleanup
    # ======================================================

    def _emit_embedding_selection(self):
        space, ref = self.current_embedding_selection()
        self.embedding_selection_changed.emit(space, ref)

    def shutdown(self):
        if self._is_shutdown:
            return

        self._is_shutdown = True

        self.browsing_audio_jobs.shutdown()
        self.browsing_spectrogram_jobs.shutdown()
        self.processed_spectrogram_jobs.shutdown()
        self.embedding_cache_jobs.shutdown()
        self.embedding_request_jobs.shutdown()
