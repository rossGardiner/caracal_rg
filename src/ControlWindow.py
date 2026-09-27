import numpy as np

from functools import partial

from PySide6.QtCore import (
    Slot,
    QByteArray,
    QBuffer,
    QIODevice,
    Qt,
)

from PySide6.QtWidgets import (
    QDockWidget,
    QMainWindow,
)

from PySide6.QtMultimedia import (
    QAudioFormat,
    QAudioSink,
    QMediaDevices,
)

from src.EmbeddingRequest import EmbeddingRequest
from src.BrowsingView import BrowsingView
from src.ProcessedView import ProcessedView
from src.LatestJobRunner import LatestJobRunner
from src.SpectrogramData import calculate_spectrogram
from src.ExplorerModel import ExplorerModel


class ControlWindow(QMainWindow):
    """
    Main application coordinator.

    GUI ownership is split between:

        BrowsingView
            source recording navigation, source spectrogram,
            playback controls

        ProcessedView
            processed audio inspection, processed embeddings,
            non-blocking pipeline status

    ControlWindow coordinates data loading, playback, and pipeline
    observation without allowing the two views to share GUI state.
    """

    def __init__(
        self,
        model: ExplorerModel,
    ):
        super().__init__()

        self._is_shutdown = False

        if not isinstance(
            model,
            ExplorerModel,
        ):
            raise TypeError(
                "model must be an ExplorerModel"
            )

        self.model = model

        # Request-generation counters belong to orchestration rather than
        # explorer state. They stay here until MVC7 moves that orchestration
        # into ExplorerController.
        self._processed_spectrogram_request_id = 0
        self._embedding_cache_request_id = 0
        self._interactive_embedding_request_id = 0

        # On-demand embedding processing is still orchestrated by this
        # coordinator for now. MVC7 will move job ownership and orchestration
        # into ExplorerController; the domain state already lives in the model.
        self.embedding_request = EmbeddingRequest(
            cache=self.model.embedding_cache,
            pipeline=self.model.pipeline,
        )

        # Background execution is a GUI policy, not a domain abstraction.
        # Each independent activity gets its own LatestJobRunner instance,
        # while all instances share the same scheduling implementation.
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

        # Every browsing request receives a monotonically increasing id.
        # Results are accepted only when they match the newest id.
        self._browsing_request_id = 0

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

        # Browsing and processed spectrograms have separate runner instances
        # so one latest-request queue cannot replace the other's work.
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

        # ==================================================
        # Audio playback state
        # ==================================================

        self._audio_sink = None
        self._audio_buffer = None
        self._playback_source = None

        # ==================================================
        # Main window
        # ==================================================

        self.setWindowTitle(
            "Caracal Visualiser"
        )

        self.resize(
            1400,
            850,
        )

        # ==================================================
        # GUI components
        # ==================================================

        self.browsing_view = BrowsingView(
            audio_packets=self.model.audio_packets,
        )

        self.processed_view = ProcessedView(
            embedding_name=self.model.embedding_name,
        )

        self.browsing_view.recording_selected.connect(
            self._recording_selected
        )

        self.browsing_view.chunk_selected.connect(
            self._chunk_selected
        )

        self.browsing_view.play_requested.connect(
            self.play_audio
        )

        self.browsing_view.stop_requested.connect(
            self.stop_browsing_audio
        )

        self.processed_view.buffer_selected.connect(
            self._processed_buffer_selected
        )

        self.processed_view.play_requested.connect(
            self.play_processed_audio
        )

        self.processed_view.stop_requested.connect(
            self.stop_processed_audio
        )

        # ==================================================
        # Dockable explorer panes
        # ==================================================

        # Browsing and processed views are peers: either can be moved to a
        # different dock area or floated into its own top-level window.
        # They are deliberately not closable yet so a pane cannot disappear
        # accidentally during normal exploration.
        self.setDockNestingEnabled(
            True
        )

        self.browsing_dock = self._create_dock(
            title="Source audio",
            object_name="source_audio_dock",
            widget=self.browsing_view,
        )

        self.processed_dock = self._create_dock(
            title="Processed audio and embeddings",
            object_name="processed_audio_dock",
            widget=self.processed_view,
        )

        self.addDockWidget(
            Qt.DockWidgetArea.LeftDockWidgetArea,
            self.browsing_dock,
        )

        self.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea,
            self.processed_dock,
        )

        # Start with the same roughly 50/50 arrangement as the old splitter.
        # The user can immediately drag either dock by its title bar.
        self.resizeDocks(
            [
                self.browsing_dock,
                self.processed_dock,
            ],
            [
                700,
                700,
            ],
            Qt.Orientation.Horizontal,
        )

        # ==================================================
        # Initial state
        # ==================================================

        self._set_browsing_controls_enabled(
            False
        )

        if self.model.audio_packets:
            self._recording_selected(
                0
            )
        else:
            self.browsing_view.set_no_recordings()

        self.statusBar().showMessage(
            "Ready"
        )

    def _create_dock(
        self,
        title: str,
        object_name: str,
        widget,
    ) -> QDockWidget:
        """Wrap an explorer pane in a movable, floatable dock."""

        dock = QDockWidget(
            title,
            self,
        )

        dock.setObjectName(
            object_name
        )

        dock.setAllowedAreas(
            Qt.DockWidgetArea.AllDockWidgetAreas
        )

        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )

        dock.setWidget(
            widget
        )

        return dock

    # ======================================================
    # Pipeline input
    # ======================================================

    @Slot(object)
    def update_data(
        self,
        buffers,
    ):
        """
        Receive one processed batch from GuiPipelineLink.

        Processed data is routed only to ProcessedView.
        """

        buffers = list(
            buffers
        )

        if not buffers:
            return

        # A new pipeline batch replaces the processed-audio inspection
        # state, but must not interrupt source-audio browsing playback.
        self._stop_audio_if_source(
            "processed"
        )

        self.model.processed_buffers = buffers
        self.model.current_processed_buffer_index = 0
        # The processed-audio inspector always shows the live batch,
        # but the PCA view is scoped to the recording selected on the
        # browsing side. Only matching chunks are merged into it.
        selected_recording_id = (
            self._selected_recording_id()
        )

        matching_buffers = [
            buffer
            for buffer
            in self.model.processed_buffers
            if buffer.recording_id
            == selected_recording_id
        ]

        self.processed_view.add_buffers(
            matching_buffers
        )

        self._select_processed_buffer(
            0
        )

        self._update_pipeline_status()

    # ======================================================
    # Playback waveform
    # ======================================================

    def _combined_waveform(
        self,
        buffers,
    ):
        """
        Combine AudioBuffers for playback.

        The caller explicitly supplies the buffer collection so
        browsing state and processed pipeline state cannot be
        mixed accidentally.
        """

        if not buffers:
            return None, None

        sample_rate = (
            buffers[0].sample_rate
        )

        first_waveform = (
            buffers[0].waveform
        )

        if first_waveform.ndim == 1:
            channel_count = 1
        else:
            channel_count = (
                first_waveform.shape[1]
            )

        waveforms = []

        for buffer in buffers:

            if buffer.sample_rate != sample_rate:
                raise ValueError(
                    "Visualiser received buffers "
                    "with different sample rates"
                )

            waveform = np.asarray(
                buffer.waveform,
                dtype=np.float32,
            )

            if waveform.ndim == 1:
                waveform = waveform[
                    :,
                    None,
                ]

            if waveform.shape[1] != channel_count:
                raise ValueError(
                    "Visualiser received buffers "
                    "with different channel counts"
                )

            waveforms.append(
                waveform
            )

        waveform = np.concatenate(
            waveforms,
            axis=0,
        )

        return (
            waveform,
            sample_rate,
        )

    # ======================================================
    # Playback
    # ======================================================

    def play_audio(
        self,
    ):
        """Play the currently loaded source/browsing audio."""

        self._play_buffers(
            buffers=self.model.browsing_buffers,
            volume_percent=(
                self.browsing_view.volume_percent()
            ),
            playback_source="browsing",
            description=(
                f"{len(self.model.browsing_buffers)} browsing buffer(s)"
            ),
        )

    def play_processed_audio(
        self,
    ):
        """Play only the processed buffer currently being inspected."""

        audio_buffer = (
            self._current_processed_buffer()
        )

        if audio_buffer is None:
            return

        if audio_buffer.chunk_index is None:
            description = (
                "processed buffer "
                f"{self.model.current_processed_buffer_index + 1}"
            )
        else:
            description = (
                "processed canonical chunk "
                f"{audio_buffer.chunk_index}"
            )

        self._play_buffers(
            buffers=[audio_buffer],
            volume_percent=(
                self.processed_view.volume_percent()
            ),
            playback_source="processed",
            description=description,
        )

    def _play_buffers(
        self,
        buffers,
        volume_percent,
        playback_source,
        description,
    ):
        """
        Play an explicit collection of buffers through the shared audio
        output device.

        Browsing and processed views have separate controls/state, but
        there is intentionally one QAudioSink: starting one source stops
        the other rather than mixing two inspection streams together.
        """

        waveform, sample_rate = (
            self._combined_waveform(
                buffers
            )
        )

        if waveform is None:
            return

        self.stop_audio()

        # Always make a copy. Playback gain must never alter
        # an AudioBuffer's waveform in place.
        waveform = np.array(
            waveform,
            dtype=np.float32,
            copy=True,
        )

        channel_count = (
            waveform.shape[1]
        )

        waveform *= (
            float(volume_percent)
            / 100.0
        )

        waveform = np.clip(
            waveform,
            -1.0,
            1.0,
        )

        waveform = np.ascontiguousarray(
            waveform
        )

        # ==================================================
        # Qt audio format
        # ==================================================

        audio_format = QAudioFormat()

        audio_format.setSampleRate(
            sample_rate
        )

        audio_format.setChannelCount(
            channel_count
        )

        audio_format.setSampleFormat(
            QAudioFormat.SampleFormat.Float
        )

        device = (
            QMediaDevices.defaultAudioOutput()
        )

        if not device.isFormatSupported(
            audio_format
        ):
            self.statusBar().showMessage(
                f"Unsupported audio format: "
                f"{sample_rate} Hz, "
                f"{channel_count} channel(s)"
            )

            return

        # ==================================================
        # In-memory audio stream
        # ==================================================

        self._audio_buffer = QBuffer(
            self
        )

        self._audio_buffer.setData(
            QByteArray(
                waveform.tobytes()
            )
        )

        self._audio_buffer.open(
            QIODevice.OpenModeFlag.ReadOnly
        )

        # ==================================================
        # Output
        # ==================================================

        self._audio_sink = QAudioSink(
            device,
            audio_format,
            self,
        )

        self._audio_sink.setVolume(
            1.0
        )

        self._audio_sink.start(
            self._audio_buffer
        )

        self._playback_source = (
            playback_source
        )

        duration = (
            len(waveform)
            / sample_rate
        )

        self.statusBar().showMessage(
            f"Playing {description}"
            f" | {duration:.2f}s"
            f" | {sample_rate} Hz"
            f" | volume {volume_percent}%"
        )

    def stop_browsing_audio(
        self,
    ):
        self._stop_audio_if_source(
            "browsing"
        )

    def stop_processed_audio(
        self,
    ):
        self._stop_audio_if_source(
            "processed"
        )

    def _stop_audio_if_source(
        self,
        playback_source,
    ):
        if (
            self._playback_source
            == playback_source
        ):
            self.stop_audio()

    def stop_audio(
        self,
    ):
        if self._audio_sink is not None:
            self._audio_sink.reset()
            self._audio_sink.deleteLater()
            self._audio_sink = None

        if self._audio_buffer is not None:
            self._audio_buffer.close()
            self._audio_buffer.deleteLater()
            self._audio_buffer = None

        self._playback_source = None

    # ======================================================
    # Processed audio inspection
    # ======================================================

    def _current_processed_buffer(
        self,
    ):
        return self.model.current_processed_buffer

    @Slot(int)
    def _processed_buffer_selected(
        self,
        buffer_index: int,
    ):
        self._select_processed_buffer(
            buffer_index
        )

    def _select_processed_buffer(
        self,
        buffer_index: int,
    ):
        """
        Select one processed AudioBuffer from the current pipeline batch
        and request its spectrogram without blocking the GUI thread.
        """

        if not self.model.processed_buffers:
            self.processed_view.clear_processed_audio()
            return

        if not (
            0
            <= buffer_index
            < len(self.model.processed_buffers)
        ):
            return

        self._stop_audio_if_source(
            "processed"
        )

        self.model.current_processed_buffer_index = (
            buffer_index
        )

        audio_buffer = (
            self._current_processed_buffer()
        )

        self.processed_view.set_buffer_selection(
            index=buffer_index,
            buffer_count=len(
                self.model.processed_buffers
            ),
            audio_buffer=audio_buffer,
        )

        # The old image belongs to another processed buffer, so remove
        # it immediately while the new FFT is calculated.
        self.processed_view.clear_spectrogram()

        self._processed_spectrogram_request_id += 1

        request_id = (
            self._processed_spectrogram_request_id
        )

        self.processed_spectrogram_jobs.submit(
            request_id,
            partial(
                calculate_spectrogram,
                waveform=audio_buffer.waveform,
                sample_rate=audio_buffer.sample_rate,
            ),
        )

    @Slot(int, object)
    def _processed_spectrogram_ready(
        self,
        request_id: int,
        result,
    ):
        if (
            request_id
            != self._processed_spectrogram_request_id
        ):
            return

        self.processed_view.set_spectrogram_result(
            result
        )

    @Slot(int, str)
    def _processed_spectrogram_failed(
        self,
        request_id: int,
        message: str,
    ):
        if (
            request_id
            != self._processed_spectrogram_request_id
        ):
            return

        self.processed_view.clear_spectrogram()

        self.statusBar().showMessage(
            "Could not calculate processed spectrogram: "
            f"{message}"
        )

    # ======================================================
    # Recording navigation
    # ======================================================

    def _recording_selected(
        self,
        selector_index: int,
    ):
        """
        Select a logical recording and configure the chunk slider
        for its canonical fixed-duration chunks.
        """

        if (
            selector_index < 0
            or selector_index >= len(
                self.model.audio_packets
            )
        ):
            self.model.current_audio_packet = None
            self.model.current_chunk_index = 0
            self.model.current_num_chunks = 0
            self.model.browsing_buffers = []

            # Invalidate any result already being loaded for the
            # previous selection.
            self._browsing_request_id += 1
            self._embedding_cache_request_id += 1
            self._interactive_embedding_request_id += 1
            self.model.interactive_embeddings_by_key.clear()

            self.processed_view.clear_recording_embeddings(
                "No recording selected"
            )

            self.processed_view.set_selected_embedding_status(
                "Selected chunk embedding: no recording selected"
            )

            self._set_browsing_controls_enabled(
                False
            )

            self.browsing_view.set_chunk_range(
                0
            )

            self.browsing_view.set_chunk_label(
                "No recording selected"
            )

            self.browsing_view.clear_spectrogram()

            return

        packet_index = (
            self.browsing_view.recording_packet_index(
                selector_index
            )
        )

        packet = self.model.audio_packets[
            packet_index
        ]

        self.model.current_audio_packet = packet
        self.model.current_chunk_index = 0
        self.model.interactive_embeddings_by_key.clear()

        self.processed_view.set_selected_embedding_status(
            "Selected chunk embedding: waiting for source audio..."
        )

        if packet.duration is None:
            raise ValueError(
                "Selected AudioPacket has no duration"
            )

        self.model.current_num_chunks = (
            self.model.chunk_grid.chunk_count(
                packet.duration
            )
        )

        if self.model.current_num_chunks <= 0:
            raise ValueError(
                "Selected AudioPacket has no canonical chunks"
            )

        self.browsing_view.set_chunk_range(
            self.model.current_num_chunks
        )

        self._request_cached_embeddings()
        self._update_chunk_label()
        self._request_selected_chunk()

    def _selected_recording_id(
        self,
    ):
        return self.model.selected_recording_id

    def _request_cached_embeddings(
        self,
    ):
        """
        Load every embedding compatible with the active pipeline for
        the recording selected in BrowsingView.

        The cache API remains synchronous; this GUI chooses to execute the
        call through a background job runner.
        """

        recording_id = (
            self._selected_recording_id()
        )

        self._embedding_cache_request_id += 1

        request_id = (
            self._embedding_cache_request_id
        )

        if recording_id is None:
            self.processed_view.clear_recording_embeddings(
                "Selected recording has no recording_id"
            )
            return

        self.processed_view.clear_recording_embeddings(
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
    def _cached_embeddings_loaded(
        self,
        request_id: int,
        embeddings,
    ):
        if (
            request_id
            != self._embedding_cache_request_id
        ):
            return

        self.processed_view.set_recording_embeddings(
            embeddings
        )

        if self.model.interactive_embeddings_by_key:
            self.processed_view.add_embeddings(
                self.model.interactive_embeddings_by_key.values()
            )

        # A live pipeline batch may have arrived while the cache was
        # being scanned. Merge the currently held matching buffers back
        # into the freshly loaded cache view so those points are not lost.
        selected_recording_id = (
            self._selected_recording_id()
        )

        matching_buffers = [
            buffer
            for buffer
            in self.model.processed_buffers
            if buffer.recording_id
            == selected_recording_id
        ]

        if matching_buffers:
            self.processed_view.add_buffers(
                matching_buffers
            )

    @Slot(int, str)
    def _cached_embeddings_failed(
        self,
        request_id: int,
        message: str,
    ):
        if (
            request_id
            != self._embedding_cache_request_id
        ):
            return

        self.processed_view.clear_recording_embeddings(
            "Could not load cached embeddings"
        )

        self.statusBar().showMessage(
            f"Could not load cached embeddings: {message}"
        )

    @Slot(int, object)
    def _interactive_embedding_ready(
        self,
        request_id: int,
        embedding,
    ):
        if (
            request_id
            != self._interactive_embedding_request_id
        ):
            return

        recording_id = embedding.get(
            "recording_id"
        )

        if (
            recording_id
            != self._selected_recording_id()
        ):
            return

        chunk_index = embedding.get(
            "chunk_index"
        )

        key = (
            str(recording_id),
            int(chunk_index),
        )

        self.model.interactive_embeddings_by_key[
            key
        ] = embedding

        self.processed_view.add_embeddings(
            [embedding]
        )

        if embedding.get(
            "loaded_from_cache",
            False,
        ):
            source_text = "loaded from cache"
        else:
            source_text = "computed and cached"

        self.processed_view.set_selected_embedding_status(
            f"Selected chunk embedding: chunk {chunk_index} "
            f"{source_text}"
        )

    @Slot(int, str)
    def _interactive_embedding_failed(
        self,
        request_id: int,
        message: str,
    ):
        if (
            request_id
            != self._interactive_embedding_request_id
        ):
            return

        self.processed_view.set_selected_embedding_status(
            "Selected chunk embedding failed: "
            f"{message}"
        )

    def _chunk_selected(
        self,
        chunk_index: int,
    ):
        """
        Select one canonical chunk within the current recording.
        """

        if self.model.current_audio_packet is None:
            return

        self.model.current_chunk_index = chunk_index

        self._update_chunk_label()
        self._request_selected_chunk()

    def _request_selected_chunk(
        self,
    ):
        """
        Schedule the currently selected chunk for background loading.

        This method runs on the Qt GUI thread, so it deliberately does
        no file I/O. It only updates lightweight GUI state and submits
        a request to the background job runner.
        """

        if self.model.current_audio_packet is None:
            return

        self._stop_audio_if_source(
            "browsing"
        )

        # The previous audio/spectrogram no longer represents the
        # slider position. Clear it immediately while the new request
        # is loading.
        self.model.browsing_buffers = []
        self.browsing_view.clear_spectrogram()

        (
            start_s,
            end_s,
        ) = self.model.chunk_grid.chunk_bounds(
            self.model.current_chunk_index,
            total_duration_s=(
                self.model.current_audio_packet.duration
            ),
        )

        self._browsing_request_id += 1
        self._interactive_embedding_request_id += 1

        request_id = (
            self._browsing_request_id
        )

        self.processed_view.set_selected_embedding_status(
            f"Selected chunk embedding: waiting for chunk "
            f"{self.model.current_chunk_index} audio..."
        )

        # Do not allow playback of the previously loaded chunk while
        # the UI is pointing at a new chunk. The slider itself remains
        # enabled and responsive.
        self._set_browsing_controls_enabled(
            False
        )

        self.statusBar().showMessage(
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

    @Slot(int, object)
    def _browsing_audio_loaded(
        self,
        request_id: int,
        audio_buffer,
    ):
        """
        Accept current source audio and schedule its spectrogram.

        Playback can become available as soon as audio loading finishes.
        The more expensive FFT is requested separately so it never runs
        on the Qt GUI thread.
        """

        if (
            request_id
            != self._browsing_request_id
        ):
            return

        self.model.browsing_buffers = [
            audio_buffer
        ]

        self._set_browsing_controls_enabled(
            True
        )

        self.statusBar().showMessage(
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

        embedding_request_id = (
            self._interactive_embedding_request_id
        )

        self.processed_view.set_selected_embedding_status(
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
    def _browsing_spectrogram_ready(
        self,
        request_id: int,
        result,
    ):
        """
        Draw a spectrogram only when it still belongs to the current
        browsing selection.
        """

        if (
            request_id
            != self._browsing_request_id
        ):
            return

        self.browsing_view.set_spectrogram_result(
            result
        )

        self.statusBar().showMessage(
            f"Loaded chunk {self.model.current_chunk_index}"
        )

    @Slot(int, str)
    def _browsing_spectrogram_failed(
        self,
        request_id: int,
        message: str,
    ):
        """
        Report a current spectrogram failure without discarding the
        successfully loaded audio.
        """

        if (
            request_id
            != self._browsing_request_id
        ):
            return

        self.browsing_view.clear_spectrogram()

        self.statusBar().showMessage(
            f"Audio loaded, but spectrogram failed: {message}"
        )

    @Slot(int, str)
    def _browsing_audio_failed(
        self,
        request_id: int,
        message: str,
    ):
        """
        Report an error only when it belongs to the current request.
        """

        if (
            request_id
            != self._browsing_request_id
        ):
            return

        self.model.browsing_buffers = []
        self.browsing_view.clear_spectrogram()

        self._set_browsing_controls_enabled(
            False
        )

        self.processed_view.set_selected_embedding_status(
            "Selected chunk embedding: source audio unavailable"
        )

        self.statusBar().showMessage(
            f"Could not load chunk: {message}"
        )

    def _update_chunk_label(
        self,
    ):
        if self.model.current_audio_packet is None:
            self.browsing_view.set_chunk_label(
                "No recording selected"
            )

            self.browsing_view.clear_spectrogram()

            return

        (
            start_s,
            end_s,
        ) = self.model.chunk_grid.chunk_bounds(
            self.model.current_chunk_index,
            total_duration_s=(
                self.model.current_audio_packet.duration
            ),
        )

        self.browsing_view.set_chunk_label(
            f"{self.model.current_chunk_index} "
            f"({start_s:.1f}s - {end_s:.1f}s)"
        )

    # ======================================================
    # Controls
    # ======================================================

    def _set_browsing_controls_enabled(
        self,
        enabled,
    ):
        self.browsing_view.set_browsing_controls_enabled(
            enabled
        )

    # ======================================================
    # Processed status
    # ======================================================

    def _update_pipeline_status(
        self,
    ):
        if not self.model.processed_buffers:
            self.processed_view.set_status(
                "Waiting for processed audio..."
            )

            return

        first_buffer = (
            self.model.processed_buffers[0]
        )

        sample_rate = (
            first_buffer.sample_rate
        )

        first_waveform = np.asarray(
            first_buffer.waveform
        )

        if first_waveform.ndim == 1:
            channel_count = 1
        else:
            channel_count = (
                first_waveform.shape[1]
            )

        total_samples = sum(
            len(buffer.waveform)
            for buffer in self.model.processed_buffers
        )

        duration = (
            total_samples
            / sample_rate
        )

        total_embeddings = len(
            self.processed_view
            .embedding_visualiser
            .embedding_vectors
        )

        self.processed_view.set_status(
            f"Latest batch: {len(self.model.processed_buffers)} processed buffer(s)"
            f" | {duration:.2f}s"
            f" | {sample_rate} Hz"
            f" | {channel_count} channel(s)"
            f" | {total_embeddings} selected-recording embeddings"
        )

    # ======================================================
    # Cleanup
    # ======================================================

    def shutdown(
        self,
    ):
        """Release explorer playback and background resources once."""

        if self._is_shutdown:
            return

        self._is_shutdown = True

        self.stop_audio()

        self.browsing_audio_jobs.shutdown()
        self.browsing_spectrogram_jobs.shutdown()
        self.processed_spectrogram_jobs.shutdown()
        self.embedding_cache_jobs.shutdown()
        self.embedding_request_jobs.shutdown()
        self.processed_view.shutdown()

    def closeEvent(
        self,
        event,
    ):
        self.shutdown()

        super().closeEvent(
            event
        )
