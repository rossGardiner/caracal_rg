"""Coordinate the active-learning review queue with the standalone pop-out."""

from functools import partial

from PySide6.QtCore import QObject, Slot

from src.LatestJobRunner import LatestJobRunner
from src.QuickReviewAudio import ReviewChunk, load_review_chunk


class QuickReviewController(QObject):
    """Use ActiveLearningController's existing label/skip actions as-is.

    Audio is fetched from the compatible Explorer *dataset* without changing
    Explorer selection, focus, or playback.  Selection changes invalidate old
    worker results, so late reads cannot show/label a different candidate.
    """

    def __init__(self, *, view, active_learning_controller, explorer_controller,
                 parent=None):
        super().__init__(parent)
        self.view = view
        self.active_learning = active_learning_controller
        self.explorer = explorer_controller
        self._request_id = 0
        self._selection_key = None
        self._recording_snapshot_id = None
        self._recordings = {}

        self.jobs = LatestJobRunner(name="quick-review-audio", parent=self)
        self.jobs.ready.connect(self._chunk_ready)
        self.jobs.failed.connect(self._chunk_failed)
        self.view.positive_requested.connect(self.label_positive)
        self.view.negative_requested.connect(self.label_negative)
        self.view.skip_requested.connect(self.skip)
        self.view.back_requested.connect(self.back)
        self.view.closed.connect(self._window_closed)
        self.active_learning.workspace_state_changed.connect(self.refresh)
        self.explorer.pipeline_activation_succeeded.connect(
            lambda *_args: self.refresh(force=True)
        )

    @Slot()
    def show(self):
        self.view.show()
        self.view.raise_()
        self.view.activateWindow()
        self.refresh(force=True)

    def refresh(self, *, force=False):
        if not self.view.isVisible():
            return
        model = self.active_learning.model
        self.view.set_back_enabled(
            model.is_ready and not model.round_running
            and model.can_go_previous_candidate
        )
        candidate = model.current_candidate
        if candidate is None or model.round_running or not model.is_ready:
            self._reset_selection()
            if model.round_running:
                status = "Learning round running…"
            elif model.review_queue and candidate is None:
                status = "End of review queue. Return to Active Learning for another round."
            else:
                status = "Prepare a classifier and run a learning round to review candidates."
            self.view.set_candidate_summary("Quick Review")
            self.view.clear_chunk(status, can_skip=False)
            return

        position = model.review_position
        label = model.current_candidate_label
        if label is not None:
            disposition = f"Label: {label.name.title()}"
        elif model.current_candidate_is_skipped:
            disposition = "Skipped"
        else:
            disposition = "Unreviewed"
        self.view.set_candidate_summary(
            f"Candidate {position[0]} / {position[1]}  ·  "
            f"{candidate.ref.recording_id}  ·  chunk {candidate.ref.chunk_index}  ·  "
            f"score {candidate.probability:.3f}  ·  {disposition}"
        )
        space, _ = self.explorer.current_embedding_selection()
        explorer_model = self.explorer.model
        key = (
            model.selected_classifier_id, id(model.last_round_result),
            model.embedding_space, candidate.ref,
            id(explorer_model.audio_packets),
        )
        if key == self._selection_key and not force:
            return

        self._request_id += 1
        self._selection_key = key
        self.view.clear_chunk("Loading spectrogram and audio…", can_skip=True)
        if space != model.embedding_space or explorer_model.audio_reader is None:
            self.view.clear_chunk(
                "Activate the classifier's matching pipeline in Explorer "
                "once to make the review audio available.",
                can_skip=True,
            )
            return

        if self._recording_snapshot_id != id(explorer_model.audio_packets):
            self._recording_snapshot_id = id(explorer_model.audio_packets)
            self._recordings = {
                str(packet.recording_id): packet
                for packet in explorer_model.audio_packets
                if packet.recording_id is not None
            }
        packet = self._recordings.get(candidate.ref.recording_id)
        if packet is None:
            self.view.clear_chunk(
                "This review recording is not present in the active dataset.",
                can_skip=True,
            )
            return

        self.jobs.submit(
            self._request_id,
            partial(
                load_review_chunk,
                ref=candidate.ref, packet=packet,
                chunk_grid=explorer_model.chunk_grid,
                audio_reader=explorer_model.audio_reader,
            ),
        )

    @Slot(int, object)
    def _chunk_ready(self, request_id, chunk):
        if request_id != self._request_id or not self.view.isVisible():
            return
        candidate = self.active_learning.model.current_candidate
        if (not isinstance(chunk, ReviewChunk) or candidate is None
                or candidate.ref != chunk.ref):
            return
        self.view.set_chunk(chunk)

    @Slot(int, str)
    def _chunk_failed(self, request_id, message):
        if request_id != self._request_id or not self.view.isVisible():
            return
        self.view.clear_chunk(
            f"Could not load review audio: {message}", can_skip=True
        )

    def _can_label_loaded(self):
        model = self.active_learning.model
        candidate = model.current_candidate
        return (
            self.view.isVisible() and model.is_ready
            and not model.round_running and candidate is not None
            and self.view.loaded_ref == candidate.ref
        )

    @Slot()
    def label_positive(self):
        if self._can_label_loaded():
            self.active_learning.label_current_candidate_positive()

    @Slot()
    def label_negative(self):
        if self._can_label_loaded():
            self.active_learning.label_current_candidate_negative()

    @Slot()
    def skip(self):
        if self.view.isVisible():
            # Unlike merely stepping past an already-labelled queue entry,
            # Quick Review Skip means "leave this example unlabelled".
            self.active_learning.skip_current_candidate_as_unlabelled()

    @Slot()
    def back(self):
        model = self.active_learning.model
        if (self.view.isVisible() and model.is_ready
                and not model.round_running and model.can_go_previous_candidate):
            # Do not clear a label just because the user revisits an example.
            self.active_learning.previous_candidate()

    @Slot()
    def _window_closed(self):
        self._reset_selection()

    def _reset_selection(self):
        self._request_id += 1
        self._selection_key = None

    def shutdown(self):
        self._reset_selection()
        self.view.stop_audio()
        self.jobs.shutdown()
