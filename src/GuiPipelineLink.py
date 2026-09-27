''' GuiPipelineLink
This is a pipeline link which is not config related.
This link provides a non-blocking interface with Qt.
'''

import threading

from PySide6.QtCore import (
    QObject,
    Signal,
    Slot,
    Qt,
)

from src.PipelineLink import PipelineLink


class _GuiSignals(QObject):
    batch_available = Signal()


class _GuiBatchDispatcher(QObject):
    """
    GUI-thread dispatcher for the latest processed batch.

    GuiPipelineLink itself is driven by the background pipeline thread.
    This QObject is created on the Qt thread, so its queued slot can safely
    hand the newest available batch to the explorer view.
    """

    def __init__(
        self,
        pipeline_link,
        explorer_view,
    ):
        super().__init__()

        self.pipeline_link = pipeline_link
        self.explorer_view = explorer_view

    @Slot()
    def deliver_latest_batch(
        self,
    ):
        batch = (
            self.pipeline_link
            .take_latest_batch_for_gui()
        )

        if batch is None:
            return

        self.explorer_view.update_data(
            batch
        )


class GuiPipelineLink(PipelineLink):
    """
    Non-blocking observer link between the processing pipeline and Qt.

    The link groups processed AudioBuffers into fixed-size inspection
    batches. Once a batch is complete it publishes the newest batch to the
    GUI, then immediately continues the callback pipeline.

    GUI delivery is deliberately coalesced: while one Qt notification is
    already pending, newer completed batches replace the older pending batch
    instead of creating an unbounded queue of GUI updates. The cache remains
    the durable output of batch precomputation; the GUI only observes the
    latest useful inspection state.
    """

    INCLUDE_IN_PIPELINE_CONFIG = False

    def __init__(
        self,
        explorer_view,
        buffer_count=10,
    ):
        super().__init__()

        if buffer_count <= 0:
            raise ValueError(
                "buffer_count must be greater than zero"
            )

        self.explorer_view = explorer_view
        self.buffer_count = buffer_count

        self.buffers = []

        # The processing thread writes these fields while the GUI thread
        # consumes them. Only one queued Qt notification is allowed at a
        # time, so a fast precompute run cannot flood the Qt event queue.
        self._gui_batch_lock = threading.Lock()
        self._latest_gui_batch = None
        self._gui_notification_pending = False

        self._signals = _GuiSignals()

        # This dispatcher is constructed on the GUI thread. The queued
        # connection therefore guarantees that the explorer view is touched only
        # from Qt's GUI thread.
        self._dispatcher = _GuiBatchDispatcher(
            pipeline_link=self,
            explorer_view=self.explorer_view,
        )

        self._signals.batch_available.connect(
            self._dispatcher.deliver_latest_batch,
            Qt.ConnectionType.QueuedConnection,
        )

    def next_audio(
        self,
        packet,
    ):
        """
        Observe processed packets without applying GUI backpressure.

        Every packet is forwarded downstream immediately. In parallel, once
        ``buffer_count`` packets have accumulated for inspection, that batch
        is published as the newest GUI state and collection starts again.

        There is no wait for user input and no downstream batching delay.
        """

        self.buffers.append(
            packet
        )

        # This link is an observer, not a batching transport. Preserve normal
        # callback timing by forwarding every packet immediately, regardless
        # of when the GUI inspection batch becomes complete.
        super().next_audio(
            packet
        )

        if len(self.buffers) < self.buffer_count:
            return

        batch = self.buffers
        self.buffers = []

        self._publish_latest_batch(
            batch
        )

    def _publish_latest_batch(
        self,
        batch,
    ):
        """
        Store the newest completed batch and schedule at most one Qt event.
        """

        should_notify = False

        with self._gui_batch_lock:
            self._latest_gui_batch = list(
                batch
            )

            if not self._gui_notification_pending:
                self._gui_notification_pending = True
                should_notify = True

        if should_notify:
            self._signals.batch_available.emit()

    def take_latest_batch_for_gui(
        self,
    ):
        """
        Return the newest pending GUI batch.

        Called only by the GUI-thread dispatcher. Clearing the notification
        flag here allows the pipeline thread to schedule one new notification
        while the explorer view renders the batch that was just taken.
        """

        with self._gui_batch_lock:
            batch = self._latest_gui_batch
            self._latest_gui_batch = None
            self._gui_notification_pending = False

        return batch

    def configuration_parameters(
        self,
    ):
        return {}
