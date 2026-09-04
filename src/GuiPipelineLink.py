''' GuiPipelineLink
This is a pipeline link which is not config related. 
This link provides an interface with Qt. 
'''

from PySide6.QtCore import (
    QObject,
    Signal,
    Qt,
    QSemaphore,
)

from src.PipelineLink import PipelineLink


class _GuiSignals(QObject):
    batch_ready = Signal(object)


class GuiPipelineLink(PipelineLink):
    """
    Pipeline link which collects a fixed number of packets,
    displays them in the GUI, then pauses until the user
    requests the next batch.

    No packets are passed downstream while the current batch
    is being inspected.
    """

    INCLUDE_IN_PIPELINE_CONFIG = False

    def __init__(
        self,
        control_window,
        buffer_count=10,
    ):
        super().__init__()

        if buffer_count <= 0:
            raise ValueError(
                "buffer_count must be greater than zero"
            )

        self.control_window = control_window
        self.buffer_count = buffer_count

        self.buffers = []

        #
        # Starts at zero:
        #
        # acquire() waits until the GUI calls release().
        #
        self._continue = QSemaphore(0)

        #
        # Signals are used so QWidget updates happen on
        # the Qt GUI thread.
        #
        self._signals = _GuiSignals()

        self._signals.batch_ready.connect(
            self.control_window.update_data,
            Qt.ConnectionType.QueuedConnection,
        )

        #
        # The control_window emits this when Next is pressed.
        #
        self.control_window.next_requested.connect(
            self.continue_pipeline
        )

    def next_audio(self, packet):
        """
        Collect packets until a complete visualisation batch exists.

        Once full:
            1. display batch
            2. wait for user
            3. forward entire batch downstream
            4. begin collecting next batch
        """

        self.buffers.append(packet)

        #
        # Still collecting.
        #
        if len(self.buffers) < self.buffer_count:
            return

        #
        # Complete batch.
        #
        batch = self.buffers
        self.buffers = []

        #
        # Tell GUI to display it.
        #
        self._signals.batch_ready.emit(
            batch
        )

        #
        # STOP HERE until Next is pressed.
        #
        self._continue.acquire()

        #
        # User requested the next batch.
        #
        # Send the displayed buffers downstream,
        # preserving their original order.
        #
        for packet in batch:
            super().next_audio(packet)

    def continue_pipeline(self):
        """
        Release the pipeline after the user presses Next.
        """

        self._continue.release()

    def configuration_parameters(self):
        return {}
