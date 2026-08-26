from PySide6.QtCore import QObject, Signal, Qt

from src.PipelineLink import PipelineLink


class _GuiSignals(QObject):
    """
    Internal Qt object responsible for signals.
    """
    data_ready = Signal(object)


class GuiPipelineLink(PipelineLink):
    """
    Pipeline link which exposes pipeline data to a Qt GUI.

    Data continues through the pipeline normally, but is also emitted
    to any connected GUI component.
    """

    # The GUI should not affect the processing configuration/hash.
    INCLUDE_IN_PIPELINE_CONFIG = False

    def __init__(self):
        super().__init__()

        self._signals = _GuiSignals()

    def register_gui(self, callback):
        """
        Register a GUI callback.

        The callback will always be invoked through Qt's event queue,
        making it safe for pipeline processing to happen on another
        thread.
        """
        self._signals.data_ready.connect(
            callback,
            Qt.ConnectionType.QueuedConnection,
        )

    def gui_data(self, packet):
        """
        Return the data that should be sent to the GUI.

        Subclasses can override this if they want to transform/extract
        data before it reaches the GUI.
        """
        return packet

    def next_audio(self, packet):
        """
        Receive pipeline data, notify the GUI, then continue forwarding.
        """
        data = self.gui_data(packet)

        if data is not None:
            self._signals.data_ready.emit(data)

        # Continue the normal pipeline.
        super().next_audio(packet)

    def configuration_parameters(self):
        return {}
