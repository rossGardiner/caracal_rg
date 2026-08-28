from PySide6.QtCore import QObject, Signal, Qt

from src.PipelineLink import PipelineLink


class _GuiSignals(QObject):
    data_ready = Signal(object)


class GuiPipelineLink(PipelineLink):
    INCLUDE_IN_PIPELINE_CONFIG = False

    def __init__(self, batch_size=8):
        super().__init__()

        self._signals = _GuiSignals()

        self.batch_size = batch_size
        self._batch = []

    def register_gui(self, callback):
        self._signals.data_ready.connect(
            callback,
            Qt.ConnectionType.QueuedConnection,
        )

    def gui_data(self, packet):
        return packet

    def next_audio(self, packet):
        data = self.gui_data(packet)

        if data is not None:
            self._batch.append(data)

            if len(self._batch) >= self.batch_size:
                batch = tuple(self._batch)
                self._batch.clear()

                self._signals.data_ready.emit(batch)

        super().next_audio(packet)

    def configuration_parameters(self):
        return {}
