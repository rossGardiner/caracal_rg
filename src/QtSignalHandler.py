import signal
from PySide6.QtCore import QTimer
'''
Defines a simple handler method which allows QApplications to periodically check for SIGINT
'''

class QtSignalHandler:
    def __init__(self, app, interval_ms=100):
        signal.signal(signal.SIGINT, lambda *_: app.quit())

        self.timer = QTimer()
        self.timer.timeout.connect(lambda: None)
        self.timer.start(interval_ms)

