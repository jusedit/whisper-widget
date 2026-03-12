"""Lightweight splash screen — imports only PyQt6, no torch/onnxruntime.

Started by launch.bat BEFORE main.py. Shows the loading animation
instantly while main.py spends ~1.5s importing torch. When main.py
signals readiness via a file, this process plays the success animation
and exits.
"""

import sys
import ctypes
from pathlib import Path

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from overlay import NotchOverlay

# Single-instance guard — prevent duplicate splash windows
_splash_mutex = ctypes.windll.kernel32.CreateMutexW(None, True, "WhisperWidget_Presplash")
if ctypes.windll.kernel32.GetLastError() == 183:  # ERROR_ALREADY_EXISTS
    sys.exit(0)

SIGNAL_FILE = Path(__file__).parent / "debug" / ".splash_ready"


def main():
    # Clean up stale signal
    SIGNAL_FILE.parent.mkdir(exist_ok=True)
    SIGNAL_FILE.unlink(missing_ok=True)

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    overlay = NotchOverlay()
    overlay.show_loading()

    def check_signal():
        if SIGNAL_FILE.exists():
            SIGNAL_FILE.unlink(missing_ok=True)
            timer.stop()
            # Play the green checkmark animation, then exit
            overlay.hide_loading()
            QTimer.singleShot(1200, app.quit)

    timer = QTimer()
    timer.timeout.connect(check_signal)
    timer.start(100)

    # Safety: quit after 30s even if main.py never signals
    QTimer.singleShot(30000, app.quit)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
