"""Auto-download Parakeet TDT v3 ONNX model from HuggingFace."""

import os
import threading
import urllib.request
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal, QObject
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QProgressBar, QPushButton, QApplication,
)
from PyQt6.QtGui import QFont

REPO = "istupakov/parakeet-tdt-0.6b-v3-onnx"
BASE_URL = f"https://huggingface.co/{REPO}/resolve/main"

FP32_FILES = {
    "encoder-model.onnx": 41_000_000,             # ~42 MB (graph structure)
    "encoder-model.onnx.data": 2_440_000_000,      # ~2.44 GB (weights)
    "decoder_joint-model.onnx": 72_000_000,        # ~73 MB
    "vocab.txt": 90_000,                           # ~90 KB
}

INT8_FILES = {
    "encoder-model.int8.onnx": 652_000_000,
    "decoder_joint-model.int8.onnx": 18_000_000,
    "vocab.txt": 90_000,
}

QUALITY_FILES = {"int8": INT8_FILES, "fp32": FP32_FILES}
QUALITY_LABELS = {
    "int8": ("Fast (INT8)", "~670 MB, slightly lower quality"),
    "fp32": ("Best (FP32)", "~2.5 GB, highest quality"),
}


def _check_files(model_dir: Path, files: dict) -> bool:
    for filename, min_size in files.items():
        path = model_dir / filename
        if not path.exists():
            return False
        if path.stat().st_size < min_size * 0.9:
            return False
    return True


def models_exist(model_dir: Path, quality: str | None = None) -> bool:
    """Check if model files exist. If quality given, check that specific set."""
    if quality:
        return _check_files(model_dir, QUALITY_FILES[quality])
    return _check_files(model_dir, FP32_FILES) or _check_files(model_dir, INT8_FILES)


class _DownloadSignals(QObject):
    progress = pyqtSignal(str, float)  # filename, overall fraction 0.0-1.0
    file_done = pyqtSignal(str)
    error = pyqtSignal(str)
    all_done = pyqtSignal()


class DownloadDialog(QDialog):
    """Modal dialog that downloads model files with progress."""

    def __init__(self, model_dir: Path, quality: str = "int8", parent=None):
        super().__init__(parent)
        self._model_dir = model_dir
        self._quality = quality
        self._model_files = QUALITY_FILES[quality]
        self._signals = _DownloadSignals()
        self._cancelled = False
        self._thread = None

        self.setWindowTitle("Whisper Widget")
        self.setFixedSize(480, 200)
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.CustomizeWindowHint
        )

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(12)

        title = QLabel("Downloading speech model")
        title.setFont(QFont("Segoe UI", 13, QFont.Weight.DemiBold))
        layout.addWidget(title)

        self._status = QLabel("Preparing download...")
        self._status.setFont(QFont("Segoe UI", 10))
        self._status.setStyleSheet("color: #888;")
        layout.addWidget(self._status)

        self._progress = QProgressBar()
        self._progress.setMinimum(0)
        self._progress.setMaximum(10000)  # 0.01% resolution for large downloads
        self._progress.setTextVisible(False)
        self._progress.setFixedHeight(8)
        self._progress.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 4px;
                background: #2c2c2e;
            }
            QProgressBar::chunk {
                border-radius: 4px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0a84ff, stop:1 #5ac8fa);
            }
        """)
        layout.addWidget(self._progress)

        total_mb = sum(self._model_files.values()) / 1024 / 1024
        label = QUALITY_LABELS[self._quality][0]
        self._size_label = QLabel(f"Model: {label} — this only happens once (~{total_mb:.0f} MB)")
        self._size_label.setFont(QFont("Segoe UI", 9))
        self._size_label.setStyleSheet("color: #666;")
        layout.addWidget(self._size_label)

        layout.addStretch()

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setFixedWidth(100)
        self._cancel_btn.clicked.connect(self._on_cancel)
        layout.addWidget(self._cancel_btn, alignment=Qt.AlignmentFlag.AlignRight)

        # Dark theme
        self.setStyleSheet("""
            QDialog { background: #1c1c1e; }
            QLabel { color: #ffffff; }
            QPushButton {
                background: #2c2c2e; color: #ffffff; border: 1px solid #3a3a3c;
                border-radius: 6px; padding: 6px 16px;
                font-family: "Segoe UI"; font-size: 10pt;
            }
            QPushButton:hover { background: #3a3a3c; }
        """)

    def _connect_signals(self):
        self._signals.progress.connect(self._on_progress)
        self._signals.file_done.connect(self._on_file_done)
        self._signals.error.connect(self._on_error)
        self._signals.all_done.connect(self._on_all_done)

    def showEvent(self, event):
        super().showEvent(event)
        self._start_download()

    def _start_download(self):
        self._thread = threading.Thread(target=self._download_all, daemon=True)
        self._thread.start()

    def _download_all(self):
        self._model_dir.mkdir(parents=True, exist_ok=True)

        total_size = sum(self._model_files.values())
        bytes_done = 0

        for filename, expected_size in self._model_files.items():
            if self._cancelled:
                return

            dest = self._model_dir / filename
            part = self._model_dir / f"{filename}.part"

            # Skip if already downloaded
            if dest.exists() and dest.stat().st_size >= expected_size * 0.9:
                bytes_done += expected_size
                self._signals.file_done.emit(filename)
                continue

            url = f"{BASE_URL}/{filename}"
            existing = part.stat().st_size if part.exists() else 0

            try:
                req = urllib.request.Request(url)
                if existing > 0:
                    req.add_header("Range", f"bytes={existing}-")

                with urllib.request.urlopen(req, timeout=30) as resp:
                    mode = "ab" if existing > 0 else "wb"
                    file_bytes = existing

                    with open(part, mode) as f:
                        while True:
                            if self._cancelled:
                                return
                            data = resp.read(256 * 1024)
                            if not data:
                                break
                            f.write(data)
                            file_bytes += len(data)
                            fraction = (bytes_done + file_bytes) / total_size
                            self._signals.progress.emit(filename, fraction)

                # Rename .part to final
                if dest.exists():
                    dest.unlink()
                part.rename(dest)
                bytes_done += file_bytes
                self._signals.file_done.emit(filename)

            except Exception as e:
                self._signals.error.emit(f"Failed to download {filename}: {e}")
                return

        self._signals.all_done.emit()

    def _on_progress(self, filename: str, fraction: float):
        self._progress.setValue(int(fraction * 10000))
        pct = fraction * 100
        total_mb = sum(self._model_files.values()) / 1024 / 1024
        done_mb = fraction * total_mb
        short_name = filename.replace(".onnx.data", " weights").replace(".onnx", "")
        short_name = short_name.split("-")[0] if "-" in short_name else short_name
        self._status.setText(
            f"Downloading {short_name}... {done_mb:.0f} / {total_mb:.0f} MB ({pct:.0f}%)"
        )

    def _on_file_done(self, filename: str):
        pass

    def _on_error(self, error: str):
        self._status.setText(error)
        self._status.setStyleSheet("color: #ff453a;")
        self._cancel_btn.setText("Close")

    def _on_all_done(self):
        self.accept()

    def _on_cancel(self):
        self._cancelled = True
        self.reject()
