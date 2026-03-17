"""
Whisper Widget
Voice-to-text using Parakeet TDT v3 ONNX model.
Ctrl+Space to toggle recording. Transcription auto-pastes at cursor.
"""

import sys
import os
import re
import logging
import subprocess
import threading
import time
import atexit
import winsound
from pathlib import Path

_startup_t0 = time.perf_counter()

# --- File logging (pythonw.exe has no console) ---
_log_dir = Path(__file__).parent / "debug"
_log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(_log_dir / "app.log", encoding="utf-8"),
        logging.StreamHandler(sys.stderr),  # also console if available
    ],
)
log = logging.getLogger("whisper")

# --- Single-instance guard (before heavy imports to fail fast) ---
import ctypes
_instance_mutex = ctypes.windll.kernel32.CreateMutexW(None, True, "WhisperWidget_SingleInstance")
if ctypes.windll.kernel32.GetLastError() == 183:  # ERROR_ALREADY_EXISTS
    log.info("Another instance already running — exiting")
    sys.exit(0)

# Fix onnxruntime DLL loading in venv: add capi dir and import BEFORE PyQt6
if sys.platform == "win32":
    import importlib.util
    _spec = importlib.util.find_spec("onnxruntime")
    if _spec and _spec.submodule_search_locations:
        _capi_dir = os.path.join(list(_spec.submodule_search_locations)[0], "capi")
        if os.path.isdir(_capi_dir):
            os.add_dll_directory(_capi_dir)

# onnxruntime MUST be imported before PyQt6 (Windows DLL conflict)
_t_imports = time.perf_counter()
import numpy as np
import onnxruntime
import pyperclip
import pyautogui
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QSize
from PyQt6.QtWidgets import QApplication, QSystemTrayIcon, QMenu
from PyQt6.QtGui import QIcon, QAction, QPixmap, QPainter, QColor, QPen, QPainterPath, QFont

from config import Config, __version__
from assets import ensure_assets, get_sound_path
from model_downloader import models_exist, DownloadDialog
from overlay import NotchOverlay
from settings_dialog import SettingsDialog
from perf_logger import PerfTimer, log_memory

log.info("PERF app.imports: %.0fms", (time.perf_counter() - _t_imports) * 1000)
log_memory("after_imports")

# --- PID file for launcher update detection ---
_pid_file = Path(os.environ.get("LOCALAPPDATA", "")) / "WhisperWidget" / ".pid"


def _write_pid():
    try:
        _pid_file.parent.mkdir(parents=True, exist_ok=True)
        _pid_file.write_text(str(os.getpid()))
    except OSError:
        pass


def _remove_pid():
    try:
        _pid_file.unlink(missing_ok=True)
    except OSError:
        pass


atexit.register(_remove_pid)


class TranscriptionSignals(QObject):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    hotkey_pressed = pyqtSignal()
    model_loaded = pyqtSignal()
    model_failed = pyqtSignal(str)
    file_transcribed = pyqtSignal(str)


class HotkeyListener(threading.Thread):
    """Listens for global hotkey using Windows API."""

    def __init__(self, callback, modifiers=0x0002, vk=0x20):
        super().__init__(daemon=True)
        self._callback = callback
        self._modifiers = modifiers
        self._vk = vk
        self._running = True

    def run(self):
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        HOTKEY_ID = 1

        if not user32.RegisterHotKey(None, HOTKEY_ID, self._modifiers, self._vk):
            # Try fallback with added Shift
            if not user32.RegisterHotKey(None, HOTKEY_ID, self._modifiers | 0x0004, self._vk):
                log.error("Could not register hotkey!")
                return

        msg = wintypes.MSG()
        try:
            while self._running:
                ret = user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 0x0001)
                if ret:
                    if msg.message == 0x0312 and msg.wParam == HOTKEY_ID:
                        self._callback()
                else:
                    time.sleep(0.01)
        finally:
            user32.UnregisterHotKey(None, HOTKEY_ID)

    def stop(self):
        self._running = False


_tray_icon_cache: dict[str, QIcon] = {}


def _generate_tray_icon(state: str = "ready") -> QIcon:
    """Generate a tray icon programmatically (cached per state)."""
    cached = _tray_icon_cache.get(state)
    if cached is not None:
        return cached
    size = 64
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor(0, 0, 0, 0))

    p = QPainter(pixmap)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)

    # Draw microphone shape
    mic_color = {
        "ready": QColor(255, 255, 255),
        "recording": QColor(239, 68, 68),
        "processing": QColor(99, 102, 241),
        "done": QColor(52, 199, 89),
    }.get(state, QColor(255, 255, 255))

    pen = QPen(mic_color)
    pen.setWidthF(4)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)

    # Mic body (rounded rect)
    p.drawRoundedRect(22, 10, 20, 28, 10, 10)

    # Mic arc
    path = QPainterPath()
    path.moveTo(16, 30)
    path.cubicTo(16, 48, 48, 48, 48, 30)
    p.drawPath(path)

    # Mic stand
    p.drawLine(32, 48, 32, 56)
    p.drawLine(24, 56, 40, 56)

    # State indicator dot
    if state != "ready":
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(mic_color)
        p.drawEllipse(46, 4, 14, 14)

    p.end()
    icon = QIcon(pixmap)
    _tray_icon_cache[state] = icon
    return icon


class VoiceWidget:
    """Main application controller."""

    def __init__(self):
        _t0 = time.perf_counter()

        self.app = QApplication(sys.argv)
        self.app.setQuitOnLastWindowClosed(False)
        self.app.setApplicationName("Whisper Widget")

        log.info("=" * 40)
        log.info("Whisper Widget v%s starting up (pid=%d)", __version__, os.getpid())
        _write_pid()

        # Minimal init — presplash.pyw handles the loading animation,
        # we just need the overlay for recording/transcribing states later
        self._config = Config()
        self.overlay = NotchOverlay()

        # State defaults (needed before _finish_init runs)
        self._transcriber = None
        self._transcriber_loading = False
        self._recording = False
        self._hotkey = None
        self._partial_texts: list[str] = []
        self._partial_lock = threading.Lock()
        self._pending_chunks = 0
        self._record_start_time = 0.0
        self._total_chunked_audio = 0.0
        self._pending_level = 0.0
        self._drop_mode = False
        self._drop_timeout: QTimer | None = None

        # Signals (must exist before event loop processes anything)
        self._signals = TranscriptionSignals()
        self._signals.finished.connect(self._on_transcription_done)
        self._signals.error.connect(self._on_transcription_error)
        self._signals.hotkey_pressed.connect(self.toggle_recording)
        self._signals.model_loaded.connect(self._on_model_loaded)
        self._signals.model_failed.connect(self._on_model_failed)
        self._signals.file_transcribed.connect(self._on_file_transcription_done)

        # Connect overlay file-drop signals
        self.overlay.file_dropped.connect(self._on_file_dropped)
        self.overlay.copy_clicked.connect(self._on_copy_clicked)
        self.overlay.close_clicked.connect(self._on_close_drop)

        log.info("PERF app.init: %.0fms", (time.perf_counter() - _t0) * 1000)

    def _finish_init(self):
        """Complete setup after event loop is running (splash is visible)."""
        _t0 = time.perf_counter()

        # Set app icon — prefer .ico file, fall back to generated
        with PerfTimer("app.finish_init.icon"):
            ico_path = Path(__file__).parent / "favicon.ico"
            if ico_path.exists():
                self._app_icon = QIcon(str(ico_path))
            else:
                self._app_icon = _generate_tray_icon("ready")
            self.app.setWindowIcon(self._app_icon)

        # Generate assets (sounds)
        with PerfTimer("app.finish_init.assets"):
            ensure_assets()

        # Check model download
        with PerfTimer("app.finish_init.model_check"):
            quality = self._config.model_quality
            model_dir = self._config.get_model_dir()
            log.info(f"Model dir: {model_dir}, quality: {quality}")
            log.info(f"Models exist: {models_exist(model_dir, quality)}")
            if not models_exist(model_dir, quality):
                from config import DEFAULT_MODEL_DIR
                model_dir = DEFAULT_MODEL_DIR
                log.info(f"Models missing, download {quality} to: {model_dir}")
                dlg = DownloadDialog(model_dir, quality=quality)
                if dlg.exec() != dlg.DialogCode.Accepted:
                    self.app.quit()
                    return

        # Audio
        with PerfTimer("app.finish_init.recorder"):
            from recorder import AudioRecorder
            self.recorder = AudioRecorder()
            self.recorder.set_level_callback(self._on_level)
            self.recorder.set_chunk_callback(self._on_speech_chunk)

        self._model_dir = model_dir

        # Level forwarding timer (started/stopped with recording)
        self._level_timer = QTimer()
        self._level_timer.timeout.connect(self._forward_level)

        # System tray
        with PerfTimer("app.finish_init.tray"):
            self._setup_tray()

        # Hotkey
        with PerfTimer("app.finish_init.hotkey"):
            self._hotkey = HotkeyListener(
                self._signals.hotkey_pressed.emit,
                self._config.hotkey_modifiers,
                self._config.hotkey_vk,
            )
            self._hotkey.start()

        hk_name = self._config.hotkey_display_name()
        log.info(f"Whisper Widget starting. Press {hk_name} to record.")
        log.info("PERF app.finish_init: %.0fms", (time.perf_counter() - _t0) * 1000)

        # Preload model (heavy imports + ONNX sessions in background)
        self._preload_model()

    def _setup_tray(self):
        self.tray = QSystemTrayIcon(self.app)
        self.tray.setIcon(self._app_icon)
        self.tray.setToolTip(f"Whisper Widget v{__version__} — {self._config.hotkey_display_name()} to record")

        menu = QMenu()

        self._status_action = QAction("Ready", self.app)
        self._status_action.setEnabled(False)
        menu.addAction(self._status_action)

        menu.addSeparator()

        settings_action = QAction("Settings...", self.app)
        settings_action.triggered.connect(self._show_settings)
        menu.addAction(settings_action)

        menu.addSeparator()

        quit_action = QAction("Quit", self.app)
        quit_action.triggered.connect(self.app.quit)
        menu.addAction(quit_action)

        self.tray.setContextMenu(menu)
        self.tray.show()

    def _show_settings(self):
        old_mods = self._config.hotkey_modifiers
        old_vk = self._config.hotkey_vk
        old_quality = self._config.model_quality

        dlg = SettingsDialog(
            on_pause_hotkey=self._pause_hotkey,
            on_resume_hotkey=self._resume_hotkey,
        )

        if dlg.exec() == SettingsDialog.DialogCode.Accepted:
            if (self._config.hotkey_modifiers != old_mods
                    or self._config.hotkey_vk != old_vk):
                self._restart_hotkey()
            else:
                self._resume_hotkey()
            self.tray.setToolTip(
                f"Whisper Widget — {self._config.hotkey_display_name()} to record"
            )
            # Model quality changed — download if needed + reload
            if self._config.model_quality != old_quality:
                self._switch_model(self._config.model_quality)

    def _pause_hotkey(self):
        """Temporarily unregister hotkey (e.g. during shortcut capture)."""
        if self._hotkey:
            self._hotkey.stop()
            self._hotkey.join(timeout=1)
            self._hotkey = None

    def _resume_hotkey(self):
        """Re-register hotkey if not already running."""
        if self._hotkey is None:
            self._hotkey = HotkeyListener(
                self._signals.hotkey_pressed.emit,
                self._config.hotkey_modifiers,
                self._config.hotkey_vk,
            )
            self._hotkey.start()

    def _restart_hotkey(self):
        if self._hotkey:
            self._hotkey.stop()
            self._hotkey.join(timeout=1)
        self._hotkey = HotkeyListener(
            self._signals.hotkey_pressed.emit,
            self._config.hotkey_modifiers,
            self._config.hotkey_vk,
        )
        self._hotkey.start()

    def _switch_model(self, quality: str):
        """Download new model quality if needed and reload."""
        model_dir = self._config.get_model_dir()
        if not models_exist(model_dir, quality):
            from config import DEFAULT_MODEL_DIR
            model_dir = DEFAULT_MODEL_DIR
            log.info(f"Downloading {quality} model to: {model_dir}")
            dlg = DownloadDialog(model_dir, quality=quality)
            if dlg.exec() != dlg.DialogCode.Accepted:
                return

        # Reload model
        self._model_dir = model_dir
        self._status_action.setText("Reloading model...")
        self.overlay.show_loading()
        self._preload_model()

    def _preload_model(self):
        def load():
            self._transcriber_loading = True
            try:
                t0 = time.perf_counter()

                with PerfTimer("model.import"):
                    from transcriber import ParakeetTranscriber

                log.info(f"Loading model from {self._model_dir}")
                self._transcriber = ParakeetTranscriber(self._model_dir)

                with PerfTimer("model.warmup"):
                    self._transcriber.warmup()

                total_ms = (time.perf_counter() - t0) * 1000
                log.info("PERF model.total: %.0fms", total_ms)
                log_memory("after_model_load")

                self._signals.model_loaded.emit()
            except Exception as e:
                log.error(f"Failed to load model: {e}", exc_info=True)
                self._signals.model_failed.emit(str(e))
            finally:
                self._transcriber_loading = False

        self._status_action.setText("Loading model...")
        threading.Thread(target=load, daemon=True).start()

    def _on_model_loaded(self):
        if hasattr(self, '_status_action'):
            self._status_action.setText("Ready")
        # Signal presplash to show success + exit
        signal = Path(__file__).parent / "debug" / ".splash_ready"
        signal.parent.mkdir(exist_ok=True)
        signal.touch()
        startup_ms = (time.perf_counter() - _startup_t0) * 1000
        log.info("PERF app.startup_total: %.0fms", startup_ms)
        log.info("Whisper Widget ready.")

    def _on_model_failed(self, error: str):
        if hasattr(self, '_status_action'):
            self._status_action.setText("Model load failed")
        # Signal presplash to close
        signal = Path(__file__).parent / "debug" / ".splash_ready"
        signal.parent.mkdir(exist_ok=True)
        signal.touch()
        if hasattr(self, 'tray'):
            self.tray.showMessage(
                "Whisper Widget", f"Model failed to load: {error}",
                QSystemTrayIcon.MessageIcon.Critical, 5000,
            )

    def _on_level(self, level: float):
        self._pending_level = level

    def _forward_level(self):
        if self._recording:
            self.overlay.set_level(self._pending_level)

    def _play_sound(self, name: str):
        """Play a sound asynchronously."""
        try:
            path = get_sound_path(name)
            if os.path.exists(path):
                winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception:
            pass

    def toggle_recording(self):
        # If drop mode is active, hotkey closes it
        if self._drop_mode:
            self._exit_drop_mode()
            return

        if self._recording:
            # Double-tap detection: if recording < 400ms, enter file drop mode
            elapsed = time.time() - self._record_start_time
            if elapsed < 0.4:
                self._cancel_recording()
                self._enter_drop_mode()
                return
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        if self._transcriber_loading:
            self.tray.showMessage(
                "Whisper Widget", "Model still loading...",
                QSystemTrayIcon.MessageIcon.Warning, 1500,
            )
            return

        self._debug = self._config.debug_mode
        log.info(f"Recording started (debug={self._debug})")

        # Clear old debug files
        if self._debug:
            debug_dir = Path(__file__).parent / "debug"
            if debug_dir.exists():
                for f in list(debug_dir.glob("*.wav")) + list(debug_dir.glob("*.txt")):
                    try:
                        f.unlink()
                    except OSError:
                        pass

        self._recording = True
        self._partial_texts.clear()
        self._chunk_audios: list[np.ndarray] = []
        self._pending_chunks = 0
        self._level_timer.start(16)
        self._total_chunked_audio = 0.0
        self._record_start_time = time.time()
        self.recorder.start()
        self.overlay.show_recording()
        self.tray.setIcon(_generate_tray_icon("recording"))
        self._status_action.setText("Recording...")
        self._play_sound("start.wav")

    def _on_speech_chunk(self, audio: np.ndarray):
        if self._transcriber is None or not self._recording:
            return

        chunk_dur = len(audio) / 16000
        self._total_chunked_audio += chunk_dur

        with self._partial_lock:
            chunk_idx = len(self._partial_texts)
            self._partial_texts.append("")
            if self._debug:
                self._chunk_audios.append(audio.copy())
            self._pending_chunks += 1

        def do_chunk_transcribe():
            try:
                t0 = time.time()
                debug = f"chunk_{chunk_idx:02d}" if self._debug else ""
                text = self._transcriber.transcribe(audio, debug_prefix=debug)
                dur = time.time() - t0
                log.info(f"Chunk {chunk_idx+1}: {chunk_dur:.1f}s audio -> {dur:.2f}s inference: {text[:60]}")
                with self._partial_lock:
                    self._partial_texts[chunk_idx] = text
                    self._pending_chunks -= 1
            except Exception as e:
                log.error(f"Chunk {chunk_idx+1} error: {e}", exc_info=True)
                with self._partial_lock:
                    self._pending_chunks -= 1

        threading.Thread(target=do_chunk_transcribe, daemon=True).start()

    def _stop_recording(self):
        self._recording = False
        self._level_timer.stop()
        remainder = self.recorder.stop()
        total_recording = time.time() - self._record_start_time

        self.overlay.show_transcribing()
        self.tray.setIcon(_generate_tray_icon("processing"))
        self._status_action.setText("Transcribing...")
        self._play_sound("stop.wav")

        remainder_dur = len(remainder) / 16000
        log.info(f"Recorded {total_recording:.1f}s ({self._total_chunked_audio:.1f}s chunked, {remainder_dur:.1f}s remainder)")

        def do_final():
            try:
                if self._transcriber is None:
                    self._signals.error.emit("Model not loaded")
                    return

                debug_pfx = "final" if self._debug else ""

                final_text = ""
                if len(remainder) >= 1600:
                    try:
                        t0 = time.time()
                        final_text = self._transcriber.transcribe(remainder, debug_prefix=debug_pfx)
                        dur = time.time() - t0
                        log.info(f"Final: {remainder_dur:.1f}s audio -> {dur:.2f}s inference: {final_text[:60]}")
                    except Exception as e:
                        log.error(f"Final chunk failed (recovering partials): {e}", exc_info=True)

                # Wait for pending chunks
                while True:
                    with self._partial_lock:
                        if self._pending_chunks == 0:
                            break
                    time.sleep(0.01)

                # Combine all parts, filtering dot-only artifacts
                with self._partial_lock:
                    all_parts = list(self._partial_texts)
                    chunk_audios = list(self._chunk_audios)
                all_parts.append(final_text)
                cleaned = [p for p in all_parts
                           if p.strip() and re.sub(r'[.\s]+', '', p)]
                full_text = " ".join(cleaned)

                total_latency = time.time() - self._record_start_time - total_recording
                total_audio = self._total_chunked_audio + remainder_dur
                rtf = total_latency / total_audio if total_audio > 0 else 0
                log.info(f"Latency: {total_latency:.2f}s -> {full_text[:80]}")
                log.info("PERF session: audio=%.1fs recording=%.1fs latency=%.2fs RTF=%.3f",
                         total_audio, total_recording, total_latency, rtf)

                # Debug: save transcripts and re-transcribe full audio for comparison
                if self._debug:
                    self._save_debug_transcripts(
                        full_text, all_parts, chunk_audios, remainder,
                    )

                self._signals.finished.emit(full_text)
            except Exception as e:
                # Last resort: still try to emit any partial texts we have
                with self._partial_lock:
                    all_parts = list(self._partial_texts)
                rescued = " ".join(p for p in all_parts
                                   if p.strip() and re.sub(r'[.\s]+', '', p))
                if rescued:
                    log.error(f"Transcription error (rescued {len(all_parts)} chunks): {e}", exc_info=True)
                    self._signals.finished.emit(rescued)
                else:
                    self._signals.error.emit(str(e))

        threading.Thread(target=do_final, daemon=True).start()

    def _save_debug_transcripts(self, chunked_text: str, parts: list[str],
                                   chunk_audios: list[np.ndarray],
                                   remainder: np.ndarray):
        """Save chunked vs full transcription comparison to debug/."""
        debug_dir = Path(__file__).parent / "debug"
        debug_dir.mkdir(exist_ok=True)

        # 1. Save chunked transcript with per-part breakdown
        with open(debug_dir / "transcript_chunked.txt", "w", encoding="utf-8") as f:
            for i, part in enumerate(parts):
                label = f"chunk_{i:02d}" if i < len(parts) - 1 else "final"
                f.write(f"[{label}] {part}\n")
            f.write(f"\n--- Combined ---\n{chunked_text}\n")

        # 2. Re-transcribe the full audio (no chunking) for comparison
        try:
            all_audio = list(chunk_audios)
            if len(remainder) >= 1600:
                all_audio.append(remainder)
            if all_audio:
                full_audio = np.concatenate(all_audio)
                log.info(f"Debug: re-transcribing full audio ({len(full_audio)/16000:.1f}s)...")
                t0 = time.time()
                full_text = self._transcriber.transcribe(full_audio, debug_prefix="full")
                dur = time.time() - t0
                log.info(f"Debug: full transcript in {dur:.2f}s: {full_text[:80]}")

                with open(debug_dir / "transcript_full.txt", "w", encoding="utf-8") as f:
                    f.write(f"{full_text}\n")
        except Exception as e:
            log.error(f"Debug: full re-transcription failed: {e}", exc_info=True)

    # --- File drop transcription ---

    def _cancel_recording(self):
        """Cancel recording without transcribing (for double-tap to drop mode)."""
        self._recording = False
        self._level_timer.stop()
        self.recorder.stop()  # discard audio

    def _enter_drop_mode(self):
        """Show file drop target overlay."""
        self._drop_mode = True
        self.overlay.show_drop_target()
        self.tray.setIcon(_generate_tray_icon("processing"))
        self._status_action.setText("Drop audio file...")
        self._play_sound("stop.wav")

        # Auto-close after 15 seconds
        self._drop_timeout = QTimer()
        self._drop_timeout.setSingleShot(True)
        self._drop_timeout.timeout.connect(self._exit_drop_mode)
        self._drop_timeout.start(15000)

    def _exit_drop_mode(self):
        """Close drop target overlay."""
        if not self._drop_mode:
            return
        self._drop_mode = False
        if self._drop_timeout:
            self._drop_timeout.stop()
            self._drop_timeout = None
        self.overlay.hide_overlay()
        self.tray.setIcon(self._app_icon)
        self._status_action.setText("Ready")

    def _on_file_dropped(self, path: str):
        """Handle file dropped on overlay."""
        if self._drop_timeout:
            self._drop_timeout.stop()
            self._drop_timeout = None

        if self._transcriber is None:
            self.tray.showMessage(
                "Whisper Widget", "Model still loading...",
                QSystemTrayIcon.MessageIcon.Warning, 1500,
            )
            return

        filename = Path(path).name
        log.info(f"File dropped for transcription: {path}")
        self.overlay.show_file_transcribing()
        self._status_action.setText(f"Transcribing {filename}...")

        def do_transcribe():
            try:
                audio = self._load_audio_file(path)
                if len(audio) < 1600:
                    log.warning("File too short to transcribe: %s", path)
                    self._signals.file_transcribed.emit("")
                    return
                audio_dur = len(audio) / 16000
                log.info(f"File loaded: {audio_dur:.1f}s audio from {filename}")
                text = self._transcriber.transcribe_chunked(audio)
                log.info(f"File transcription done ({filename}): {text[:80]}")
                self._signals.file_transcribed.emit(text)
            except Exception as e:
                log.error(f"File transcription error: {e}", exc_info=True)
                self._signals.error.emit(f"File transcription failed: {e}")

        threading.Thread(target=do_transcribe, daemon=True).start()

    @staticmethod
    def _load_audio_file(path: str) -> np.ndarray:
        """Load any audio file as 16kHz mono float32 using ffmpeg."""
        result = subprocess.run(
            [
                'ffmpeg', '-i', path,
                '-vn',              # no video
                '-ar', '16000',     # 16kHz
                '-ac', '1',         # mono
                '-f', 'f32le',      # raw float32 little-endian
                '-loglevel', 'error',
                '-',                # stdout
            ],
            capture_output=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg error: {result.stderr.decode().strip()}")
        return np.frombuffer(result.stdout, dtype=np.float32)

    def _on_file_transcription_done(self, text: str):
        """Handle completed file transcription."""
        if text.strip():
            self.overlay.show_file_done(text)
            self._status_action.setText("Transcription ready — click Copy")
        else:
            self._drop_mode = False
            self.overlay.show_success()
            self.tray.setIcon(self._app_icon)
            self._status_action.setText("No speech detected")
            QTimer.singleShot(2000, lambda: self._status_action.setText("Ready"))

    def _on_copy_clicked(self):
        """Copy file transcription to clipboard."""
        text = self.overlay._result_text
        if text:
            pyperclip.copy(text)
            log.info("File transcription copied to clipboard (%d chars)", len(text))
            self.tray.showMessage(
                "Whisper Widget", "Transcription copied to clipboard",
                QSystemTrayIcon.MessageIcon.Information, 1500,
            )
        self._drop_mode = False
        self.overlay.hide_overlay()
        self.tray.setIcon(self._app_icon)
        self._status_action.setText("Ready")

    def _on_close_drop(self):
        """Handle close click on file done overlay."""
        self._drop_mode = False
        self.overlay.hide_overlay()
        self.tray.setIcon(self._app_icon)
        self._status_action.setText("Ready")

    def _on_transcription_done(self, text: str):
        if text.strip():
            self.overlay.show_success()
            self._paste_text(text)
        else:
            self.overlay.hide_overlay()
        self.tray.setIcon(self._app_icon)
        self._status_action.setText("Ready")

    def _on_transcription_error(self, error: str):
        self._drop_mode = False
        self.overlay.hide_overlay()
        self.tray.setIcon(self._app_icon)
        self._status_action.setText("Ready")
        self.tray.showMessage(
            "Transcription Error", error,
            QSystemTrayIcon.MessageIcon.Critical, 3000,
        )
        log.error(f"Transcription error: {error}")

    def _paste_text(self, text: str):
        pyperclip.copy(text)
        QTimer.singleShot(50, lambda: pyautogui.hotkey('ctrl', 'v'))

    def run(self):
        # Defer all heavy init to after event loop starts —
        # splash is already visible and animating at this point
        QTimer.singleShot(0, self._finish_init)
        sys.exit(self.app.exec())


def _exception_hook(exc_type, exc_value, exc_tb):
    """Log uncaught exceptions to file (pythonw.exe swallows them otherwise)."""
    log.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))
    sys.__excepthook__(exc_type, exc_value, exc_tb)


if __name__ == "__main__":
    sys.excepthook = _exception_hook
    widget = VoiceWidget()
    widget.run()
