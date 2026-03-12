"""Settings dialog with Apple-inspired dark theme."""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QKeyEvent
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox,
    QPushButton, QLineEdit, QFrame, QComboBox,
)

from config import Config, VK_NAMES

# Windows VK code mappings
QT_KEY_TO_VK = {
    Qt.Key.Key_Space: 0x20, Qt.Key.Key_Return: 0x0D, Qt.Key.Key_Escape: 0x1B,
    Qt.Key.Key_Tab: 0x09, Qt.Key.Key_Backspace: 0x08, Qt.Key.Key_Delete: 0x2E,
    Qt.Key.Key_Insert: 0x2D,
    **{getattr(Qt.Key, f"Key_F{i}"): 0x70 + i - 1 for i in range(1, 13)},
    **{getattr(Qt.Key, f"Key_{chr(c)}"): c for c in range(0x41, 0x5B)},
    **{getattr(Qt.Key, f"Key_{i}"): 0x30 + i for i in range(10)},
}

DARK_STYLE = """
QDialog {
    background: #1c1c1e;
}
QLabel {
    color: #e5e5e7;
    font-family: "Segoe UI";
}
QLabel#section {
    color: #8e8e93;
    font-family: "Segoe UI";
    font-size: 8pt;
    font-weight: bold;
    padding-left: 2px;
}
QLabel#hint {
    color: #636366;
    font-family: "Segoe UI";
    font-size: 8pt;
    padding-left: 2px;
}
QCheckBox {
    color: #e5e5e7;
    font-family: "Segoe UI";
    font-size: 10pt;
    spacing: 8px;
}
QCheckBox::indicator {
    width: 18px; height: 18px;
    border-radius: 4px;
    border: 2px solid #48484a;
    background: #2c2c2e;
}
QCheckBox::indicator:checked {
    background: #0a84ff;
    border-color: #0a84ff;
}
QLineEdit {
    background: #2c2c2e;
    color: #ffffff;
    border: 1px solid #3a3a3c;
    border-radius: 8px;
    padding: 10px 14px;
    font-family: "Segoe UI";
    font-size: 10pt;
    selection-background-color: #0a84ff;
}
QLineEdit:focus {
    border-color: #0a84ff;
}
QLineEdit[capturing="true"] {
    border-color: #ff9f0a;
    background: #2a2520;
}
QFrame#separator {
    background: #2c2c2e;
    max-height: 1px;
}
QPushButton {
    background: #2c2c2e;
    color: #ffffff;
    border: 1px solid #3a3a3c;
    border-radius: 8px;
    padding: 8px 20px;
    font-family: "Segoe UI";
    font-size: 10pt;
}
QPushButton:hover {
    background: #3a3a3c;
}
QPushButton#primary {
    background: #0a84ff;
    border-color: #0a84ff;
    font-weight: bold;
}
QPushButton#primary:hover {
    background: #409cff;
}
QPushButton#change {
    padding: 8px 14px;
    font-size: 9pt;
}
QComboBox {
    background: #2c2c2e;
    color: #ffffff;
    border: 1px solid #3a3a3c;
    border-radius: 8px;
    padding: 8px 14px;
    font-family: "Segoe UI";
    font-size: 10pt;
}
QComboBox:hover {
    border-color: #48484a;
}
QComboBox::drop-down {
    border: none;
    width: 24px;
}
QComboBox::down-arrow {
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid #8e8e93;
    margin-right: 8px;
}
QComboBox QAbstractItemView {
    background: #2c2c2e;
    color: #ffffff;
    border: 1px solid #3a3a3c;
    selection-background-color: #0a84ff;
    outline: none;
}
"""


def _hotkey_name(modifiers: int, vk: int) -> str:
    """Format hotkey as display string without touching Config."""
    parts = []
    if modifiers & 0x0002:
        parts.append("Ctrl")
    if modifiers & 0x0001:
        parts.append("Alt")
    if modifiers & 0x0004:
        parts.append("Shift")
    if modifiers & 0x0008:
        parts.append("Win")
    parts.append(VK_NAMES.get(vk, f"0x{vk:02X}"))
    return " + ".join(parts)


class HotkeyEdit(QLineEdit):
    """Line edit that captures key combinations."""

    capture_started = pyqtSignal()
    capture_ended = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._capturing = False
        self._modifiers = 0
        self._vk = 0

    def set_hotkey(self, modifiers: int, vk: int):
        self._modifiers = modifiers
        self._vk = vk
        self.setText(_hotkey_name(modifiers, vk))

    def get_hotkey(self) -> tuple[int, int]:
        return self._modifiers, self._vk

    def start_capture(self):
        self._capturing = True
        self.setProperty("capturing", True)
        self.style().unpolish(self)
        self.style().polish(self)
        self.setText("Press new shortcut...")
        self.setFocus()
        self.capture_started.emit()

    def _stop_capture(self):
        self._capturing = False
        self.setProperty("capturing", False)
        self.style().unpolish(self)
        self.style().polish(self)
        self.setText(_hotkey_name(self._modifiers, self._vk))
        self.capture_ended.emit()

    def mousePressEvent(self, event):
        if not self._capturing:
            self.start_capture()
        else:
            super().mousePressEvent(event)

    def keyPressEvent(self, event: QKeyEvent):
        if not self._capturing:
            return super().keyPressEvent(event)

        key = event.key()

        # Escape cancels capture
        if key == Qt.Key.Key_Escape:
            self._stop_capture()
            return

        # Ignore modifier-only presses, show live preview
        if key in (Qt.Key.Key_Control, Qt.Key.Key_Shift, Qt.Key.Key_Alt, Qt.Key.Key_Meta):
            parts = []
            qt_mods = event.modifiers()
            if qt_mods & Qt.KeyboardModifier.ControlModifier:
                parts.append("Ctrl")
            if qt_mods & Qt.KeyboardModifier.AltModifier:
                parts.append("Alt")
            if qt_mods & Qt.KeyboardModifier.ShiftModifier:
                parts.append("Shift")
            if qt_mods & Qt.KeyboardModifier.MetaModifier:
                parts.append("Win")
            self.setText(" + ".join(parts) + " + ..." if parts else "Press new shortcut...")
            return

        # Map Qt modifiers to Windows modifier bitmask
        qt_mods = event.modifiers()
        win_mods = 0
        if qt_mods & Qt.KeyboardModifier.ControlModifier:
            win_mods |= 0x0002
        if qt_mods & Qt.KeyboardModifier.AltModifier:
            win_mods |= 0x0001
        if qt_mods & Qt.KeyboardModifier.ShiftModifier:
            win_mods |= 0x0004
        if qt_mods & Qt.KeyboardModifier.MetaModifier:
            win_mods |= 0x0008

        # Need at least one modifier
        if win_mods == 0:
            self.setText("Need modifier + key")
            return

        vk = QT_KEY_TO_VK.get(key, 0)
        if vk == 0:
            self.setText("Unsupported key")
            return

        self._modifiers = win_mods
        self._vk = vk
        self._stop_capture()

    def focusOutEvent(self, event):
        if self._capturing:
            self._stop_capture()
        super().focusOutEvent(event)


class SettingsDialog(QDialog):
    """Settings dialog with Apple-inspired dark theme."""

    def __init__(self, on_pause_hotkey=None, on_resume_hotkey=None, parent=None):
        super().__init__(parent)
        self._config = Config()
        self._on_pause_hotkey = on_pause_hotkey
        self._on_resume_hotkey = on_resume_hotkey
        self.setWindowTitle("Settings")
        self.setFixedSize(380, 480)
        self.setStyleSheet(DARK_STYLE)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 20)
        layout.setSpacing(0)

        # Title
        title = QLabel("Whisper Widget")
        title.setFont(QFont("Segoe UI", 15, QFont.Weight.DemiBold))
        title.setStyleSheet("color: #ffffff;")
        layout.addWidget(title)

        subtitle = QLabel("Voice-to-text with Parakeet TDT v3")
        subtitle.setFont(QFont("Segoe UI", 9))
        subtitle.setStyleSheet("color: #636366;")
        layout.addWidget(subtitle)

        layout.addSpacing(20)

        # --- Hotkey section ---
        section1 = QLabel("SHORTCUT")
        section1.setObjectName("section")
        layout.addWidget(section1)

        layout.addSpacing(8)

        hotkey_row = QHBoxLayout()
        hotkey_row.setSpacing(8)

        self._hotkey_edit = HotkeyEdit()
        self._hotkey_edit.set_hotkey(
            self._config.hotkey_modifiers,
            self._config.hotkey_vk,
        )
        self._hotkey_edit.capture_started.connect(self._on_capture_start)
        self._hotkey_edit.capture_ended.connect(self._on_capture_end)
        hotkey_row.addWidget(self._hotkey_edit)

        change_btn = QPushButton("Change")
        change_btn.setObjectName("change")
        change_btn.setFixedWidth(72)
        change_btn.clicked.connect(self._hotkey_edit.start_capture)
        hotkey_row.addWidget(change_btn)

        layout.addLayout(hotkey_row)

        layout.addSpacing(4)

        hint = QLabel("Click Change or the field, then press a new shortcut")
        hint.setObjectName("hint")
        layout.addWidget(hint)

        layout.addSpacing(20)

        # Separator
        sep = QFrame()
        sep.setObjectName("separator")
        sep.setFixedHeight(1)
        layout.addWidget(sep)

        layout.addSpacing(16)

        # --- Model section ---
        section2 = QLabel("MODEL")
        section2.setObjectName("section")
        layout.addWidget(section2)

        layout.addSpacing(8)

        model_row = QHBoxLayout()
        model_row.setSpacing(8)

        model_label = QLabel("Quality")
        model_label.setFixedWidth(50)
        model_row.addWidget(model_label)

        self._quality_combo = QComboBox()
        self._quality_combo.addItem("Fast (INT8) — 670 MB", "int8")
        self._quality_combo.addItem("Best (FP32) — 2.5 GB", "fp32")
        current = self._config.model_quality
        idx = self._quality_combo.findData(current)
        if idx >= 0:
            self._quality_combo.setCurrentIndex(idx)
        model_row.addWidget(self._quality_combo)

        layout.addLayout(model_row)

        self._model_hint = QLabel("Requires restart + download if changed")
        self._model_hint.setObjectName("hint")
        layout.addWidget(self._model_hint)

        layout.addSpacing(16)

        # Separator
        sep2 = QFrame()
        sep2.setObjectName("separator")
        sep2.setFixedHeight(1)
        layout.addWidget(sep2)

        layout.addSpacing(16)

        # --- General section ---
        section3 = QLabel("GENERAL")
        section3.setObjectName("section")
        layout.addWidget(section3)

        layout.addSpacing(10)

        self._autostart_cb = QCheckBox("Launch at login")
        self._autostart_cb.setChecked(self._config.autostart)
        layout.addWidget(self._autostart_cb)

        layout.addSpacing(8)

        self._debug_cb = QCheckBox("Debug mode")
        self._debug_cb.setChecked(self._config.debug_mode)
        layout.addWidget(self._debug_cb)

        debug_hint = QLabel("Saves audio + transcripts to debug/ folder")
        debug_hint.setObjectName("hint")
        layout.addWidget(debug_hint)

        layout.addStretch()

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        btn_row.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFixedWidth(90)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        save_btn = QPushButton("Save")
        save_btn.setObjectName("primary")
        save_btn.setFixedWidth(90)
        save_btn.clicked.connect(self._save)
        btn_row.addWidget(save_btn)

        layout.addLayout(btn_row)

    def _on_capture_start(self):
        """Pause the global hotkey so key combos don't trigger recording."""
        if self._on_pause_hotkey:
            self._on_pause_hotkey()

    def _on_capture_end(self):
        """Resume the global hotkey after capture is done."""
        if self._on_resume_hotkey:
            self._on_resume_hotkey()

    def _save(self):
        mods, vk = self._hotkey_edit.get_hotkey()
        self._config.set_hotkey(mods, vk)
        self._config.autostart = self._autostart_cb.isChecked()
        self._config.debug_mode = self._debug_cb.isChecked()
        self._config.model_quality = self._quality_combo.currentData()
        self._config.save()
        self.accept()

    def reject(self):
        # Make sure hotkey is resumed if dialog is cancelled mid-capture
        if self._on_resume_hotkey:
            self._on_resume_hotkey()
        super().reject()

    @property
    def hotkey_changed(self) -> bool:
        mods, vk = self._hotkey_edit.get_hotkey()
        return (mods != self._config.hotkey_modifiers
                or vk != self._config.hotkey_vk)
