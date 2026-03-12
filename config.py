"""Application settings management."""

import json
import os
import sys
import winreg
from pathlib import Path

APP_NAME = "WhisperWidget"
APP_DIR = Path(os.environ.get("APPDATA", "")) / APP_NAME
SETTINGS_PATH = APP_DIR / "settings.json"
DEFAULT_MODEL_DIR = APP_DIR / "models" / "parakeet-tdt-v3"

# Also check local models dir (for dev / portable mode)
LOCAL_MODEL_DIR = Path(__file__).parent / "models" / "parakeet-tdt-v3"

DEFAULTS = {
    "hotkey_modifiers": 0x0002,  # MOD_CONTROL
    "hotkey_vk": 0x20,           # VK_SPACE
    "autostart": False,
    "debug_mode": False,
    "model_quality": "int8",     # "int8" (fast, 670MB) or "fp32" (best, 2.5GB)
}


class Config:
    """Singleton settings manager backed by JSON file."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._data = dict(DEFAULTS)
            cls._instance._load()
        return cls._instance

    def _load(self):
        if SETTINGS_PATH.exists():
            try:
                with open(SETTINGS_PATH, encoding="utf-8") as f:
                    stored = json.load(f)
                self._data.update(stored)
            except (json.JSONDecodeError, OSError):
                pass

    def save(self):
        APP_DIR.mkdir(parents=True, exist_ok=True)
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)

    def get(self, key, default=None):
        return self._data.get(key, default)

    def set(self, key, value):
        self._data[key] = value

    @property
    def hotkey_modifiers(self) -> int:
        return self._data["hotkey_modifiers"]

    @property
    def hotkey_vk(self) -> int:
        return self._data["hotkey_vk"]

    @property
    def autostart(self) -> bool:
        return self._data.get("autostart", False)

    @autostart.setter
    def autostart(self, enabled: bool):
        self._data["autostart"] = enabled
        self._set_autostart_registry(enabled)
        self.save()

    @property
    def debug_mode(self) -> bool:
        return self._data.get("debug_mode", False)

    @debug_mode.setter
    def debug_mode(self, enabled: bool):
        self._data["debug_mode"] = enabled
        self.save()

    @property
    def model_quality(self) -> str:
        return self._data.get("model_quality", "int8")

    @model_quality.setter
    def model_quality(self, value: str):
        self._data["model_quality"] = value
        self.save()

    def get_model_dir(self) -> Path:
        """Return model directory, preferring local if it exists."""
        if LOCAL_MODEL_DIR.exists() and (LOCAL_MODEL_DIR / "vocab.txt").exists():
            return LOCAL_MODEL_DIR
        return DEFAULT_MODEL_DIR

    def _set_autostart_registry(self, enable: bool):
        try:
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Run",
                0, winreg.KEY_SET_VALUE,
            )
            if enable:
                # Packaged exe: launcher writes its path to .launcher_path
                # Dev mode: fall back to launch.bat
                app_dir = Path(__file__).parent
                launcher_marker = app_dir / ".launcher_path"
                if launcher_marker.exists():
                    cmd = launcher_marker.read_text(encoding="utf-8").strip()
                else:
                    cmd = str(app_dir / "launch.bat")
                winreg.SetValueEx(key, APP_NAME, 0, winreg.REG_SZ, cmd)
            else:
                try:
                    winreg.DeleteValue(key, APP_NAME)
                except FileNotFoundError:
                    pass
            winreg.CloseKey(key)
        except OSError:
            pass

    def set_hotkey(self, modifiers: int, vk: int):
        self._data["hotkey_modifiers"] = modifiers
        self._data["hotkey_vk"] = vk
        self.save()

    def hotkey_display_name(self) -> str:
        return format_hotkey_name(self._data["hotkey_modifiers"], self._data["hotkey_vk"])


VK_NAMES = {
    0x20: "Space", 0x0D: "Enter", 0x1B: "Escape", 0x09: "Tab",
    **{0x70 + i: f"F{i+1}" for i in range(12)},
    **{0x41 + i: chr(0x41 + i) for i in range(26)},
    **{0x30 + i: str(i) for i in range(10)},
}


def format_hotkey_name(modifiers: int, vk: int) -> str:
    """Format modifier bitmask + VK code as a display string like 'Ctrl + Space'."""
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
