"""Single-exe launcher for Whisper Widget.

Compiled with PyInstaller into a small .exe that:
1. Extracts app source files to %LOCALAPPDATA%/WhisperWidget
2. Sets up embedded Python + installs deps (no system Python needed)
3. Launches presplash + main app (no console window)

The exe can live anywhere (Desktop, Downloads, Start Menu) — the app
always runs from a fixed location in LocalAppData.

Build:  see build.bat (downloads embedded Python, bundles everything)
"""

import ctypes
import hashlib
import logging
import os
import re
import sys
import shutil
import subprocess
import threading
import time
import zipfile
from pathlib import Path

APP_NAME = "WhisperWidget"
APP_DIR = Path(os.environ.get("LOCALAPPDATA", "")) / APP_NAME
PYTHON_DIR = APP_DIR / "python"

# Files to bundle / extract (app source)
APP_FILES = [
    "main.py", "presplash.pyw", "overlay.py", "config.py", "recorder.py",
    "transcriber.py", "vad.py", "settings_dialog.py", "model_downloader.py",
    "assets.py", "perf_logger.py", "requirements.txt", "favicon.ico",
]

# Bundled Python setup files
PYTHON_ZIP = "python-embed.zip"
GET_PIP = "get-pip.py"

# --- Logging ---
APP_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = APP_DIR / "launcher.log"
_log_handler = logging.FileHandler(str(LOG_FILE), encoding="utf-8")
_log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
logging.basicConfig(
    handlers=[_log_handler],
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
log = logging.getLogger("launcher")


def _get_bundle_dir() -> Path:
    """Where bundled data files live (PyInstaller _MEIPASS or script dir)."""
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    return Path(__file__).parent


def _files_changed() -> bool:
    """Check if bundled app files differ from installed ones."""
    bundle = _get_bundle_dir()
    for name in APP_FILES:
        src = bundle / name
        dst = APP_DIR / name
        if not src.exists():
            continue
        if not dst.exists():
            return True
        if src.stat().st_size != dst.stat().st_size:
            return True
        if src.read_bytes() != dst.read_bytes():
            return True
    return False


def _is_app_running() -> bool:
    """Check if the app is currently running via PID file."""
    pid_file = APP_DIR / ".pid"
    if not pid_file.exists():
        return False
    try:
        pid = int(pid_file.read_text().strip())
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid
        )
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    except (ValueError, OSError):
        return False


def _kill_existing():
    """Terminate any running WhisperWidget app process using PID file."""
    pid_file = APP_DIR / ".pid"
    if not pid_file.exists():
        return
    try:
        pid = int(pid_file.read_text().strip())
        PROCESS_TERMINATE = 0x0001
        SYNCHRONIZE = 0x00100000
        handle = ctypes.windll.kernel32.OpenProcess(
            PROCESS_TERMINATE | SYNCHRONIZE, False, pid
        )
        if handle:
            ctypes.windll.kernel32.TerminateProcess(handle, 0)
            ctypes.windll.kernel32.WaitForSingleObject(handle, 3000)
            ctypes.windll.kernel32.CloseHandle(handle)
            log.info("Terminated existing app (PID %d)", pid)
        pid_file.unlink(missing_ok=True)
    except (ValueError, OSError) as e:
        log.warning("Could not kill existing process: %s", e)


def _deps_changed() -> bool:
    """Check if requirements.txt changed since last successful install."""
    req_file = APP_DIR / "requirements.txt"
    hash_file = APP_DIR / ".requirements_hash"
    if not req_file.exists():
        return False
    current_hash = hashlib.sha256(req_file.read_bytes()).hexdigest()[:16]
    if hash_file.exists():
        if hash_file.read_text().strip() == current_hash:
            return False
    return True


def _mark_deps_installed():
    """Record requirements.txt hash after successful install."""
    req_file = APP_DIR / "requirements.txt"
    hash_file = APP_DIR / ".requirements_hash"
    if req_file.exists():
        current_hash = hashlib.sha256(req_file.read_bytes()).hexdigest()[:16]
        hash_file.write_text(current_hash)


def _extract_app_files():
    """Copy app source files from bundle to app directory."""
    APP_DIR.mkdir(parents=True, exist_ok=True)
    bundle = _get_bundle_dir()
    log.info("Extracting app files from %s to %s", bundle, APP_DIR)
    for name in APP_FILES:
        src = bundle / name
        dst = APP_DIR / name
        if src.exists():
            shutil.copy2(src, dst)
            log.debug("  copied %s", name)
        else:
            log.warning("  MISSING in bundle: %s", name)


def _needs_setup() -> bool:
    """Check if embedded Python needs setup."""
    pythonw = PYTHON_DIR / "pythonw.exe"
    pip = PYTHON_DIR / "Scripts" / "pip.exe"
    needs = not pythonw.exists() or not pip.exists()
    log.info("Needs setup? %s (pythonw=%s, pip=%s)", needs, pythonw.exists(), pip.exists())
    return needs


def _needs_deps() -> bool:
    """Check if dependencies are installed."""
    site_pkgs = PYTHON_DIR / "Lib" / "site-packages"
    # Quick check: does PyQt6 exist?
    has_pyqt = (site_pkgs / "PyQt6").exists() if site_pkgs.exists() else False
    log.info("Needs deps? %s (PyQt6 installed: %s)", not has_pyqt, has_pyqt)
    return not has_pyqt


def _setup_python(on_status=None):
    """Extract embedded Python and bootstrap pip."""
    bundle = _get_bundle_dir()

    # Step 1: Extract embedded Python
    if not (PYTHON_DIR / "python.exe").exists():
        if on_status:
            on_status("Extracting Python runtime...")
        log.info("Extracting embedded Python to %s", PYTHON_DIR)

        zip_path = bundle / PYTHON_ZIP
        if not zip_path.exists():
            msg = f"Embedded Python not found: {zip_path}"
            log.error(msg)
            if on_status:
                on_status(f"Error: {msg}")
            return False

        PYTHON_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(PYTHON_DIR)
        log.info("Extracted %d files", len(os.listdir(PYTHON_DIR)))

        # Configure ._pth: enable site-packages + add app dir to path
        for pth in PYTHON_DIR.glob("python*._pth"):
            text = pth.read_text(encoding="utf-8")
            modified = False
            if "#import site" in text:
                text = text.replace("#import site", "import site")
                modified = True
            # Add parent dir (..) so Python finds app modules (config.py, etc.)
            if ".." not in text:
                text = text.replace("\n.", "\n.\n..", 1)
                modified = True
            if modified:
                pth.write_text(text, encoding="utf-8")
                log.info("Configured %s", pth.name)
            break

    # Step 2: Install pip
    pip_exe = PYTHON_DIR / "Scripts" / "pip.exe"
    if not pip_exe.exists():
        if on_status:
            on_status("Installing pip...")
        log.info("Bootstrapping pip with get-pip.py")

        get_pip_path = bundle / GET_PIP
        if not get_pip_path.exists():
            msg = f"get-pip.py not found: {get_pip_path}"
            log.error(msg)
            if on_status:
                on_status(f"Error: {msg}")
            return False

        python = str(PYTHON_DIR / "python.exe")
        result = subprocess.run(
            [python, str(get_pip_path), "--no-warn-script-location"],
            creationflags=subprocess.CREATE_NO_WINDOW,
            capture_output=True, text=True,
        )
        log.info("get-pip exit code: %d", result.returncode)
        if result.stdout.strip():
            for line in result.stdout.strip().splitlines()[-10:]:
                log.info("get-pip: %s", line)
        if result.stderr.strip():
            for line in result.stderr.strip().splitlines()[-10:]:
                log.warning("get-pip err: %s", line)
        if result.returncode != 0:
            if on_status:
                on_status(f"Error installing pip (code {result.returncode})")
            return False

    # Step 3: Install setuptools + wheel (needed for sdist packages like pyautogui)
    setuptools_marker = PYTHON_DIR / "Lib" / "site-packages" / "setuptools"
    if not setuptools_marker.exists():
        if on_status:
            on_status("Installing build tools...")
        log.info("Installing setuptools + wheel")
        python = str(PYTHON_DIR / "python.exe")
        result = subprocess.run(
            [str(pip_exe), "install", "setuptools", "wheel", "--no-warn-script-location"],
            creationflags=subprocess.CREATE_NO_WINDOW,
            capture_output=True, text=True,
        )
        log.info("setuptools install exit code: %d", result.returncode)
        if result.stderr.strip():
            for line in result.stderr.strip().splitlines()[-5:]:
                log.warning("setuptools err: %s", line)

    log.info("Python setup complete")
    return True


def _install_deps(on_status=None):
    """Install Python dependencies via pip."""
    pip = str(PYTHON_DIR / "Scripts" / "pip.exe")
    req = str(APP_DIR / "requirements.txt")

    if not Path(pip).exists():
        log.error("pip.exe not found at %s", pip)
        if on_status:
            on_status("Error: pip not found")
        return False

    if on_status:
        on_status("Resolving dependencies...")
    log.info("Running: %s install -r %s", pip, req)

    proc = subprocess.Popen(
        [pip, "install", "-r", req, "--no-warn-script-location"],
        creationflags=subprocess.CREATE_NO_WINDOW,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )

    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        log.info("pip: %s", line)
        if on_status:
            low = line.lower()
            if low.startswith("collecting "):
                match = re.match(r"Collecting\s+(\S+)", line, re.IGNORECASE)
                pkg = match.group(1).split(">=")[0].split("<=")[0].split("==")[0].split("<")[0].split(">")[0] if match else "..."
                on_status(f"Resolving {pkg}...")
            elif "downloading " in low:
                match = re.search(r"Downloading\s+\S+/([^\s/]+)", line)
                fname = match.group(1)[:40] if match else "..."
                on_status(f"Downloading {fname}...")
            elif low.startswith("installing collected"):
                pkgs = line.split(":", 1)[-1].strip() if ":" in line else ""
                pkg_count = len(pkgs.split(",")) if pkgs else 0
                on_status(f"Installing {pkg_count} packages (this takes a while)...")
            elif "successfully installed" in low:
                on_status("Done!")

    rc = proc.wait()
    log.info("pip install exit code: %d", rc)

    if rc != 0:
        if on_status:
            on_status(f"Error installing dependencies (code {rc})")
        return False

    if on_status:
        on_status("Done!")
    log.info("Dependencies installed")
    _mark_deps_installed()
    return True


def _verify_imports():
    """Quick smoke test: try importing critical modules with python.exe to catch errors."""
    python = str(PYTHON_DIR / "python.exe")
    test_code = "import onnxruntime; import PyQt6.QtWidgets; print('OK')"
    log.info("Verifying imports...")
    result = subprocess.run(
        [python, "-c", test_code],
        cwd=str(APP_DIR),
        creationflags=subprocess.CREATE_NO_WINDOW,
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        log.error("Import verification FAILED (code %d)", result.returncode)
        if result.stderr.strip():
            for line in result.stderr.strip().splitlines():
                log.error("  %s", line)
        if result.stdout.strip():
            log.info("  stdout: %s", result.stdout.strip())
        return False
    log.info("Import verification OK")
    return True


def _launch_app():
    """Start presplash + main app."""
    pythonw = str(PYTHON_DIR / "pythonw.exe")
    python = str(PYTHON_DIR / "python.exe")
    log.info("Launching app with %s", pythonw)

    if not Path(pythonw).exists():
        log.error("pythonw.exe not found — cannot launch")
        return

    presplash = APP_DIR / "presplash.pyw"
    if presplash.exists():
        log.info("Starting presplash")
        subprocess.Popen(
            [pythonw, str(presplash)],
            cwd=str(APP_DIR),
            creationflags=subprocess.CREATE_NO_WINDOW,
        )

    log.info("Starting main.py")
    subprocess.Popen(
        [pythonw, str(APP_DIR / "main.py")],
        cwd=str(APP_DIR),
        creationflags=subprocess.CREATE_NO_WINDOW,
    )


def _show_error(message: str):
    """Show error dialog using tkinter."""
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Whisper Widget", message)
        root.destroy()
    except Exception:
        log.error("Could not show error dialog: %s", message)


def _create_desktop_shortcut():
    """Create a desktop shortcut to the launcher exe."""
    if not getattr(sys, "frozen", False):
        return
    try:
        import ctypes
        from ctypes import wintypes
        # Get Desktop path via SHGetFolderPath
        CSIDL_DESKTOP = 0x0000
        buf = ctypes.create_unicode_buffer(wintypes.MAX_PATH)
        ctypes.windll.shell32.SHGetFolderPathW(None, CSIDL_DESKTOP, None, 0, buf)
        desktop = Path(buf.value)

        shortcut_path = desktop / f"{APP_NAME}.lnk"
        if shortcut_path.exists():
            log.info("Desktop shortcut already exists")
            return

        # Use PowerShell to create .lnk shortcut
        exe_path = sys.executable
        ps_cmd = (
            f'$ws = New-Object -ComObject WScript.Shell; '
            f'$s = $ws.CreateShortcut("{shortcut_path}"); '
            f'$s.TargetPath = "{exe_path}"; '
            f'$s.WorkingDirectory = "{exe_path.rsplit(chr(92), 1)[0] if chr(92) in exe_path else ""}"; '
            f'$s.Description = "Whisper Widget - Voice to Text"; '
            f'$s.IconLocation = "{exe_path},0"; '
            f'$s.Save()'
        )
        subprocess.run(
            ["powershell", "-Command", ps_cmd],
            creationflags=subprocess.CREATE_NO_WINDOW,
            capture_output=True,
        )
        log.info("Created desktop shortcut: %s", shortcut_path)
    except Exception as e:
        log.warning("Could not create desktop shortcut: %s", e)


def _setup_with_ui():
    """Show tkinter progress window during setup."""
    import tkinter as tk

    root = tk.Tk()
    root.title("Whisper Widget")
    root.configure(bg="#1c1c1e")
    root.resizable(False, False)
    root.attributes("-topmost", True)

    w, h = 420, 140
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    root.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")

    title = tk.Label(
        root, text="Setting up Whisper Widget",
        font=("Segoe UI", 13, "bold"), fg="#ffffff", bg="#1c1c1e",
    )
    title.pack(pady=(20, 4))

    status = tk.Label(
        root, text="Preparing...",
        font=("Segoe UI", 10), fg="#8e8e93", bg="#1c1c1e",
    )
    status.pack(pady=(0, 8))

    error_label = tk.Label(
        root, text="",
        font=("Segoe UI", 8), fg="#ff453a", bg="#1c1c1e",
        wraplength=380,
    )
    error_label.pack(pady=(0, 8))

    def update_status(text):
        root.after(0, lambda: status.configure(text=text))
        log.info("UI status: %s", text)

    setup_ok = [False]

    def do_setup():
        try:
            if not _setup_python(on_status=update_status):
                root.after(0, lambda: error_label.configure(
                    text=f"Check log: {LOG_FILE}"
                ))
                return
            if not _install_deps(on_status=update_status):
                root.after(0, lambda: error_label.configure(
                    text=f"Check log: {LOG_FILE}"
                ))
                return
            setup_ok[0] = True
            root.after(100, root.destroy)
        except Exception as e:
            log.exception("Setup crashed")
            root.after(0, lambda: status.configure(text=f"Error: {e}"))
            root.after(0, lambda: error_label.configure(
                text=f"Check log: {LOG_FILE}"
            ))

    threading.Thread(target=do_setup, daemon=True).start()
    root.mainloop()
    return setup_ok[0]


def _write_exe_path():
    """Save the launcher exe path so the app can use it for autostart."""
    if getattr(sys, "frozen", False):
        marker = APP_DIR / ".launcher_path"
        marker.write_text(sys.executable, encoding="utf-8")
        log.info("Wrote launcher path: %s", sys.executable)


def _read_bundled_version() -> str:
    """Read __version__ from bundled config.py."""
    config_py = _get_bundle_dir() / "config.py"
    if config_py.exists():
        for line in config_py.read_text(encoding="utf-8").splitlines()[:10]:
            if line.startswith("__version__"):
                try:
                    return line.split('"')[1]
                except IndexError:
                    pass
    return "unknown"


def main():
    t_start = time.time()
    version = _read_bundled_version()
    log.info("=" * 50)
    log.info("Launcher v%s starting (frozen=%s)", version, getattr(sys, "frozen", False))
    log.info("sys.executable = %s", sys.executable)
    log.info("APP_DIR = %s", APP_DIR)

    is_frozen = getattr(sys, "frozen", False)

    # Detect if this is an update (bundled files differ from installed)
    if is_frozen:
        is_update = _files_changed()
        if is_update:
            log.info("Update detected — installing v%s", version)
            _kill_existing()
        elif _is_app_running():
            log.info("App already running (v%s), exiting", version)
            return

    # Extract latest source files (updates app code)
    t0 = time.time()
    _extract_app_files()
    log.info("PERF launcher.extract: %.0fms", (time.time() - t0) * 1000)

    # Save exe path for autostart registry
    _write_exe_path()

    # Create desktop shortcut on first run
    _create_desktop_shortcut()

    # Setup: extract Python + install deps (with UI if needed)
    need_setup = _needs_setup()
    need_deps = _needs_deps()
    deps_changed = _deps_changed()
    ran_setup = False
    if need_setup or need_deps or deps_changed:
        log.info("Setup needed (python=%s, deps=%s, deps_changed=%s)",
                 need_setup, need_deps, deps_changed)
        if not _setup_with_ui():
            log.error("Setup failed — not launching app")
            return
        ran_setup = True

    # Verify imports work — only when setup ran or files were updated
    t0 = time.time()
    if ran_setup and not _verify_imports():
        log.warning("Import verification failed — re-running setup to repair")
        if not _setup_with_ui():
            log.error("Repair failed — not launching app")
            return
        if not _verify_imports():
            log.error("Still broken after repair")
            _show_error(
                "Whisper Widget failed to start.\n\n"
                f"Check log: {LOG_FILE}"
            )
            return
    log.info("PERF launcher.verify: %.0fms", (time.time() - t0) * 1000)

    _launch_app()
    log.info("PERF launcher.total: %.0fms", (time.time() - t_start) * 1000)
    log.info("Launcher done")


if __name__ == "__main__":
    main()
