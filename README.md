# Whisper Widget

Offline voice-to-text for Windows. Press a hotkey, speak, release — transcription auto-pastes at your cursor. No cloud, no API keys, everything runs locally.

Uses [Parakeet TDT v3](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx) (NVIDIA ONNX) for fast, accurate speech recognition on CPU.

## Features

- **Global hotkey** (default `Ctrl+Space`) — works from any app
- **Fully offline** — no internet needed after model download
- **Auto-paste** — transcription goes straight to your cursor
- **Dynamic Island UI** — macOS-inspired floating overlay with live waveform
- **Chunked transcription** — long recordings are split at silence gaps and transcribed incrementally
- **INT8 / FP32 models** — choose between speed (~670 MB) and quality (~2.5 GB)
- **Single-instance** — named mutex prevents duplicate processes
- **Performance tracking** — comprehensive PERF logging for every stage (startup, model load, inference breakdown)
- **System tray** — lives in the tray with settings, autostart, and model quality switching
- **Self-updating installer** — packaged exe detects changes and updates the installation

## Requirements

- Windows 10/11
- Python 3.11+ (for development mode)
- ~4 GB disk space (model + dependencies)
- Microphone

## Quick Start

### Development Mode

```
git clone https://github.com/jusedit/whisper-widget.git
cd whisper-widget
launch.bat
```

On first run this will:
1. Create a Python virtual environment
2. Install dependencies from `requirements.txt`
3. Download the speech model (~670 MB INT8) from HuggingFace
4. Show a loading animation while the model initializes
5. Sit in the system tray, ready for `Ctrl+Space`

### Packaged Executable

Download `WhisperWidget.exe` from [Releases](https://github.com/jusedit/whisper-widget/releases) or build it yourself:

```
build.bat
```

The exe is fully self-contained — no Python installation required. It extracts an embedded Python runtime and installs dependencies on first run. App files live in `%LOCALAPPDATA%\WhisperWidget`.

Running a newer exe automatically updates the existing installation (kills the old process, extracts new files, detects dependency changes).

## Usage

| Action | What happens |
|--------|-------------|
| `Ctrl+Space` | Start recording — overlay shows live waveform |
| `Ctrl+Space` again | Stop recording — transcription auto-pastes at cursor |
| Right-click tray icon | Settings, quit |

The microphone is **only active while recording**. No always-on listening.

### Changing the Hotkey

Right-click tray icon > **Settings** > click **Change** > press new key combination.

### Model Quality

Switch between INT8 (fast, ~670 MB) and FP32 (best quality, ~2.5 GB) in Settings. Changing quality triggers a download if the model isn't cached.

### Debug Mode

Enable in Settings to save audio and transcripts to `debug/` for troubleshooting:
- `chunk_NN_raw.wav` / `chunk_NN_vad.wav` — before/after VAD trimming
- `final_raw.wav` / `final_vad.wav` — remainder audio
- `transcript_chunked.txt` — per-chunk transcription
- `transcript_full.txt` — full audio re-transcribed for comparison

## How It Works

```
Hotkey pressed
    -> Mic stream opens (16kHz mono)
    -> Audio analyzed for silence gaps
    -> Completed speech segments transcribed in background (chunked mode)

Hotkey pressed again
    -> Mic closes
    -> Remaining audio transcribed
    -> All chunks combined, copied to clipboard, pasted via Ctrl+V
```

### Audio Pipeline

1. **Recording** — `sounddevice` captures 16kHz mono audio in 1024-sample blocks
2. **Chunking** — silence detection (RMS < 0.005 for 1.5s) splits long recordings at natural pauses
3. **VAD Trimming** — Silero VAD v5 trims leading/trailing silence, keeping internal pauses intact
4. **Preprocessing** — 128-bin log-mel spectrogram via vectorized NumPy STFT
5. **Encoder** — ONNX Runtime runs the Parakeet encoder on CPU
6. **Decoder** — TDT greedy decode with duration prediction
7. **Output** — SentencePiece token IDs decoded to text, artifacts cleaned

### Performance

The app logs detailed timing data with `PERF` prefix for analysis:

```
PERF app.imports: 120ms              # torch, onnxruntime, PyQt6
PERF model.encoder_session: 8500ms   # ONNX session creation
PERF model.decoder_session: 2100ms
PERF model.warmup: 350ms             # eliminates first-inference penalty
PERF model.total: 11200ms
PERF app.startup_total: 12500ms      # process start to ready
PERF transcribe.vad_trim: 45ms       # per-inference breakdown
PERF transcribe.mel: 30ms
PERF transcribe.encoder: 280ms
PERF transcribe.decoder: 250ms
PERF transcribe.total: 620ms (audio=4.5s RTF=0.138)
PERF memory [after_model_load]: rss=450MB peak=520MB
```

Typical real-time factor (RTF) is 0.1–0.2 on a modern CPU (5–10x faster than real-time).

## Project Structure

```
whisper-widget/
  main.py                 # App entry point, hotkey listener, recording flow
  transcriber.py          # Parakeet TDT v3 ONNX inference engine
  recorder.py             # Mic recording with VAD-based chunking
  overlay.py              # Dynamic Island-style floating UI
  config.py               # Settings (JSON + Windows Registry)
  settings_dialog.py      # Settings UI (PyQt6)
  model_downloader.py     # HuggingFace model auto-download with progress
  perf_logger.py          # Performance tracking (PerfTimer, memory logging)
  assets.py               # Sound effect generation
  presplash.pyw           # Splash screen (lightweight, no torch import)
  launcher.py             # Single-exe launcher (compiled by PyInstaller)
  launch.bat              # Dev mode launcher
  build.bat               # Build script for packaged exe
  requirements.txt        # Python dependencies
  favicon.ico             # App icon
  tests/
    benchmark_wer.py      # Word error rate benchmark (Thorsten-DE dataset)
    test_long_audio.py    # Long audio chunking test
```

## Data Locations

| What | Where |
|------|-------|
| Settings | `%APPDATA%\WhisperWidget\settings.json` |
| Model (auto-download) | `%APPDATA%\WhisperWidget\models\parakeet-tdt-v3\` |
| Model (dev/portable) | `./models/parakeet-tdt-v3/` (takes priority if present) |
| Installed app | `%LOCALAPPDATA%\WhisperWidget\` (packaged mode) |
| Logs | `debug/app.log` (relative to app dir) |
| Autostart registry | `HKCU\Software\Microsoft\Windows\CurrentVersion\Run\WhisperWidget` |

## Building

```
build.bat
```

This downloads an embedded Python 3.13 package, bundles it with the app source via PyInstaller, and produces `dist/WhisperWidget.exe` (~23 MB). The exe is fully portable — drop it anywhere.

On first launch, the exe:
1. Extracts Python runtime to `%LOCALAPPDATA%\WhisperWidget\python\`
2. Installs dependencies via pip (~1-2 min)
3. Launches the app

Subsequent launches skip setup and start in seconds. Running a newer exe updates the installation automatically.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Speech model | Parakeet TDT v3 (NVIDIA, ONNX) |
| Inference | ONNX Runtime (CPU) |
| VAD | Silero VAD v5 |
| Audio capture | sounddevice (PortAudio) |
| Preprocessing | NumPy vectorized STFT, 128-bin log-mel |
| UI framework | PyQt6 |
| Paste mechanism | pyperclip + pyautogui |
| Packaging | PyInstaller (single exe) |

## License

[MIT](LICENSE)
