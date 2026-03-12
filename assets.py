"""Generate and manage app assets (sounds, icons)."""

import math
import struct
import wave
import os
from pathlib import Path

ASSETS_DIR = Path(__file__).parent / "assets"


def _generate_blop(filename: str, freq_start: float, freq_end: float,
                   duration: float = 0.1, sample_rate: int = 44100,
                   volume: float = 0.3):
    """Generate a soft blop sound as a WAV file."""
    n_samples = int(sample_rate * duration)
    samples = []

    for i in range(n_samples):
        t = i / sample_rate
        progress = i / n_samples

        # Frequency sweep
        freq = freq_start + (freq_end - freq_start) * progress

        # Envelope: quick attack, exponential decay
        attack = min(1.0, t / 0.005)  # 5ms attack
        decay = math.exp(-t * 30)      # fast decay
        envelope = attack * decay

        # Main tone + soft harmonic
        sample = math.sin(2 * math.pi * freq * t)
        sample += 0.3 * math.sin(2 * math.pi * freq * 2 * t)  # octave harmonic
        sample += 0.1 * math.sin(2 * math.pi * freq * 3 * t)  # 3rd harmonic

        sample *= envelope * volume
        samples.append(max(-1.0, min(1.0, sample)))

    # Convert to 16-bit PCM
    pcm = b"".join(struct.pack("<h", int(s * 32767)) for s in samples)

    filepath = ASSETS_DIR / filename
    with wave.open(str(filepath), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


def ensure_assets():
    """Generate all assets if they don't exist."""
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    start_path = ASSETS_DIR / "start.wav"
    stop_path = ASSETS_DIR / "stop.wav"

    if not start_path.exists():
        # Rising blop: G5 -> C6 (activation)
        _generate_blop("start.wav", freq_start=784, freq_end=1047,
                       duration=0.09, volume=0.25)

    if not stop_path.exists():
        # Falling blop: E5 -> C5 (deactivation)
        _generate_blop("stop.wav", freq_start=659, freq_end=523,
                       duration=0.08, volume=0.2)


def get_sound_path(name: str) -> str:
    """Get path to a sound asset."""
    return str(ASSETS_DIR / name)
