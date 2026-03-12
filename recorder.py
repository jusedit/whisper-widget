"""Audio recording module using sounddevice with VAD-based chunking."""

import numpy as np
import sounddevice as sd
import threading


class AudioRecorder:
    """Records audio from the default input device at 16kHz mono.

    The mic stream is only active while recording — it opens on start()
    and closes on stop(). No always-on listening.

    Supports chunked mode: detects silence gaps during recording and
    emits completed speech segments via a callback for incremental
    transcription.
    """

    SAMPLE_RATE = 16000
    CHANNELS = 1
    DTYPE = np.float32
    BLOCKSIZE = 1024

    # Silence detection for chunking
    SILENCE_THRESHOLD = 0.005  # RMS below this = silence
    SILENCE_DURATION = 1.5     # seconds of clear silence to trigger chunk boundary
    MIN_CHUNK_SECS = 5.0       # don't emit chunks shorter than 5s
    MAX_CHUNK_SECS = 60.0      # force split at any small pause after this duration

    def __init__(self):
        self._chunks: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._level_callback = None
        self._chunk_callback = None
        self._recording = False
        self._stream = None

        # Silence tracking
        self._silence_blocks = 0
        self._silence_blocks_threshold = int(self.SILENCE_DURATION * self.SAMPLE_RATE / self.BLOCKSIZE)
        self._had_speech = False
        self._chunk_start_idx = 0

    def set_level_callback(self, cb):
        """Set callback(float) called with RMS level 0..1 per block."""
        self._level_callback = cb

    def set_chunk_callback(self, cb):
        """Set callback(np.ndarray) called when a speech chunk is complete."""
        self._chunk_callback = cb

    def start(self):
        with self._lock:
            self._chunks = []
            self._recording = True
            self._silence_blocks = 0
            self._had_speech = False
            self._chunk_start_idx = 0

        # Open mic stream
        self._stream = sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            channels=self.CHANNELS,
            dtype=self.DTYPE,
            blocksize=self.BLOCKSIZE,
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop(self) -> np.ndarray:
        """Stop recording, close mic, return untranscribed remainder."""
        # Close mic stream
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        with self._lock:
            self._recording = False
            remaining = self._chunks[self._chunk_start_idx:]
            self._chunks.clear()
            self._chunk_start_idx = 0
            self._had_speech = False
            if not remaining:
                return np.array([], dtype=self.DTYPE)
            return np.concatenate(remaining)

    def is_recording(self) -> bool:
        return self._recording

    def _audio_callback(self, indata, frames, time_info, status):
        chunk = indata[:, 0].copy()
        rms = float(np.sqrt(np.mean(chunk ** 2)))

        with self._lock:
            if self._recording:
                self._chunks.append(chunk)
                self._detect_chunk_boundary(rms)

        if self._level_callback:
            self._level_callback(min(rms * 5, 1.0))

    def _detect_chunk_boundary(self, rms: float):
        """Detect silence gap and emit completed speech chunk."""
        if rms < self.SILENCE_THRESHOLD:
            self._silence_blocks += 1
        else:
            self._silence_blocks = 0
            self._had_speech = True

        if not self._had_speech or not self._chunk_callback:
            return

        chunk_blocks = len(self._chunks) - self._chunk_start_idx - self._silence_blocks
        chunk_secs = chunk_blocks * self.BLOCKSIZE / self.SAMPLE_RATE

        # Normal silence boundary
        should_emit = (self._silence_blocks >= self._silence_blocks_threshold
                       and chunk_secs >= self.MIN_CHUNK_SECS)

        # Force split at max duration on any small pause (3+ silent blocks ~192ms)
        if not should_emit and chunk_secs >= self.MAX_CHUNK_SECS and self._silence_blocks >= 3:
            should_emit = True

        if should_emit:
            end_idx = len(self._chunks) - self._silence_blocks
            segment = self._chunks[self._chunk_start_idx:end_idx]
            if segment:
                audio = np.concatenate(segment)
                self._chunk_callback(audio)
            # Free emitted chunks to prevent unbounded memory growth
            del self._chunks[:end_idx]
            self._chunk_start_idx = 0
            self._had_speech = False
            self._silence_blocks = 0
