"""Parakeet TDT v3 ONNX transcription engine."""

import os
import re
import time
import logging
import numpy as np
import onnxruntime as ort
from pathlib import Path
from vad import load_silero_vad_onnx, get_speech_timestamps
from perf_logger import PerfTimer

log = logging.getLogger("whisper")

# Default model path (local dev/portable mode)
DEFAULT_MODEL_DIR = Path(__file__).parent / "models" / "parakeet-tdt-v3"

# Audio preprocessing constants matching NeMo's AudioToMelSpectrogramPreprocessor
SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 160       # 10ms stride at 16kHz
WIN_LENGTH = 400       # 25ms window at 16kHz
N_MELS = 128
PREEMPH = 0.97
LOG_GUARD = 2**-24  # ~5.96e-8, matching parakeet-rs

# TDT constants
BLANK_IDX = 8192
VOCAB_SIZE = 8193
N_DURATIONS = 5

# Cached mel filterbank and window (computed once)
_MEL_FB = None
_WINDOW = None


def _hz_to_mel(hz):
    """Convert Hz to Mel scale (Slaney/O'Shaughnessy, matches librosa default)."""
    hz = np.asarray(hz, dtype=np.float64)
    with np.errstate(divide="ignore"):
        mel = np.where(
            hz < 1000.0,
            3.0 * hz / 200.0,
            15.0 + 27.0 * np.log(np.maximum(hz, 1e-10) / 1000.0) / np.log(6.4),
        )
    return mel


def _mel_to_hz(mel):
    """Convert Mel scale to Hz (Slaney/O'Shaughnessy)."""
    mel = np.asarray(mel, dtype=np.float64)
    hz = np.where(
        mel < 15.0,
        200.0 * mel / 3.0,
        1000.0 * np.exp((mel - 15.0) * np.log(6.4) / 27.0),
    )
    return hz


def _mel_filterbank(sr, n_fft, n_mels):
    """Compute mel filterbank matrix (matches librosa.filters.mel with Slaney norm)."""
    fmin, fmax = 0.0, sr / 2.0
    mel_min = _hz_to_mel(fmin)
    mel_max = _hz_to_mel(fmax)

    # n_mels + 2 linearly spaced points in mel scale
    mels = np.linspace(float(mel_min), float(mel_max), n_mels + 2)
    hz = _mel_to_hz(mels)

    # FFT bin frequencies
    fft_freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    # Create triangular filters (vectorized)
    fdiff = np.diff(hz)
    ramps = hz[:, np.newaxis] - fft_freqs[np.newaxis, :]

    weights = np.zeros((n_mels, len(fft_freqs)), dtype=np.float64)
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    # Slaney normalization
    enorm = 2.0 / (hz[2 : n_mels + 2] - hz[:n_mels])
    weights *= enorm[:, np.newaxis]

    return weights.astype(np.float32)


def _init_mel():
    global _MEL_FB, _WINDOW
    if _MEL_FB is None:
        _MEL_FB = _mel_filterbank(SAMPLE_RATE, N_FFT, N_MELS)
        # Symmetric Hann window matching parakeet-rs, zero-padded to N_FFT
        win = np.hanning(WIN_LENGTH).astype(np.float32)
        _WINDOW = np.zeros(N_FFT, dtype=np.float32)
        _WINDOW[:WIN_LENGTH] = win


def compute_mel_spectrogram(audio: np.ndarray) -> np.ndarray:
    """Compute 128-bin log-mel spectrogram matching parakeet-rs preprocessing.

    Uses vectorized NumPy STFT (no librosa.stft overhead).
    """
    _init_mel()

    # Pre-emphasis
    emphasized = np.empty(len(audio), dtype=np.float32)
    emphasized[0] = audio[0]
    emphasized[1:] = audio[1:] - PREEMPH * audio[:-1]

    # Zero-pad (center=True, pad_mode='constant')
    pad = N_FFT // 2
    padded = np.pad(emphasized, (pad, pad), mode='constant')

    # Vectorized framing using stride tricks
    n_frames = 1 + (len(padded) - N_FFT) // HOP_LENGTH
    strides = (padded.strides[0] * HOP_LENGTH, padded.strides[0])
    frames = np.lib.stride_tricks.as_strided(padded, shape=(n_frames, N_FFT), strides=strides).copy()

    # Apply window and FFT
    frames *= _WINDOW
    spectrum = np.fft.rfft(frames, n=N_FFT)
    power = np.abs(spectrum) ** 2

    # Mel filterbank + log
    mel = power @ _MEL_FB.T  # (time, n_mels)
    log_mel = np.log(mel + LOG_GUARD)

    # Per-feature normalization (Bessel's correction ddof=1, matching parakeet-rs)
    mel_t = log_mel.T  # (n_mels, time)
    mean = mel_t.mean(axis=1, keepdims=True)
    std = np.maximum(mel_t.std(axis=1, ddof=1, keepdims=True), 1e-5)
    mel_t = (mel_t - mean) / std

    return mel_t.astype(np.float32)  # (n_mels, time)


class ParakeetTranscriber:
    """Offline Parakeet TDT v3 transcriber using ONNX Runtime."""

    def __init__(self, model_dir: str | Path | None = None, quality: str | None = None):
        model_dir = Path(model_dir or DEFAULT_MODEL_DIR)

        with PerfTimer("model.vocab"):
            self._vocab = self._load_vocab(model_dir / "vocab.txt")

        with PerfTimer("model.vad"):
            self._vad = load_silero_vad_onnx()

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 2
        opts.intra_op_num_threads = 4
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = ['CPUExecutionProvider']

        # Use configured quality, fall back to whatever exists
        if quality is None:
            from config import Config
            quality = Config().model_quality

        if quality == "int8" and (model_dir / "encoder-model.int8.onnx").exists():
            encoder_name = "encoder-model.int8.onnx"
            decoder_name = "decoder_joint-model.int8.onnx"
        elif (model_dir / "encoder-model.onnx").exists():
            encoder_name = "encoder-model.onnx"
            decoder_name = "decoder_joint-model.onnx"
        else:
            encoder_name = "encoder-model.int8.onnx"
            decoder_name = "decoder_joint-model.int8.onnx"

        log.info("Loading ONNX sessions: %s, %s", encoder_name, decoder_name)

        with PerfTimer("model.encoder_session"):
            self._encoder = ort.InferenceSession(
                str(model_dir / encoder_name),
                sess_options=opts,
                providers=providers,
            )

        with PerfTimer("model.decoder_session"):
            self._decoder_joint = ort.InferenceSession(
                str(model_dir / decoder_name),
                sess_options=opts,
                providers=providers,
            )

    def warmup(self):
        """Run minimal inference to warm up ONNX Runtime JIT/memory pools.

        Eliminates the ~6x first-inference penalty observed in production.
        """
        dummy_mel = np.zeros((1, 128, 10), dtype=np.float32)
        dummy_len = np.array([10], dtype=np.int64)
        enc_out, enc_len = self._encoder.run(
            None, {"audio_signal": dummy_mel, "length": dummy_len}
        )
        state1 = np.zeros((2, 1, 640), dtype=np.float32)
        state2 = np.zeros((2, 1, 640), dtype=np.float32)
        targets = np.array([[BLANK_IDX]], dtype=np.int32)
        target_len = np.array([1], dtype=np.int32)
        self._decoder_joint.run(None, {
            "encoder_outputs": enc_out[:, :, 0:1],
            "targets": targets,
            "target_length": target_len,
            "input_states_1": state1,
            "input_states_2": state2,
        })

    @staticmethod
    def _load_vocab(path: Path) -> list[str]:
        tokens = []
        with open(path, encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                tokens.append(parts[0] if parts else '')
        return tokens

    def _vad_trim(self, audio: np.ndarray) -> np.ndarray:
        """Trim leading/trailing silence using Silero VAD.

        Only trims the edges — internal pauses are kept intact so
        the model sees natural speech flow.
        """
        stamps = get_speech_timestamps(audio, self._vad, sampling_rate=SAMPLE_RATE)
        if not stamps:
            return audio  # no speech detected, keep all
        # Trim to [first_speech_start - pad, last_speech_end + pad]
        start_pad = 4800  # 300ms before first speech (protect first word)
        end_pad = 1600    # 100ms after last speech
        start = max(0, stamps[0]["start"] - start_pad)
        end = min(len(audio), stamps[-1]["end"] + end_pad)
        return audio[start:end]

    # Max chunk duration for encoder safety (30s audio = ~3000 mel frames)
    MAX_CHUNK_SECS = 30.0

    def transcribe_chunked(self, audio: np.ndarray) -> str:
        """Transcribe long audio by VAD-splitting into chunks.

        Uses Silero VAD to find speech segments, groups them into
        chunks of at most MAX_CHUNK_SECS, and transcribes each chunk.
        """
        if len(audio) < 1600:
            return ""

        max_chunk_samples = int(self.MAX_CHUNK_SECS * SAMPLE_RATE)

        # Short audio: transcribe directly
        if len(audio) <= max_chunk_samples:
            return self.transcribe(audio)

        t_total = time.perf_counter()
        total_dur = len(audio) / SAMPLE_RATE

        with PerfTimer("transcribe_chunked.vad"):
            stamps = get_speech_timestamps(audio, self._vad, sampling_rate=SAMPLE_RATE)

        if not stamps:
            return ""

        # Group speech segments into chunks that fit within max duration
        pad_before = 4800   # 300ms context before
        pad_after = 1600    # 100ms context after
        chunks = []
        cur_start = stamps[0]["start"]
        cur_end = stamps[0]["end"]

        for seg in stamps[1:]:
            if seg["end"] - cur_start > max_chunk_samples:
                chunks.append((cur_start, cur_end))
                cur_start = seg["start"]
            cur_end = seg["end"]
        chunks.append((cur_start, cur_end))

        # Transcribe each chunk
        texts = []
        for i, (start, end) in enumerate(chunks):
            s = max(0, start - pad_before)
            e = min(len(audio), end + pad_after)
            chunk_audio = audio[s:e]
            chunk_dur = len(chunk_audio) / SAMPLE_RATE

            t0 = time.perf_counter()
            text = self.transcribe(chunk_audio)
            dur = time.perf_counter() - t0
            log.info("Chunk %d/%d: %.1fs audio -> %.2fs inference: %s",
                     i + 1, len(chunks), chunk_dur, dur, text[:60])
            if text.strip():
                texts.append(text)

        total_ms = (time.perf_counter() - t_total) * 1000
        log.info("PERF transcribe_chunked.total: %.0fms (audio=%.1fs, %d chunks)",
                 total_ms, total_dur, len(chunks))

        return " ".join(texts)

    def transcribe(self, audio: np.ndarray, debug_prefix: str = "") -> str:
        """Transcribe 16kHz float32 mono audio to text.

        If debug_prefix is set, saves pre/post VAD audio to debug/ folder.
        """
        if len(audio) < 1600:
            return ""

        t_total = time.perf_counter()
        audio_duration = len(audio) / SAMPLE_RATE

        # Pad 200ms silence at start so first word isn't clipped
        # (mic opens with speech already in progress, no natural lead-in)
        lead_pad = np.zeros(3200, dtype=np.float32)  # 200ms at 16kHz
        audio = np.concatenate([lead_pad, audio])

        if debug_prefix:
            self._save_debug(f"{debug_prefix}_raw.wav", audio)

        with PerfTimer("transcribe.vad_trim"):
            audio = self._vad_trim(audio)
        if len(audio) < 1600:
            return ""

        if debug_prefix:
            self._save_debug(f"{debug_prefix}_vad.wav", audio)

        with PerfTimer("transcribe.mel"):
            mel = compute_mel_spectrogram(audio)  # (128, T)

        mel_input = mel[np.newaxis, :, :]  # (1, 128, T)
        length = np.array([mel.shape[1]], dtype=np.int64)

        with PerfTimer("transcribe.encoder"):
            enc_out, enc_len = self._encoder.run(
                None,
                {"audio_signal": mel_input, "length": length},
            )

        with PerfTimer("transcribe.decoder"):
            text = self._tdt_greedy_decode(enc_out, int(enc_len[0]))

        total_ms = (time.perf_counter() - t_total) * 1000
        rtf = (total_ms / 1000) / audio_duration if audio_duration > 0 else 0
        log.info("PERF transcribe.total: %.0fms (audio=%.1fs RTF=%.3f)", total_ms, audio_duration, rtf)

        return text

    @staticmethod
    def _save_debug(filename: str, audio: np.ndarray):
        """Save debug audio to debug/ folder."""
        import wave
        debug_dir = Path(__file__).parent / "debug"
        debug_dir.mkdir(exist_ok=True)
        path = debug_dir / filename
        with wave.open(str(path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
            wf.writeframes(pcm.tobytes())

    def _tdt_greedy_decode(self, enc_out: np.ndarray, enc_len: int) -> str:
        """Greedy TDT decode with duration prediction."""
        state1 = np.zeros((2, 1, 640), dtype=np.float32)
        state2 = np.zeros((2, 1, 640), dtype=np.float32)

        tokens = []
        t = 0
        targets = np.array([[BLANK_IDX]], dtype=np.int32)
        target_len = np.array([1], dtype=np.int32)

        max_steps = enc_len * 10  # safety limit
        steps = 0

        while t < enc_len and steps < max_steps:
            steps += 1

            enc_slice = enc_out[:, :, t:t+1]  # (1, 1024, 1)

            outputs, _, state1, state2 = self._decoder_joint.run(
                None,
                {
                    "encoder_outputs": enc_slice,
                    "targets": targets,
                    "target_length": target_len,
                    "input_states_1": state1,
                    "input_states_2": state2,
                },
            )

            logits = outputs[0, 0, 0, :]  # (8198,)
            token_logits = logits[:VOCAB_SIZE]
            duration_logits = logits[VOCAB_SIZE:]

            token_id = int(np.argmax(token_logits))
            dur = int(np.argmax(duration_logits))

            if token_id == BLANK_IDX:
                # Blank: advance encoder by duration (at least 1)
                t += max(1, dur)
            else:
                # Non-blank: emit token, update decoder input, advance
                tokens.append(token_id)
                targets = np.array([[token_id]], dtype=np.int32)
                # TDT: for non-blank, advance by duration
                t += max(1, dur)

        return self._decode_tokens(tokens)

    def _decode_tokens(self, token_ids: list[int]) -> str:
        """Convert token IDs to text using SentencePiece-style decoding."""
        pieces = []
        for tid in token_ids:
            if 0 <= tid < len(self._vocab):
                token = self._vocab[tid]
                if token.startswith('<') and token.endswith('>'):
                    continue
                pieces.append(token)

        text = ''.join(pieces)
        text = text.replace('\u2581', ' ')
        # Clean TDT artifacts: collapse multi-dots, strip leading dots
        text = re.sub(r'\.{2,}', ' ', text)  # "...." -> " "
        text = re.sub(r'^\s*\.?\s+', '', text)  # leading whitespace/dot
        text = re.sub(r'\s+', ' ', text)  # collapse whitespace
        return text.strip()
