"""Silero VAD wrapper using pure numpy + ONNX Runtime (no torch dependency).

Drop-in replacement for silero_vad's load_silero_vad() and get_speech_timestamps().
Uses the ONNX model shipped with the silero-vad package.
"""

import os
import numpy as np
import onnxruntime as ort


def load_silero_vad_onnx() -> "SileroVADOnnx":
    """Load Silero VAD ONNX model from the installed silero_vad package.

    Finds the model file directly via importlib.util to avoid importing
    the silero_vad package (which would pull in torch).
    """
    import importlib.util

    spec = importlib.util.find_spec("silero_vad")
    if spec is None or spec.submodule_search_locations is None:
        raise ImportError("silero_vad package not found")

    pkg_dir = list(spec.submodule_search_locations)[0]
    model_path = os.path.join(pkg_dir, "data", "silero_vad.onnx")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Silero VAD ONNX model not found at {model_path}")

    return SileroVADOnnx(model_path)


class SileroVADOnnx:
    """Pure numpy wrapper around the Silero VAD ONNX model."""

    def __init__(self, path: str):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            path, providers=["CPUExecutionProvider"], sess_options=opts,
        )
        self.sample_rates = [8000, 16000]
        self.reset_states()

    def reset_states(self, batch_size: int = 1):
        self._state = np.zeros((2, batch_size, 128), dtype=np.float32)
        self._context = np.zeros(0, dtype=np.float32)
        self._last_sr = 0
        self._last_batch_size = 0

    def __call__(self, x: np.ndarray, sr: int) -> float:
        """Run VAD on a single audio chunk. Returns speech probability."""
        # Ensure 2D: (batch, samples)
        if x.ndim == 1:
            x = x[np.newaxis, :]

        num_samples = 512 if sr == 16000 else 256
        if x.shape[-1] != num_samples:
            raise ValueError(
                f"Expected {num_samples} samples for {sr}Hz, got {x.shape[-1]}"
            )

        batch_size = x.shape[0]
        context_size = 64 if sr == 16000 else 32

        if not self._last_batch_size:
            self.reset_states(batch_size)
        if self._last_sr and self._last_sr != sr:
            self.reset_states(batch_size)
        if self._last_batch_size and self._last_batch_size != batch_size:
            self.reset_states(batch_size)

        if self._context.size == 0:
            self._context = np.zeros((batch_size, context_size), dtype=np.float32)

        x = np.concatenate([self._context, x], axis=1)

        ort_inputs = {
            "input": x.astype(np.float32),
            "state": self._state.astype(np.float32),
            "sr": np.array(sr, dtype=np.int64),
        }
        out, state = self.session.run(None, ort_inputs)
        self._state = state
        self._context = x[..., -context_size:]
        self._last_sr = sr
        self._last_batch_size = batch_size

        return float(out[0, 0])


def get_speech_timestamps(
    audio: np.ndarray,
    model: SileroVADOnnx,
    *,
    threshold: float = 0.5,
    sampling_rate: int = 16000,
    min_speech_duration_ms: int = 250,
    max_speech_duration_s: float = float("inf"),
    min_silence_duration_ms: int = 100,
    speech_pad_ms: int = 30,
    neg_threshold: float | None = None,
    min_silence_at_max_speech: int = 98,
    use_max_poss_sil_at_max_speech: bool = True,
) -> list[dict]:
    """Find speech timestamps in audio using Silero VAD.

    Pure numpy port of silero_vad.utils_vad.get_speech_timestamps().
    Returns list of dicts with 'start' and 'end' keys (in samples).
    """
    audio = np.asarray(audio, dtype=np.float32).ravel()

    if sampling_rate > 16000 and (sampling_rate % 16000 == 0):
        step = sampling_rate // 16000
        sampling_rate = 16000
        audio = audio[::step]
    else:
        step = 1

    if sampling_rate not in (8000, 16000):
        raise ValueError("Supported sampling rates: 8000, 16000 (or multiples of 16000)")

    window_size_samples = 512 if sampling_rate == 16000 else 256

    model.reset_states()
    min_speech_samples = sampling_rate * min_speech_duration_ms / 1000
    speech_pad_samples = sampling_rate * speech_pad_ms / 1000
    max_speech_samples = (
        sampling_rate * max_speech_duration_s - window_size_samples - 2 * speech_pad_samples
    )
    min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
    min_silence_samples_at_max_speech = sampling_rate * min_silence_at_max_speech / 1000

    audio_length_samples = len(audio)

    # Compute speech probabilities per window
    speech_probs = []
    for start in range(0, audio_length_samples, window_size_samples):
        chunk = audio[start : start + window_size_samples]
        if len(chunk) < window_size_samples:
            chunk = np.pad(chunk, (0, window_size_samples - len(chunk)))
        prob = model(chunk, sampling_rate)
        speech_probs.append(prob)

    # State machine to find speech segments
    if neg_threshold is None:
        neg_threshold = max(threshold - 0.15, 0.01)

    triggered = False
    speeches: list[dict] = []
    current_speech: dict = {}
    temp_end = 0
    prev_end = next_start = 0
    possible_ends: list[tuple] = []

    for i, speech_prob in enumerate(speech_probs):
        cur_sample = window_size_samples * i

        if (speech_prob >= threshold) and temp_end:
            sil_dur = cur_sample - temp_end
            if sil_dur > min_silence_samples_at_max_speech:
                possible_ends.append((temp_end, sil_dur))
            temp_end = 0
            if next_start < prev_end:
                next_start = cur_sample

        if (speech_prob >= threshold) and not triggered:
            triggered = True
            current_speech["start"] = cur_sample
            continue

        if triggered and (cur_sample - current_speech["start"] > max_speech_samples):
            if use_max_poss_sil_at_max_speech and possible_ends:
                prev_end, dur = max(possible_ends, key=lambda x: x[1])
                current_speech["end"] = prev_end
                speeches.append(current_speech)
                current_speech = {}
                next_start = prev_end + dur
                if next_start < prev_end + cur_sample:
                    current_speech["start"] = next_start
                else:
                    triggered = False
                prev_end = next_start = temp_end = 0
                possible_ends = []
            else:
                if prev_end:
                    current_speech["end"] = prev_end
                    speeches.append(current_speech)
                    current_speech = {}
                    if next_start < prev_end:
                        triggered = False
                    else:
                        current_speech["start"] = next_start
                    prev_end = next_start = temp_end = 0
                    possible_ends = []
                else:
                    current_speech["end"] = cur_sample
                    speeches.append(current_speech)
                    current_speech = {}
                    prev_end = next_start = temp_end = 0
                    triggered = False
                    possible_ends = []
                    continue

        if (speech_prob < neg_threshold) and triggered:
            if not temp_end:
                temp_end = cur_sample
            sil_dur_now = cur_sample - temp_end

            if (
                not use_max_poss_sil_at_max_speech
                and sil_dur_now > min_silence_samples_at_max_speech
            ):
                prev_end = temp_end

            if sil_dur_now < min_silence_samples:
                continue
            else:
                current_speech["end"] = temp_end
                if (current_speech["end"] - current_speech["start"]) > min_speech_samples:
                    speeches.append(current_speech)
                current_speech = {}
                prev_end = next_start = temp_end = 0
                triggered = False
                possible_ends = []
                continue

    if current_speech and (audio_length_samples - current_speech["start"]) > min_speech_samples:
        current_speech["end"] = audio_length_samples
        speeches.append(current_speech)

    # Apply padding
    for i, speech in enumerate(speeches):
        if i == 0:
            speech["start"] = int(max(0, speech["start"] - speech_pad_samples))
        if i != len(speeches) - 1:
            silence_duration = speeches[i + 1]["start"] - speech["end"]
            if silence_duration < 2 * speech_pad_samples:
                speech["end"] += int(silence_duration // 2)
                speeches[i + 1]["start"] = int(
                    max(0, speeches[i + 1]["start"] - silence_duration // 2)
                )
            else:
                speech["end"] = int(min(audio_length_samples, speech["end"] + speech_pad_samples))
                speeches[i + 1]["start"] = int(
                    max(0, speeches[i + 1]["start"] - speech_pad_samples)
                )
        else:
            speech["end"] = int(min(audio_length_samples, speech["end"] + speech_pad_samples))

    if step > 1:
        for speech in speeches:
            speech["start"] *= step
            speech["end"] *= step

    return speeches
