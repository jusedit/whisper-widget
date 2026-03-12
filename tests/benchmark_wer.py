"""Benchmark WER against Thorsten-DE German speech dataset."""

import sys
import os
import re
import time
import numpy as np
import librosa

# Add project to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transcriber import ParakeetTranscriber


def levenshtein_distance(ref_words, hyp_words):
    """Word-level Levenshtein distance."""
    n, m = len(ref_words), len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[n][m]


def normalize_text(text):
    """Normalize text for fair WER comparison."""
    text = text.lower()
    # Remove all punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def compute_wer(ref, hyp, normalized=True):
    """Compute Word Error Rate."""
    if normalized:
        ref = normalize_text(ref)
        hyp = normalize_text(hyp)
    ref_words = ref.lower().split()
    hyp_words = hyp.lower().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    dist = levenshtein_distance(ref_words, hyp_words)
    return dist / len(ref_words)


def main():
    dataset_dir = os.path.join(os.path.dirname(__file__), "thorsten-de")
    wavs_dir = os.path.join(dataset_dir, "wavs")
    metadata_path = os.path.join(dataset_dir, "metadata_train.csv")

    # Load metadata
    metadata = {}
    with open(metadata_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|", 1)
            if len(parts) == 2:
                metadata[parts[0]] = parts[1]

    print(f"Loaded {len(metadata)} metadata entries")

    # Get wav files that have metadata
    wav_files = sorted(os.listdir(wavs_dir))
    matched = []
    for wf in wav_files:
        fid = wf.replace(".wav", "")
        if fid in metadata:
            matched.append((wf, fid, metadata[fid]))

    print(f"Found {len(matched)} wav files with matching metadata")

    # Sample
    num_samples = min(200, len(matched))
    rng = np.random.RandomState(42)
    indices = rng.choice(len(matched), size=num_samples, replace=False)
    samples = [matched[i] for i in sorted(indices)]

    print(f"\nBenchmarking {num_samples} samples...")
    print(f"Loading model...")

    transcriber = ParakeetTranscriber()

    # Track both raw and normalized WER
    total_raw_wer = 0.0
    total_norm_wer = 0.0
    total_audio_sec = 0.0
    total_infer_sec = 0.0
    raw_misses = []
    norm_misses = []
    empty_hyp = []
    short_clips = 0  # audio < 1.5s

    for i, (wav_file, fid, ref_text) in enumerate(samples):
        wav_path = os.path.join(wavs_dir, wav_file)
        try:
            audio, sr = librosa.load(wav_path, sr=16000, mono=True)
        except Exception as e:
            continue

        audio_dur = len(audio) / 16000.0
        total_audio_sec += audio_dur
        if audio_dur < 1.5:
            short_clips += 1

        t0 = time.time()
        hyp = transcriber.transcribe(audio)
        infer_time = time.time() - t0
        total_infer_sec += infer_time

        raw_wer = compute_wer(ref_text, hyp, normalized=False)
        norm_wer = compute_wer(ref_text, hyp, normalized=True)
        total_raw_wer += raw_wer
        total_norm_wer += norm_wer

        if raw_wer > 0.2:
            raw_misses.append((fid, ref_text, hyp, raw_wer, norm_wer, audio_dur))
        if norm_wer > 0.2:
            norm_misses.append((fid, ref_text, hyp, raw_wer, norm_wer, audio_dur))
        if not hyp.strip() or hyp.strip().replace('.', '') == '':
            empty_hyp.append((fid, ref_text, audio_dur))

        if (i + 1) % 10 == 0 or norm_wer > 0.2:
            print(f"  [{i+1:3d}/{num_samples}] raw={raw_wer:.0%} norm={norm_wer:.0%} ({audio_dur:.1f}s -> {infer_time:.2f}s)")
            if norm_wer > 0.2:
                print(f"    REF: {ref_text}")
                print(f"    HYP: {hyp}")

    avg_raw = total_raw_wer / num_samples
    avg_norm = total_norm_wer / num_samples
    rtf = total_infer_sec / total_audio_sec if total_audio_sec > 0 else 0

    print(f"\n{'='*60}")
    print(f"RESULTS: {num_samples} samples")
    print(f"  Raw WER:        {avg_raw:.1%}  (misses>20%: {len(raw_misses)})")
    print(f"  Normalized WER: {avg_norm:.1%}  (misses>20%: {len(norm_misses)})")
    print(f"  Empty outputs:  {len(empty_hyp)}")
    print(f"  Short clips (<1.5s): {short_clips}")
    print(f"  Total audio: {total_audio_sec:.1f}s")
    print(f"  Total inference: {total_infer_sec:.1f}s")
    print(f"  RTF: {rtf:.3f}")

    if norm_misses:
        print(f"\nNORMALIZED MISSES (real errors, WER>20%):")
        for fid, ref, hyp, raw_w, norm_w, dur in norm_misses:
            print(f"  [{norm_w:.0%}] {fid} ({dur:.1f}s)")
            print(f"    REF: {ref}")
            print(f"    HYP: {hyp}")

    # Show punctuation-only misses
    punct_only = [m for m in raw_misses if m[0] not in [n[0] for n in norm_misses]]
    if punct_only:
        print(f"\nPUNCTUATION/CASE ONLY (raw miss but norm OK): {len(punct_only)}")
        for fid, ref, hyp, raw_w, norm_w, dur in punct_only:
            print(f"  [{raw_w:.0%}->{norm_w:.0%}] {ref} -> {hyp}")

    if empty_hyp:
        print(f"\nEMPTY OUTPUTS:")
        for fid, ref, dur in empty_hyp:
            print(f"  {fid} ({dur:.1f}s): {ref}")


if __name__ == "__main__":
    main()
