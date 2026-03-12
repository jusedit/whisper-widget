"""Test chunking on a long audio segment with final config."""

import sys
import os
import re
import time
import numpy as np
import librosa

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(encoding='utf-8')

from transcriber import ParakeetTranscriber

MP3_PATH = os.path.join(
    os.path.dirname(__file__),
    "André Stern familylab-Vortrag »Und ich war nie in der Schule« - familylab Deutschland (128k).mp3",
)

SAMPLE_RATE = 16000
BLOCKSIZE = 1024


def detect_chunks(audio, silence_threshold=0.005, silence_duration=1.5,
                  min_chunk_secs=5.0, max_chunk_secs=60.0):
    """Simulate chunk detection with max chunk duration."""
    silence_blocks_threshold = int(silence_duration * SAMPLE_RATE / BLOCKSIZE)

    chunks = []
    chunk_start = 0
    silence_blocks = 0
    had_speech = False

    n_blocks = len(audio) // BLOCKSIZE

    for i in range(n_blocks):
        block = audio[i * BLOCKSIZE:(i + 1) * BLOCKSIZE]
        rms = float(np.sqrt(np.mean(block ** 2)))

        if rms < silence_threshold:
            silence_blocks += 1
        else:
            silence_blocks = 0
            had_speech = True

        if not had_speech:
            continue

        chunk_blocks = (i + 1) * BLOCKSIZE - chunk_start
        chunk_secs = chunk_blocks / SAMPLE_RATE - (silence_blocks * BLOCKSIZE / SAMPLE_RATE)

        # Normal silence boundary
        should_emit = (silence_blocks >= silence_blocks_threshold
                       and chunk_secs >= min_chunk_secs)

        # Force split at max duration on any small pause
        if not should_emit and chunk_secs >= max_chunk_secs and silence_blocks >= 3:
            should_emit = True

        if should_emit:
            end_sample = (i + 1 - silence_blocks) * BLOCKSIZE
            if end_sample > chunk_start:
                chunks.append(audio[chunk_start:end_sample])
            chunk_start = end_sample
            had_speech = False
            silence_blocks = 0

    # Remainder
    if chunk_start < len(audio):
        remainder = audio[chunk_start:]
        if len(remainder) >= 1600:
            chunks.append(remainder)

    return chunks


def main():
    # Load 5-minute segment from ~10 min in
    offset_sec = 600
    duration_sec = 300

    print(f"Loading {duration_sec}s segment starting at {offset_sec}s...")
    audio, sr = librosa.load(MP3_PATH, sr=16000, mono=True,
                             offset=offset_sec, duration=duration_sec)
    actual_dur = len(audio) / 16000
    print(f"Loaded {actual_dur:.1f}s of audio\n")

    print("Loading model...")
    transcriber = ParakeetTranscriber()

    # Final config
    print(f"\n{'='*70}")
    print(f"CHUNKED TRANSCRIPTION (max=60s, silence=1.5s)")
    print(f"{'='*70}")

    chunks = detect_chunks(audio, max_chunk_secs=60.0)
    print(f"Detected {len(chunks)} chunks:")
    for i, c in enumerate(chunks):
        print(f"  Chunk {i+1}: {len(c)/16000:.1f}s")

    t0 = time.time()
    chunk_texts = []
    for i, chunk in enumerate(chunks):
        ct0 = time.time()
        text = transcriber.transcribe(chunk)
        ct = time.time() - ct0
        chunk_dur = len(chunk) / 16000
        preview = text[:80].replace('\n', ' ')
        print(f"  Chunk {i+1}: {chunk_dur:.1f}s -> {ct:.2f}s: {preview}...")
        chunk_texts.append(text)

    total_time = time.time() - t0

    # Filter dot-only artifacts
    cleaned = [t for t in chunk_texts
               if t.strip() and re.sub(r'[.\s]+', '', t)]
    full_text = " ".join(cleaned)

    print(f"\nTime: {total_time:.1f}s (RTF: {total_time/actual_dur:.3f})")
    print(f"Words: {len(full_text.split())}")
    print(f"\n--- FULL TRANSCRIPT ---")
    words = full_text.split()
    line = ""
    for w in words:
        if len(line) + len(w) + 1 > 100:
            print(line)
            line = w
        else:
            line = (line + " " + w).strip()
    if line:
        print(line)

    # Test a different 5-min segment (earlier in talk, likely cleaner)
    print(f"\n\n{'='*70}")
    print(f"BONUS: Testing 2 min from minute 3 (likely cleaner speech)")
    print(f"{'='*70}")

    audio2, _ = librosa.load(MP3_PATH, sr=16000, mono=True,
                             offset=180, duration=120)
    dur2 = len(audio2) / 16000

    chunks2 = detect_chunks(audio2, max_chunk_secs=60.0)
    print(f"Detected {len(chunks2)} chunks:")
    for i, c in enumerate(chunks2):
        print(f"  Chunk {i+1}: {len(c)/16000:.1f}s")

    t0 = time.time()
    texts2 = []
    for i, chunk in enumerate(chunks2):
        ct0 = time.time()
        text = transcriber.transcribe(chunk)
        ct = time.time() - ct0
        preview = text[:80].replace('\n', ' ')
        print(f"  Chunk {i+1}: {len(chunk)/16000:.1f}s -> {ct:.2f}s: {preview}...")
        texts2.append(text)

    time2 = time.time() - t0
    cleaned2 = [t for t in texts2 if t.strip() and re.sub(r'[.\s]+', '', t)]
    text2 = " ".join(cleaned2)

    print(f"\nTime: {time2:.1f}s (RTF: {time2/dur2:.3f})")
    print(f"\n--- TRANSCRIPT ---")
    words = text2.split()
    line = ""
    for w in words:
        if len(line) + len(w) + 1 > 100:
            print(line)
            line = w
        else:
            line = (line + " " + w).strip()
    if line:
        print(line)


if __name__ == "__main__":
    main()
