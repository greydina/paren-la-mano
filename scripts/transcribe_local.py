#!/usr/bin/env python3
"""Transcribe audio files using faster-whisper (small model, CPU int8).

Splits large audio files into chunks to stay within memory limits
(designed for EC2 with 3.8GB RAM).

Usage:
    python scripts/transcribe_local.py <youtube_id>
    python scripts/transcribe_local.py --all
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

AUDIO_DIR = Path(__file__).resolve().parent.parent / "data" / "audio"
TRANSCRIPT_DIR = Path(__file__).resolve().parent.parent / "data" / "transcripts"
SUPPORTED_EXTENSIONS = (".m4a", ".mp4", ".wav", ".mp3", ".ogg", ".webm")

# Chunk duration in seconds — 10 minutes keeps WAV chunks under 20MB
CHUNK_DURATION = 600


def find_audio_file(youtube_id: str) -> Path | None:
    """Find the audio file for a given youtube_id, trying multiple extensions."""
    for ext in SUPPORTED_EXTENSIONS:
        path = AUDIO_DIR / f"{youtube_id}{ext}"
        if path.exists():
            return path
    return None


def list_audio_ids() -> list[str]:
    """Return youtube_ids for all audio files in the audio directory."""
    ids = []
    if not AUDIO_DIR.exists():
        return ids
    for f in sorted(AUDIO_DIR.iterdir()):
        if f.suffix in SUPPORTED_EXTENSIONS:
            ids.append(f.stem)
    return ids


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def get_audio_duration(audio_path: Path) -> float:
    """Get audio duration in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path),
        ],
        capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def extract_chunk(audio_path: Path, start_sec: float, duration: float, out_path: Path) -> None:
    """Extract a chunk of audio using ffmpeg."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-ss", str(start_sec), "-t", str(duration),
            "-i", str(audio_path), "-ar", "16000", "-ac", "1",
            str(out_path),
        ],
        capture_output=True, check=True,
    )


def transcribe_file(model, audio_path: Path, youtube_id: str) -> None:
    """Transcribe a single audio file in chunks and write JSON + TXT outputs."""
    txt_path = TRANSCRIPT_DIR / f"{youtube_id}.txt"
    json_path = TRANSCRIPT_DIR / f"{youtube_id}.json"

    if json_path.exists():
        print(f"  Skipping {youtube_id} (already transcribed)")
        return

    print(f"  Transcribing {audio_path.name} ...")
    start = time.time()

    total_duration = get_audio_duration(audio_path)
    n_chunks = int(total_duration // CHUNK_DURATION) + (
        1 if total_duration % CHUNK_DURATION > 0 else 0
    )
    print(f"  Audio duration: {format_timestamp(total_duration)} ({total_duration:.0f}s)")
    print(f"  Processing in {n_chunks} chunks of {CHUNK_DURATION}s each")

    segments_data = []
    full_text_parts = []

    with tempfile.TemporaryDirectory() as tmpdir:
        chunk_path = Path(tmpdir) / "chunk.wav"

        for chunk_idx in range(n_chunks):
            chunk_start = chunk_idx * CHUNK_DURATION
            chunk_elapsed = time.time() - start

            print(
                f"  Chunk {chunk_idx + 1}/{n_chunks} "
                f"({format_timestamp(chunk_start)}-"
                f"{format_timestamp(min(chunk_start + CHUNK_DURATION, total_duration))}), "
                f"elapsed {chunk_elapsed:.0f}s",
                flush=True,
            )

            # Extract chunk to temp file
            extract_chunk(audio_path, chunk_start, CHUNK_DURATION, chunk_path)

            # Transcribe the chunk
            segments_iter, info = model.transcribe(
                str(chunk_path), language="es", beam_size=1,
                word_timestamps=True,
            )

            for seg in segments_iter:
                # Adjust timestamps to be relative to the full audio
                abs_start = round(seg.start + chunk_start, 3)
                abs_end = round(seg.end + chunk_start, 3)
                text = seg.text.strip()
                if text:
                    words = []
                    if seg.words:
                        for w in seg.words:
                            words.append({
                                "start": round(w.start + chunk_start, 3),
                                "end": round(w.end + chunk_start, 3),
                                "word": w.word,
                                "probability": round(w.probability, 4),
                            })
                    segments_data.append(
                        {"start": abs_start, "end": abs_end, "text": text, "words": words}
                    )
                    full_text_parts.append(text)

            # Remove chunk to free disk
            chunk_path.unlink(missing_ok=True)

    elapsed = time.time() - start

    # Write JSON
    output = {
        "youtube_id": youtube_id,
        "language": "es",
        "duration": round(total_duration, 2),
        "segments": segments_data,
    }
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Write plain text — one segment per line
    txt_path.write_text("\n".join(full_text_parts), encoding="utf-8")

    n_seg = len(segments_data)
    mins = elapsed / 60
    print(
        f"  Done: {n_seg} segments, {format_timestamp(total_duration)} audio, "
        f"took {mins:.1f} min ({elapsed:.0f}s)"
    )
    print(f"  Output: {json_path}")
    print(f"          {txt_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio with faster-whisper"
    )
    parser.add_argument(
        "youtube_id", nargs="?", help="YouTube video ID to transcribe"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Transcribe all audio files in data/audio/",
    )
    args = parser.parse_args()

    if not args.youtube_id and not args.all:
        parser.print_help()
        sys.exit(1)

    # Determine which IDs to process
    if args.all:
        ids = list_audio_ids()
        if not ids:
            print(f"No audio files found in {AUDIO_DIR}")
            sys.exit(1)
        print(f"Found {len(ids)} audio file(s) to process")
    else:
        audio = find_audio_file(args.youtube_id)
        if not audio:
            print(
                f"Error: no audio file found for {args.youtube_id} in {AUDIO_DIR}"
            )
            sys.exit(1)
        ids = [args.youtube_id]

    # Load model once
    print("Loading faster-whisper small model (CPU, int8) ...")
    from faster_whisper import WhisperModel

    model = WhisperModel("small", device="cpu", compute_type="int8")
    print("Model loaded.\n")

    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

    for i, vid_id in enumerate(ids, 1):
        print(f"Episode {i} of {len(ids)}: {vid_id}")
        audio = find_audio_file(vid_id)
        if not audio:
            print(f"  Warning: no audio file found, skipping")
            continue
        transcribe_file(model, audio, vid_id)
        print()

    print("All done.")


if __name__ == "__main__":
    main()
