#!/usr/bin/env python3
"""Tag transcript chunks with speaker identity from diarization output.

The diarize.py script already performs voice-profile matching during
diarization, so segments in data/diarization/<youtube_id>.json already
carry named speakers (Roberto, Luquitas, Joaquin, German, or SPEAKER_N).

This script cross-references those diarization segments with transcript
chunks (from data/embeddings/<youtube_id>_chunks.json) to determine the
dominant speaker for each chunk based on temporal overlap.

Output:
  - Updates data/embeddings/<youtube_id>_chunks.json in place, adding a
    "speaker" field to each chunk record.

Usage:
    python scripts/identify_speakers.py <youtube_id>
    python scripts/identify_speakers.py --all
    python scripts/identify_speakers.py --all --force
"""

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DIARIZATION_DIR = BASE_DIR / "data" / "diarization"
EMBEDDINGS_DIR = BASE_DIR / "data" / "embeddings"
TRANSCRIPTS_DIR = BASE_DIR / "data" / "transcripts"

# Known speakers from voice profiles
KNOWN_SPEAKERS = {"Roberto", "Luquitas", "Joaquin", "German"}


def overlap_duration(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Calculate the overlap duration between two time intervals."""
    overlap_start = max(a_start, b_start)
    overlap_end = min(a_end, b_end)
    return max(0.0, overlap_end - overlap_start)


def load_diarization(youtube_id: str) -> list[dict] | None:
    """Load diarization segments for a youtube_id. Returns None if missing."""
    path = DIARIZATION_DIR / f"{youtube_id}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("segments", [])


def load_chunks(youtube_id: str) -> list[dict] | None:
    """Load transcript chunks for a youtube_id. Returns None if missing."""
    path = EMBEDDINGS_DIR / f"{youtube_id}_chunks.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def assign_speaker_to_chunk(
    chunk_start: float,
    chunk_end: float,
    diarization_segments: list[dict],
) -> tuple[str, float]:
    """Determine the dominant speaker for a chunk based on diarization overlap.

    Returns (speaker_name, confidence) where confidence is the fraction of
    chunk duration covered by the dominant speaker.
    """
    chunk_duration = chunk_end - chunk_start
    if chunk_duration <= 0:
        return ("Unknown", 0.0)

    speaker_overlap: dict[str, float] = {}

    for seg in diarization_segments:
        seg_start = seg["start"]
        seg_end = seg["end"]

        # Quick skip: if segment is entirely before or after the chunk
        if seg_end <= chunk_start or seg_start >= chunk_end:
            continue

        ov = overlap_duration(chunk_start, chunk_end, seg_start, seg_end)
        if ov > 0:
            speaker = seg["speaker"]
            speaker_overlap[speaker] = speaker_overlap.get(speaker, 0.0) + ov

    if not speaker_overlap:
        return ("Unknown", 0.0)

    # Pick the speaker with the most overlap
    dominant_speaker = max(speaker_overlap, key=speaker_overlap.get)
    confidence = speaker_overlap[dominant_speaker] / chunk_duration

    return (dominant_speaker, round(confidence, 3))


def process_episode(youtube_id: str, force: bool = False) -> bool:
    """Tag chunks with speakers for a single episode. Returns True if work was done."""
    chunks = load_chunks(youtube_id)
    if chunks is None:
        print(f"  [{youtube_id}] No chunks file found, skipping.")
        return False

    # Check if already processed (first chunk has speaker field)
    if not force and chunks and "speaker" in chunks[0]:
        print(f"  [{youtube_id}] Already has speaker tags, skipping (use --force to redo).")
        return False

    diar_segments = load_diarization(youtube_id)
    if diar_segments is None:
        print(f"  [{youtube_id}] No diarization file found, skipping.")
        return False

    if not diar_segments:
        print(f"  [{youtube_id}] Diarization has no segments, skipping.")
        return False

    # Tag each chunk with dominant speaker
    known_count = 0
    total_count = len(chunks)
    speaker_chunk_counts: dict[str, int] = {}

    for chunk in chunks:
        start = chunk.get("start_time", 0.0)
        end = chunk.get("end_time", 0.0)

        speaker, confidence = assign_speaker_to_chunk(start, end, diar_segments)
        chunk["speaker"] = speaker
        chunk["speaker_confidence"] = confidence

        speaker_chunk_counts[speaker] = speaker_chunk_counts.get(speaker, 0) + 1
        if speaker in KNOWN_SPEAKERS:
            known_count += 1

    # Save updated chunks
    chunks_path = EMBEDDINGS_DIR / f"{youtube_id}_chunks.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    # Report stats
    known_pct = (known_count / total_count * 100) if total_count > 0 else 0
    print(f"  [{youtube_id}] Tagged {total_count} chunks — {known_count} known speaker ({known_pct:.1f}%)")
    for spk, count in sorted(speaker_chunk_counts.items(), key=lambda x: -x[1]):
        pct = count / total_count * 100
        marker = " *" if spk in KNOWN_SPEAKERS else ""
        print(f"    {spk}: {count} chunks ({pct:.1f}%){marker}")

    return True


def list_processable_ids() -> list[str]:
    """Return youtube_ids that have both diarization and chunks files."""
    chunk_ids = set()
    diar_ids = set()

    if EMBEDDINGS_DIR.exists():
        for p in EMBEDDINGS_DIR.glob("*_chunks.json"):
            # Strip _chunks.json suffix to get youtube_id
            chunk_ids.add(p.name.removesuffix("_chunks.json"))

    if DIARIZATION_DIR.exists():
        for p in DIARIZATION_DIR.glob("*.json"):
            diar_ids.add(p.stem)

    return sorted(chunk_ids & diar_ids)


def main():
    parser = argparse.ArgumentParser(
        description="Tag transcript chunks with speaker identity from diarization"
    )
    parser.add_argument(
        "youtube_id", nargs="?", help="YouTube video ID to process"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Process all episodes that have both diarization and chunks",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-process even if chunks already have speaker tags",
    )
    args = parser.parse_args()

    if not args.youtube_id and not args.all:
        parser.print_help()
        sys.exit(1)

    if args.all:
        ids = list_processable_ids()
        if not ids:
            print("No episodes found with both diarization and chunks data.")
            print(f"  Diarization dir: {DIARIZATION_DIR}")
            print(f"  Embeddings dir:  {EMBEDDINGS_DIR}")
            sys.exit(0)
        print(f"Found {len(ids)} episode(s) to process")
    else:
        ids = [args.youtube_id]

    processed = 0
    total_chunks = 0
    total_known = 0

    for i, yt_id in enumerate(ids, 1):
        print(f"\n[{i}/{len(ids)}] {yt_id}")
        if process_episode(yt_id, force=args.force):
            processed += 1

            # Accumulate stats for summary
            chunks = load_chunks(yt_id)
            if chunks:
                total_chunks += len(chunks)
                total_known += sum(
                    1 for c in chunks
                    if c.get("speaker") in KNOWN_SPEAKERS
                )

    print(f"\nDone. Processed {processed}/{len(ids)} episodes.")
    if total_chunks > 0:
        pct = total_known / total_chunks * 100
        print(f"Overall: {total_known}/{total_chunks} chunks assigned to known speakers ({pct:.1f}%)")
        if pct < 80:
            print("WARNING: Less than 80% of chunks assigned to known speakers.")
            print("  This may indicate missing or poor diarization data.")


if __name__ == "__main__":
    main()
