#!/usr/bin/env python3
"""Chunk transcripts and generate embeddings for Paren La Mano episodes.

Reads transcript JSON files (with timed segments from Whisper), chunks them
into ~500-token windows with ~50-token overlap respecting sentence boundaries,
prepends episode metadata for better retrieval, and generates embeddings using
paraphrase-multilingual-MiniLM-L12-v2 (384 dims).

Output per episode:
  data/embeddings/<youtube_id>_chunks.json  — chunk text + metadata
  data/embeddings/<youtube_id>.npy          — numpy embeddings (N x 384)
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
TRANSCRIPTS_DIR = BASE_DIR / "data" / "transcripts"
EMBEDDINGS_DIR = BASE_DIR / "data" / "embeddings"
EPISODES_FILE = BASE_DIR / "data" / "episodes.json"

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_TOKENS = 500
OVERLAP_TOKENS = 50
TOKEN_FACTOR = 1.3  # approx tokens per whitespace-delimited word for Spanish


def estimate_tokens(text: str) -> int:
    """Rough token count: words * factor."""
    return int(len(text.split()) * TOKEN_FACTOR)


def load_episodes() -> dict:
    """Return dict mapping youtube_id -> {titulo, fecha}."""
    if not EPISODES_FILE.exists():
        print(f"Warning: {EPISODES_FILE} not found, metadata will be empty.")
        return {}
    with open(EPISODES_FILE, "r", encoding="utf-8") as f:
        episodes = json.load(f)
    return {
        ep["youtube_id"]: {"titulo": ep["titulo"], "fecha": ep["fecha"]}
        for ep in episodes
    }


# ---------------------------------------------------------------------------
# Transcript loading — prefer JSON (has timing), fall back to plain .txt
# ---------------------------------------------------------------------------

def load_transcript_json(path: Path) -> list[dict]:
    """Load Whisper JSON transcript. Returns list of segments with start, end, text."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    segments = data.get("segments", [])
    if not segments:
        return []
    return [
        {
            "start": float(s.get("start", 0)),
            "end": float(s.get("end", 0)),
            "text": s.get("text", "").strip(),
        }
        for s in segments
        if s.get("text", "").strip()
    ]


def load_transcript_txt(path: Path) -> list[dict]:
    """Load plain text transcript (no timing info)."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    # Treat as single segment spanning 0..0 (unknown timing)
    return [{"start": 0.0, "end": 0.0, "text": text}]


def load_transcript(youtube_id: str) -> list[dict] | None:
    """Load transcript segments for a given youtube_id. Returns None if missing."""
    json_path = TRANSCRIPTS_DIR / f"{youtube_id}.json"
    txt_path = TRANSCRIPTS_DIR / f"{youtube_id}.txt"

    if json_path.exists():
        segments = load_transcript_json(json_path)
        if segments:
            return segments
    if txt_path.exists():
        segments = load_transcript_txt(txt_path)
        if segments:
            return segments
    return None


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def split_sentences(text: str) -> list[str]:
    """Split text on sentence-ending punctuation or newlines."""
    parts = SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in parts if s.strip()]


# ---------------------------------------------------------------------------
# Chunking with timing
# ---------------------------------------------------------------------------

def chunk_segments(
    segments: list[dict],
    max_tokens: int = CHUNK_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[dict]:
    """Chunk transcript segments into overlapping windows respecting sentence boundaries.

    Each returned chunk has: text, token_count, start_time, end_time.
    """
    # Flatten segments into sentence-level units with timing.
    # Each sentence inherits timing from the segment it came from.
    sentence_units: list[dict] = []  # {text, start, end}
    for seg in segments:
        sents = split_sentences(seg["text"])
        if not sents:
            continue
        for sent in sents:
            sentence_units.append({
                "text": sent,
                "start": seg["start"],
                "end": seg["end"],
            })

    if not sentence_units:
        return []

    chunks: list[dict] = []
    current: list[dict] = []  # list of sentence_unit dicts
    current_tokens = 0

    for unit in sentence_units:
        unit_tokens = estimate_tokens(unit["text"])

        if current_tokens + unit_tokens > max_tokens and current:
            # Emit chunk
            chunk_text = " ".join(u["text"] for u in current)
            chunks.append({
                "text": chunk_text,
                "token_count": current_tokens,
                "start_time": current[0]["start"],
                "end_time": current[-1]["end"],
            })

            # Rewind for overlap
            overlap_acc = 0
            rewind_idx = len(current)
            for i in range(len(current) - 1, -1, -1):
                t = estimate_tokens(current[i]["text"])
                if overlap_acc + t > overlap_tokens:
                    break
                overlap_acc += t
                rewind_idx = i
            current = current[rewind_idx:]
            current_tokens = sum(estimate_tokens(u["text"]) for u in current)

        current.append(unit)
        current_tokens += unit_tokens

    # Final chunk
    if current:
        chunk_text = " ".join(u["text"] for u in current)
        chunks.append({
            "text": chunk_text,
            "token_count": current_tokens,
            "start_time": current[0]["start"],
            "end_time": current[-1]["end"],
        })

    return chunks


# ---------------------------------------------------------------------------
# Per-episode processing
# ---------------------------------------------------------------------------

def process_episode(youtube_id: str, episodes: dict, model: SentenceTransformer) -> bool:
    """Process a single episode. Returns True if work was done."""
    chunks_path = EMBEDDINGS_DIR / f"{youtube_id}_chunks.json"
    embeddings_path = EMBEDDINGS_DIR / f"{youtube_id}.npy"

    if chunks_path.exists() and embeddings_path.exists():
        print(f"  [{youtube_id}] Already processed, skipping.")
        return False

    segments = load_transcript(youtube_id)
    if segments is None:
        print(f"  [{youtube_id}] Transcript not found, skipping.")
        return False

    # Episode metadata
    meta = episodes.get(youtube_id, {})
    episode_title = meta.get("titulo", "Desconocido")
    episode_date = meta.get("fecha", "")
    header = f"Episodio: {episode_title} ({episode_date})"

    # Chunk
    chunks = chunk_segments(segments)
    if not chunks:
        print(f"  [{youtube_id}] No chunks produced (empty transcript?), skipping.")
        return False

    total_chars = sum(len(c["text"]) for c in chunks)
    print(f"  [{youtube_id}] {len(chunks)} chunks from {total_chars} chars")

    # Prepend metadata to each chunk text for embedding
    texts_for_embedding = [f"{header}\n\n{c['text']}" for c in chunks]

    # Build chunk metadata records
    chunk_records = []
    for i, c in enumerate(chunks):
        chunk_records.append({
            "chunk_index": i,
            "chunk_id": f"{youtube_id}_{i:04d}",
            "youtube_id": youtube_id,
            "episode_title": episode_title,
            "episode_date": episode_date,
            "start_time": round(c["start_time"], 2),
            "end_time": round(c["end_time"], 2),
            "text": texts_for_embedding[i],
            "token_count": c["token_count"],
        })

    # Generate embeddings (batch, fits in memory for typical episode sizes)
    embeddings = model.encode(
        texts_for_embedding,
        convert_to_numpy=True,
        show_progress_bar=len(texts_for_embedding) > 20,
        batch_size=32,
    )

    # Save
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunk_records, f, ensure_ascii=False, indent=2)
    np.save(embeddings_path, embeddings)

    print(f"  [{youtube_id}] Saved {len(chunks)} chunks + embeddings {embeddings.shape}")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate chunks and embeddings for PLM transcripts"
    )
    parser.add_argument("youtube_id", nargs="?", help="YouTube ID to process")
    parser.add_argument("--all", action="store_true", help="Process all transcripts in data/transcripts/")
    parser.add_argument("--force", action="store_true", help="Re-process even if output exists")
    args = parser.parse_args()

    if not args.youtube_id and not args.all:
        parser.error("Provide a youtube_id or --all")

    episodes = load_episodes()

    # Determine which IDs to process
    if args.all:
        # Collect IDs from both .json and .txt transcripts (prefer .json)
        ids_set = set()
        for p in TRANSCRIPTS_DIR.glob("*.json"):
            ids_set.add(p.stem)
        for p in TRANSCRIPTS_DIR.glob("*.txt"):
            ids_set.add(p.stem)
        ids = sorted(ids_set)
        if not ids:
            print("No transcripts found in", TRANSCRIPTS_DIR)
            sys.exit(0)
    else:
        ids = [args.youtube_id]

    # If --force, remove existing outputs so they get re-generated
    if args.force:
        for yt_id in ids:
            for suffix in ("_chunks.json", ".npy"):
                p = EMBEDDINGS_DIR / f"{yt_id}{suffix}"
                if p.exists():
                    p.unlink()

    print(f"Loading model ({MODEL_NAME})...")
    model = SentenceTransformer(MODEL_NAME)

    processed = 0
    for i, yt_id in enumerate(ids, 1):
        print(f"[{i}/{len(ids)}] Processing {yt_id}")
        if process_episode(yt_id, episodes, model):
            processed += 1

    print(f"\nDone. Processed {processed}/{len(ids)} episodes.")


if __name__ == "__main__":
    main()
