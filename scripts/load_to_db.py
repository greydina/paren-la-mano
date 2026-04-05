#!/usr/bin/env python3
"""Load transcribed episodes into the PLM PostgreSQL + pgvector database.

Reads episode metadata, transcript text, chunk JSON and numpy embeddings
produced by the upstream pipeline (download_youtube -> transcribe_local ->
generate_chunks) and upserts them into the videos / transcripts / chunks
tables in plm_db.

Usage:
    python scripts/load_to_db.py --all          # load every available episode
    python scripts/load_to_db.py <youtube_id>   # load a single episode
    python scripts/load_to_db.py --all --force   # reload even if already present
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import psycopg2
from psycopg2.extras import execute_values

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "plm_db",
    "user": "plm_user",
    "password": "plm_pass_2026",
}

EMBEDDING_DIM = 384  # paraphrase-multilingual-MiniLM-L12-v2

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
EPISODES_FILE = DATA_DIR / "episodes.json"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def ensure_schema(conn):
    """Ensure the chunks table has the right column types for our pipeline.

    Handles two migrations:
      1. embedding vector dimension: 1024 -> 384  (legacy TikTok data)
      2. time columns: start_char/end_char -> start_time/end_time (float)
    """
    with conn.cursor() as cur:
        # --- 1. Vector dimension -------------------------------------------
        cur.execute("""
            SELECT atttypmod FROM pg_attribute
            JOIN pg_class ON pg_class.oid = pg_attribute.attrelid
            WHERE pg_class.relname = 'chunks' AND pg_attribute.attname = 'embedding'
        """)
        row = cur.fetchone()
        if row is not None:
            current_dim = row[0]
            if current_dim != EMBEDDING_DIM:
                print(f"Altering chunks.embedding from vector({current_dim}) to vector({EMBEDDING_DIM})...")
                cur.execute("DELETE FROM chunks")  # must clear incompatible vectors
                cur.execute("DROP INDEX IF EXISTS chunks_embedding_idx")
                cur.execute(
                    f"ALTER TABLE chunks ALTER COLUMN embedding TYPE vector({EMBEDDING_DIM})"
                )
                cur.execute("""
                    CREATE INDEX chunks_embedding_idx
                    ON chunks USING hnsw (embedding vector_cosine_ops)
                """)
                conn.commit()
                print("  Vector dimension updated.")

        # --- 2. Time columns -----------------------------------------------
        # The generate_chunks.py pipeline outputs start_time / end_time (floats
        # in seconds).  The original schema had start_char / end_char (ints).
        # Migrate if needed.
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'chunks' AND column_name = 'start_char'
        """)
        if cur.fetchone() is not None:
            print("Migrating chunks: start_char/end_char -> start_time/end_time ...")
            cur.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS start_char")
            cur.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS end_char")
            cur.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS start_time DOUBLE PRECISION")
            cur.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS end_time DOUBLE PRECISION")
            conn.commit()
            print("  Time columns migrated.")
        else:
            # Ensure columns exist even on fresh schemas
            cur.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'chunks' AND column_name = 'start_time'
            """)
            if cur.fetchone() is None:
                cur.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS start_time DOUBLE PRECISION")
                cur.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS end_time DOUBLE PRECISION")
                conn.commit()

        # --- 3. Speaker columns -----------------------------------------------
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'chunks' AND column_name = 'speaker'
        """)
        if cur.fetchone() is None:
            print("Adding speaker columns to chunks table...")
            cur.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS speaker TEXT")
            cur.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS speaker_confidence DOUBLE PRECISION")
            cur.execute("CREATE INDEX IF NOT EXISTS chunks_speaker_idx ON chunks (speaker)")
            conn.commit()
            print("  Speaker columns added.")


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_episodes():
    with open(EPISODES_FILE) as f:
        return json.load(f)


def find_episode(episodes, youtube_id):
    for ep in episodes:
        if ep["youtube_id"] == youtube_id:
            return ep
    return None


def get_available_ids():
    """Return youtube_ids that have both chunks and embeddings files."""
    ids = set()
    for p in EMBEDDINGS_DIR.glob("*_chunks.json"):
        yt_id = p.name.replace("_chunks.json", "")
        npy = EMBEDDINGS_DIR / f"{yt_id}.npy"
        if npy.exists():
            ids.add(yt_id)
    return sorted(ids)


def already_loaded(cur, youtube_id):
    """Check if video has chunks already loaded."""
    cur.execute("SELECT COUNT(*) FROM chunks WHERE video_id = %s", (youtube_id,))
    return cur.fetchone()[0] > 0


# ---------------------------------------------------------------------------
# Upsert functions
# ---------------------------------------------------------------------------

def upsert_video(cur, ep):
    yt_id = ep["youtube_id"]
    url = f"https://www.youtube.com/watch?v={yt_id}"
    upload_date = ep.get("fecha")
    cur.execute("""
        INSERT INTO videos (video_id, url, title, description, upload_date)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (video_id) DO UPDATE SET
            title = EXCLUDED.title,
            description = EXCLUDED.description,
            upload_date = EXCLUDED.upload_date
    """, (yt_id, url, ep.get("titulo"), ep.get("descripcion"), upload_date))


def upsert_transcript(cur, youtube_id):
    txt_path = TRANSCRIPTS_DIR / f"{youtube_id}.txt"
    json_path = TRANSCRIPTS_DIR / f"{youtube_id}.json"
    if not txt_path.exists():
        print(f"  Warning: transcript text file not found: {txt_path}")
        return False
    full_text = txt_path.read_text(encoding="utf-8")
    word_count = len(full_text.split())
    cur.execute("""
        INSERT INTO transcripts (video_id, full_text, word_count, json_path, txt_path)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (video_id) DO UPDATE SET
            full_text = EXCLUDED.full_text,
            word_count = EXCLUDED.word_count,
            json_path = EXCLUDED.json_path,
            txt_path = EXCLUDED.txt_path
    """, (
        youtube_id,
        full_text,
        word_count,
        str(json_path) if json_path.exists() else None,
        str(txt_path),
    ))
    return True


def load_chunks(cur, youtube_id):
    """Load chunk text + embeddings into the chunks table.

    Chunk JSON format (from generate_chunks.py):
      {chunk_index, chunk_id, youtube_id, episode_title, episode_date,
       start_time, end_time, text, token_count}
    """
    chunks_file = EMBEDDINGS_DIR / f"{youtube_id}_chunks.json"
    npy_file = EMBEDDINGS_DIR / f"{youtube_id}.npy"

    with open(chunks_file) as f:
        chunks = json.load(f)
    embeddings = np.load(npy_file)

    if len(chunks) != len(embeddings):
        print(f"  Error: chunk count ({len(chunks)}) != embedding count ({len(embeddings)})")
        return 0

    if embeddings.shape[1] != EMBEDDING_DIM:
        print(f"  Error: embedding dim {embeddings.shape[1]} != expected {EMBEDDING_DIM}")
        return 0

    # Delete existing chunks for this video (clean upsert)
    cur.execute("DELETE FROM chunks WHERE video_id = %s", (youtube_id,))

    rows = []
    for i, chunk in enumerate(chunks):
        vec_str = "[" + ",".join(f"{v:.8f}" for v in embeddings[i]) + "]"
        # Support both dict (generate_chunks output) and plain string formats
        if isinstance(chunk, str):
            text = chunk
            token_count = None
            start_time = None
            end_time = None
            speaker = None
            speaker_confidence = None
        else:
            text = chunk.get("text", "")
            token_count = chunk.get("token_count")
            start_time = chunk.get("start_time")
            end_time = chunk.get("end_time")
            speaker = chunk.get("speaker")
            speaker_confidence = chunk.get("speaker_confidence")
        rows.append((youtube_id, i, text, token_count, start_time, end_time, speaker, speaker_confidence, vec_str))

    execute_values(
        cur,
        """INSERT INTO chunks (video_id, chunk_index, text, token_count, start_time, end_time, speaker, speaker_confidence, embedding)
           VALUES %s""",
        rows,
        template="(%s, %s, %s, %s, %s, %s, %s, %s, %s)",
    )
    return len(rows)


# ---------------------------------------------------------------------------
# Episode processing
# ---------------------------------------------------------------------------

def process_episode(conn, episodes, youtube_id, force=False):
    ep = find_episode(episodes, youtube_id)
    if ep is None:
        print(f"  Warning: {youtube_id} not found in episodes.json, using minimal metadata")
        ep = {"youtube_id": youtube_id, "titulo": None, "descripcion": None, "fecha": None}

    with conn.cursor() as cur:
        if not force and already_loaded(cur, youtube_id):
            print(f"  Skipping {youtube_id}: already loaded")
            return {"videos": 0, "transcripts": 0, "chunks": 0}

        upsert_video(cur, ep)
        t_ok = upsert_transcript(cur, youtube_id)
        n = load_chunks(cur, youtube_id)
        conn.commit()
        title = ep.get("titulo") or youtube_id
        print(f"  Loaded {youtube_id} ({title}): {n} chunks")
        return {"videos": 1, "transcripts": 1 if t_ok else 0, "chunks": n}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Load episodes into PLM database")
    parser.add_argument("youtube_id", nargs="?", help="YouTube video ID to load")
    parser.add_argument("--all", action="store_true", help="Load all available episodes")
    parser.add_argument("--force", action="store_true", help="Reload even if already loaded")
    args = parser.parse_args()

    if not args.youtube_id and not args.all:
        parser.error("Provide a youtube_id or --all")

    episodes = load_episodes()
    conn = get_connection()

    totals = {"videos": 0, "transcripts": 0, "chunks": 0}

    try:
        ensure_schema(conn)

        if args.all:
            ids = get_available_ids()
            if not ids:
                print("No embeddings found in", EMBEDDINGS_DIR)
                return
            print(f"Processing {len(ids)} episodes...")
            for yt_id in ids:
                result = process_episode(conn, episodes, yt_id, force=args.force)
                for k in totals:
                    totals[k] += result[k]
        else:
            chunks_file = EMBEDDINGS_DIR / f"{args.youtube_id}_chunks.json"
            npy_file = EMBEDDINGS_DIR / f"{args.youtube_id}.npy"
            if not chunks_file.exists() or not npy_file.exists():
                print(f"Error: embeddings not found for {args.youtube_id}")
                print(f"  Expected: {chunks_file}")
                print(f"  Expected: {npy_file}")
                sys.exit(1)
            result = process_episode(conn, episodes, args.youtube_id, force=args.force)
            for k in totals:
                totals[k] += result[k]

        print(f"\nSummary: {totals['videos']} videos, "
              f"{totals['transcripts']} transcripts, "
              f"{totals['chunks']} chunks loaded.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
