# CLAUDE.md — Paren La Mano

## Project Overview

Web archive and semantic search platform for the "Paren La Mano" YouTube show (Argentine Spanish podcast/stream on Vorterix).

## Current State

- **Static frontend**: `index.html`, `episodes.html`, `historia.html`, `chat.html`, `ranking.html`
- **Chat API**: `api/server.py` — FastAPI + sentence-transformers semantic search over episode descriptions + Letterboxd-style ratings (SQLite)
- **Episode catalog**: `data/episodes.json` — 28 episodes with youtube_id, title, date, duration, description

## Active Work: YouTube Transcription Ingestion

See [PRD.md](PRD.md) for architecture and [TASKS.md](TASKS.md) for task backlog.

Goal: Feed PostgreSQL+pgvector with full transcriptions of YouTube episodes for semantic search over actual content (not just titles/descriptions).

Stack: faster-whisper (local), sentence-transformers (local), PostgreSQL+pgvector (local on EC2). Zero cloud cost.

## Key Commands

```bash
# Run chat API
cd api && uvicorn server:app --host 0.0.0.0 --port 8889

# Run static site
python3 -m http.server 8000

# Pipeline scripts (in scripts/)
python scripts/download_youtube.py --all        # Download audio
python scripts/transcribe_local.py --all        # Transcribe with whisper
python scripts/generate_chunks.py --all         # Chunk + embed
python scripts/load_to_db.py --all              # Load to PostgreSQL
python scripts/ingest_all.py                    # End-to-end pipeline
```

## Infrastructure

- **EC2**: 2 vCPU, 3.8GB RAM, 119GB disk free
- **PostgreSQL**: localhost:5432, db=plm_db, user=plm_user
- **S3 buckets**: plm-transcripts-390402545272, plm-media-processing-390402545272, plm-datasets-390402545272
- **Git remote**: https://github.com/greydina/paren-la-mano.git

## Embedding Model

Using `paraphrase-multilingual-MiniLM-L12-v2` (384 dims) — same model in api/server.py. Good for Spanish content.

## Speaker Profiles (Phase 2)

Voice embeddings for 4 speakers (Roberto, Luquitas, Joaquin, German) stored in S3. pyannote-audio installed for diarization.

## Language

Content is in Argentine Spanish. Queries, episode data, and UI text are all in Spanish.
