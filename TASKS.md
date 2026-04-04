# Paren La Mano — Task Backlog

> See [PRD.md](PRD.md) for full context and architecture decisions.

## Phase 1: Transcription + Vector DB Ingestion

### Task 1: YouTube audio download script
**Status**: TODO
**Priority**: P0
**Blocked by**: —

Create `scripts/download_youtube.py`:
- Takes a youtube_id (or `--all` to process episodes.json)
- Downloads audio-only (m4a/mp4) via yt-dlp to `data/audio/`
- Skips already-downloaded files
- For the 5 episodes already in S3 as full MP4 (`plm-media-processing-390402545272/videos/`), extract audio with ffmpeg instead of re-downloading
- Rate limiting between downloads
- Log progress to stdout

**S3 episodes already available** (youtube IDs extracted from filenames):
- `50UtSvG80yA`, `SecttWanihg`, `-UIAVpa9KgY`, `g-BN4_9ORSs`, `2N4v89apuUY`

**Acceptance Criteria**:
- `python scripts/download_youtube.py --all` downloads audio for all 28 episodes
- Skips already-downloaded
- Audio files in `data/audio/<youtube_id>.m4a`

---

### Task 2: Local transcription script (faster-whisper)
**Status**: TODO
**Priority**: P0
**Blocked by**: Task 1

Create `scripts/transcribe_local.py`:
- Uses faster-whisper with `medium` model (fallback to `small` if OOM)
- Language: Spanish (`es`)
- Input: audio file from `data/audio/`
- Output: `data/transcripts/<youtube_id>.json` (with word-level timestamps) and `data/transcripts/<youtube_id>.txt` (plain text)
- Supports `--all` to batch-process all audio files
- Skips already-transcribed
- Reports progress (episode X of N, estimated time remaining)

**Note**: EC2 has 3.8GB RAM, 2 vCPU. Process one episode at a time. Expect ~1x realtime with `small`, ~0.5x with `medium`.

**Acceptance Criteria**:
- `python scripts/transcribe_local.py --all` transcribes all audio files
- JSON output includes word-level timestamps
- Plain text output is clean and readable

---

### Task 3: Chunking + embedding generation script
**Status**: TODO
**Priority**: P0
**Blocked by**: Task 2

Create `scripts/generate_chunks.py`:
- Reads transcripts from `data/transcripts/<youtube_id>.txt`
- Chunks by ~500 tokens with 50-token overlap, respecting sentence boundaries
- Prepends episode metadata (title + date) to each chunk for better retrieval
- Generates embeddings using `paraphrase-multilingual-MiniLM-L12-v2` (384 dims, already used in server.py)
- Saves to `data/embeddings/<youtube_id>_chunks.json` and `data/embeddings/<youtube_id>.npy`
- Supports `--all`

**Decision**: Use the same model as `api/server.py` (paraphrase-multilingual-MiniLM-L12-v2) for consistency. This means 384-dim embeddings, not the 1024-dim Bedrock Titan in the PLM schema. Either:
  - (a) Update the pgvector chunks table to 384 dims, or
  - (b) Create a new table / separate DB for this project

**Acceptance Criteria**:
- Chunks are ~500 tokens with metadata prefix
- Embeddings generated locally, no API calls
- Output files saved per episode

---

### Task 4: Database loading script
**Status**: TODO
**Priority**: P0
**Blocked by**: Task 3

Create `scripts/load_to_db.py`:
- Loads episode metadata, transcripts, and chunk embeddings into PostgreSQL
- Handles the dimension mismatch (384 vs 1024) — either alter schema or create new table
- Upserts to avoid duplicates
- Supports `--all` and individual youtube_id
- Reports what was loaded

**Decision needed**: Use existing PLM database (plm_db) or create a new `plm_yt` database? Using existing is simpler but the old TikTok data uses 1024-dim embeddings.

**Acceptance Criteria**:
- All 28 episodes loaded with metadata, full transcript, and chunked embeddings
- Semantic search via pgvector works

---

### Task 5: Batch ingestion orchestrator
**Status**: TODO
**Priority**: P1
**Blocked by**: Tasks 1-4

Create `scripts/ingest_all.py`:
- End-to-end pipeline: download → transcribe → chunk → embed → load
- Processes all 28 episodes
- Resume support (skips completed steps per episode)
- Progress reporting and cost tracking (even though $0, track time)
- `--dry-run` flag to preview what would be done

**Acceptance Criteria**:
- Single command to ingest everything
- Resumable after interruption
- Clear progress output

---

### Task 6: Update chat API to query pgvector
**Status**: TODO
**Priority**: P0
**Blocked by**: Task 4

Update `api/server.py`:
- Add a PostgreSQL query path alongside the existing in-memory search
- Search over transcript chunks (not just episode descriptions)
- Return matching chunks with episode context and timestamps
- Keep the existing in-memory fallback for when DB is unavailable

**Acceptance Criteria**:
- Chat returns answers based on actual transcript content
- "cuando hablaron del mate?" returns specific episode + relevant text
- Existing episode-description search still works as fallback

---

### Task 7: Sync episodes.json with full episode catalog
**Status**: TODO
**Priority**: P1
**Blocked by**: —

The 5 S3 episodes (full programs from Vorterix) are NOT in the current episodes.json (which has 28 clips). Need to:
- Determine if these 5 are separate full episodes or contain some of the 28 clips
- Add them to episodes.json if they're distinct
- Or map clip youtube_ids to their parent full episodes

**Acceptance Criteria**:
- Complete episode catalog in episodes.json
- Clear mapping between clips and full episodes (if applicable)

---

## Phase 2: Speaker Identification

### Task 8: Speaker diarization pipeline
**Status**: TODO (Phase 2)
**Priority**: P1
**Blocked by**: Task 2

Create `scripts/diarize.py`:
- Uses pyannote-audio for speaker diarization
- Input: audio from `data/audio/`
- Output: `data/diarization/<youtube_id>.json` with speaker segments (start, end, speaker_label)
- Process sequentially (RAM constraint)

**Acceptance Criteria**:
- Each episode has diarization output with speaker segments
- Segments include start/end timestamps

---

### Task 9: Speaker identification using voice profiles
**Status**: TODO (Phase 2)
**Priority**: P1
**Blocked by**: Task 8

Create `scripts/identify_speakers.py`:
- Loads voice profiles from S3 (`plm-datasets-390402545272/plm-voices/plm_final_voice_profiles.json`)
- Profiles available: **Roberto, Luquitas, Joaquin, German** (256-dim embeddings)
- Matches diarization segments against profiles using cosine similarity
- Updates diarization output with identified speaker names
- Tags transcript chunks with speaker identity

**Acceptance Criteria**:
- >80% of segments assigned to a known speaker
- Transcript chunks tagged with speaker name
- Enables queries like "qué dijo Roberto sobre X"

---

### Task 10: Speaker-aware search
**Status**: TODO (Phase 2)
**Priority**: P2
**Blocked by**: Task 9

Update chat API and DB schema to support speaker-filtered search:
- Filter by speaker in search queries
- Display speaker names in results
- "qué dijo Luquitas sobre fútbol?" returns only Luquitas segments

---

## Inventory: What's Already Done

| Asset | Location | Status |
|-------|----------|--------|
| 28 episode metadata | `data/episodes.json` | Ready |
| 5 full episode MP4s | `s3://plm-media-processing-390402545272/videos/` | Ready |
| 3 TikTok transcripts | `s3://plm-transcripts-390402545272/transcripts/` | Legacy |
| 4 voice profiles | `s3://plm-datasets-390402545272/plm-voices/` | Ready |
| PostgreSQL + pgvector | localhost:5432 / plm_db | Running |
| Chat API (descriptions only) | `api/server.py` | Running |
| Diarization experiments | `s3://plm-datasets-390402545272/experiments/` | Research |
