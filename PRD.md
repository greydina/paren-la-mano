# Paren La Mano — YouTube Transcription Ingestion PRD

## Vision

Alimentar la base de datos vectorial del proyecto Paren La Mano con transcripciones completas de los episodios de YouTube, habilitando búsqueda semántica sobre el contenido real de cada programa (no solo títulos y descripciones).

## Context

### What Exists Today

- **28 episodios catalogados** en `data/episodes.json` con título, fecha, youtube_id, duración y descripción corta
- **API de chat** (`api/server.py`) con búsqueda semántica in-memory usando sentence-transformers, pero solo sobre títulos + descripciones (no transcripciones)
- **PostgreSQL + pgvector** corriendo en el EC2 local con esquema PLM (tablas: videos, transcripts, chunks con embeddings de 1024 dims)
- **5 episodios completos ya descargados** en S3 (`plm-media-processing-390402545272/videos/`):
  - `50UtSvG80yA` — ARRANCÓ PAREN LA MANO 2025 - DESDE LOS ÁNGELES
  - `SecttWanihg` — AYUDAMOS EN EL AMOR A NUESTROS OYENTES
  - `-UIAVpa9KgY` — EL TRIBUNERO CON FLAVIO AZZARO
  - `g-BN4_9ORSs` — LA VENGANZA DE MARIO PERGOLINI
  - `2N4v89apuUY` — PLM CON P DE PRIME
- **3 clips TikTok** ya transcritos y cargados en la DB (IDs: 7591261383543508232, 7605204496876080401, 7616912091609566484)
- **Voice profiles** para 4 speakers (Roberto, Luquitas, Joaquin, German) con embeddings de 256 dims en S3 (`plm-datasets-390402545272/plm-voices/`)
- **Herramientas instaladas en EC2**: yt-dlp, whisper, faster-whisper, ffmpeg, pyannote-audio, sentence-transformers, sqlite-vec, psycopg2

### Infrastructure (EC2)

- **Instance**: 2 vCPU, 3.8GB RAM, 119GB disk free
- **PostgreSQL**: local, running, pgvector extension enabled
- **Python**: whisper (openai-whisper + faster-whisper), pyannote-audio, sentence-transformers

## Architecture

```
episodes.json (28 episodios)
        |
        v
[Phase 1] yt-dlp download audio → data/audio/<youtube_id>.m4a
        |
        v
[Phase 1] faster-whisper transcribe → data/transcripts/<youtube_id>.json + .txt
        |
        v
[Phase 1] Chunk + embed (sentence-transformers) → PostgreSQL pgvector
        |
        v
[Phase 1] Update chat API to query pgvector instead of in-memory
        |
        v
[Phase 2] pyannote-audio diarization + voice profile matching → speaker-tagged segments
```

## Cost Strategy

| Component | Option | Cost |
|-----------|--------|------|
| Transcription | **faster-whisper local** (medium model, Spanish) | $0 |
| Embeddings | **sentence-transformers local** (paraphrase-multilingual-MiniLM-L12-v2) | $0 |
| Vector DB | **PostgreSQL + pgvector local** (already running on EC2) | $0 |
| Audio download | **yt-dlp** (audio only) | $0 |
| Storage | Local disk (119GB free, ~2GB needed for audio) | $0 |
| **Total** | | **$0** |

No AWS Transcribe, no Bedrock Titan, no RDS. Everything runs on the existing EC2.

### Fallback

If EC2 RAM is too tight for whisper medium model + pyannote simultaneously, use whisper `small` model (still good for Spanish) or process sequentially.

## Scope

### Phase 1: Transcription + Vector DB (MVP)

1. Download audio from all 28 YouTube episodes (audio-only via yt-dlp)
2. Transcribe using faster-whisper locally
3. Chunk transcripts and generate embeddings with sentence-transformers
4. Load into PostgreSQL + pgvector (reuse existing PLM schema)
5. Update chat API to search over real transcriptions
6. Skip episodes already in S3 (5 full episodes) — extract audio from those

### Phase 2: Speaker Identification

1. Run pyannote-audio diarization on each episode
2. Match speaker segments against existing voice profiles (Roberto, Luquitas, Joaquin, German)
3. Tag transcript chunks with speaker identity
4. Enable queries like "qué dijo Roberto sobre..." or "cuando habló Luquitas de..."

## Success Criteria

- All 28 episodes transcribed and searchable
- Semantic search returns relevant results for Spanish natural language queries
- Chat API uses full transcript content (not just episode descriptions)
- Phase 2: Speaker labels on >80% of transcript segments

## Constraints

- EC2 has only 3.8GB RAM — process episodes sequentially, not in parallel
- 2 vCPU — transcription will be slow (~1x realtime with whisper small, ~0.5x with medium)
- Total audio ~10.5 hours — expect ~10-20 hours transcription time for medium model
- Keep everything in `paren-la-mano` repo, commit progress incrementally
