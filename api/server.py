"""
Paren la Mano — RAG Chat API
Semantic search over episode data using sentence-transformers.
pgvector search over transcript chunks when PostgreSQL is available.
Letterboxd-style ratings system for episodes.
Run: uvicorn server:app --host 0.0.0.0 --port 8889
"""

import json
import os
import random
import re
import sqlite3
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, CrossEncoder

try:
    import psycopg2
    import psycopg2.extras
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EPISODES_PATH = Path(__file__).resolve().parent.parent / "data" / "episodes.json"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
RERANKER_MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
TOP_K = 5
RERANK_CANDIDATES = 20
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://plm_user:plm_pass_2026@localhost:5432/plm_db",
)
KNOWN_SPEAKERS = ["Roberto", "Luquitas", "Joaquin", "German"]
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TAGGING_DIR = DATA_DIR / "tagging"
TAGS_FILE = TAGGING_DIR / "tags.json"
DIARIZATION_DIR = DATA_DIR / "diarization"
AUDIO_DIR = DATA_DIR / "audio"
TAGGING_SPEAKERS = [
    "Luquitas", "Roberto", "Joaquin", "German", "Alfredo", "Jazmin",
    "Otra persona", "Voces mezcladas", "Ruido/Musica", "Audio grabado",
]
# Patterns to detect speaker intent in Spanish queries (case-insensitive)
_SPEAKER_PATTERNS = [
    r"(?:qu[eé]\s+dijo|dice|decia|comento|hablo|habl[oó]|opina|opino|opin[oó]|menciono|mencion[oó]|cont[oó]|conto)\s+(\w+)",
    r"(?:seg[uú]n|para)\s+(\w+)",
    r"(?:cuando|donde)\s+(\w+)\s+(?:dijo|dice|hablo|habl[oó]|comento|coment[oó]|menciono|mencion[oó])",
]
RATINGS_DB_PATH = os.environ.get(
    "RATINGS_DB_PATH",
    str(Path(__file__).resolve().parent / "ratings.db"),
)

# LLM generation config (OpenRouter)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "anthropic/claude-3.5-haiku"

llm_client: "OpenAI | None" = None
if HAS_OPENAI:
    try:
        llm_client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
        )
        print(f"LLM client initialized (model: {LLM_MODEL})")
    except Exception as exc:
        print(f"WARNING: Failed to initialize LLM client: {exc}")
        llm_client = None
else:
    print("WARNING: openai package not installed. LLM generation disabled.")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Paren la Mano Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Load data & model at startup
# ---------------------------------------------------------------------------
episodes: list[dict] = []
episode_texts: list[str] = []
episode_embeddings: np.ndarray | None = None
model: SentenceTransformer | None = None
reranker: CrossEncoder | None = None
pg_available: bool = False


def _get_pg():
    """Return a new PostgreSQL connection or None on failure."""
    if not HAS_PSYCOPG2:
        return None
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except Exception as exc:
        print(f"PostgreSQL connection failed: {exc}")
        return None


@app.on_event("startup")
def load_data():
    global episodes, episode_texts, episode_embeddings, model, reranker, pg_available

    # Load episodes
    with open(EPISODES_PATH, "r", encoding="utf-8") as f:
        episodes = json.load(f)

    # Build text corpus: combine titulo + descripcion for richer embeddings
    episode_texts = [
        f"{ep['titulo']}. {ep['descripcion']}" for ep in episodes
    ]

    # Load multilingual model (good for Spanish)
    print(f"Loading model {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded.")

    # Load cross-encoder reranker (graceful degradation if it fails)
    try:
        print(f"Loading reranker {RERANKER_MODEL_NAME} ...")
        reranker = CrossEncoder(RERANKER_MODEL_NAME)
        print("Reranker loaded.")
    except Exception as exc:
        print(f"WARNING: Failed to load reranker: {exc}. Reranking disabled.")
        reranker = None

    # Pre-compute episode embeddings
    episode_embeddings = model.encode(episode_texts, convert_to_numpy=True)
    # Normalise for cosine similarity via dot product
    norms = np.linalg.norm(episode_embeddings, axis=1, keepdims=True)
    episode_embeddings = episode_embeddings / norms
    print(f"Indexed {len(episodes)} episodes.")

    # Check PostgreSQL + pgvector availability
    conn = _get_pg()
    if conn:
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL"
            )
            chunk_count = cur.fetchone()[0]
            cur.close()
            conn.close()
            if chunk_count > 0:
                pg_available = True
                print(f"pgvector search enabled — {chunk_count} embedded chunks available.")
            else:
                print("PostgreSQL connected but no embedded chunks found. Using in-memory search.")
        except Exception as exc:
            print(f"pgvector check failed: {exc}. Using in-memory search.")
            conn.close()
    else:
        print("PostgreSQL unavailable. Using in-memory episode search.")


# ---------------------------------------------------------------------------
# Ratings DB helpers
# ---------------------------------------------------------------------------
def _get_db() -> sqlite3.Connection:
    """Return a new connection to the ratings SQLite database."""
    conn = sqlite3.connect(RATINGS_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _init_ratings_db() -> None:
    """Create table + index if they don't exist, seed with sample data."""
    conn = _get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ratings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            season INTEGER NOT NULL,
            episode INTEGER NOT NULL,
            score REAL NOT NULL CHECK(score >= 1 AND score <= 10),
            comment TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_ratings_season_episode
        ON ratings(season, episode)
    """)
    conn.commit()

    # Seed with sample data if table is empty
    row = conn.execute("SELECT COUNT(*) AS cnt FROM ratings").fetchone()
    if row["cnt"] == 0:
        _seed_ratings(conn)
    conn.close()


def _seed_ratings(conn: sqlite3.Connection) -> None:
    """Insert ~50 sample ratings across seasons 1-5, episodes 1-10."""
    sample_comments = [
        "Muy buen programa",
        "Increible episodio, me rei mucho",
        "Flojo, esperaba mas",
        "De lo mejor de la temporada",
        "Buenisimo el invitado",
        "Me aburrio un poco",
        "Genial como siempre",
        "El mejor programa del ano",
        "No me gusto tanto",
        "Imperdible, lo recomiendo",
        "Muy divertido, grandes anecdotas",
        "Regular, nada especial",
        None,
        None,
        None,
    ]
    random.seed(42)
    rows = []
    for _ in range(50):
        season = random.randint(1, 5)
        episode = random.randint(1, 10)
        score = round(random.uniform(5.0, 10.0), 1)
        comment = random.choice(sample_comments)
        rows.append((season, episode, score, comment))
    conn.executemany(
        "INSERT INTO ratings (season, episode, score, comment) VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    print(f"Seeded {len(rows)} sample ratings.")


@app.on_event("startup")
def init_ratings():
    _init_ratings_db()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    query: str
    speaker: str | None = None


class EpisodeResult(BaseModel):
    id: int
    titulo: str
    fecha: str
    descripcion: str
    youtube_id: str
    duracion: str | None = None
    numero: int | None = None
    score: float


class ChunkResult(BaseModel):
    text: str
    episode_title: str
    episode_date: str | None = None
    youtube_id: str
    start_time: float | None = None
    end_time: float | None = None
    speaker: str | None = None
    score: float
    rerank_score: float | None = None
    youtube_url: str | None = None


class ChatResponse(BaseModel):
    answer: str
    episodes: list[EpisodeResult]
    chunks: list[ChunkResult] = []
    source: str = "episodes"  # "episodes" or "chunks"
    generated: bool = False  # True when answer was generated by LLM


# -- Ratings models --

class RatingRequest(BaseModel):
    season: int = Field(..., ge=1)
    episode: int = Field(..., ge=1)
    score: float = Field(..., ge=1, le=10)
    comment: str | None = None


class RatingSubmitResponse(BaseModel):
    success: bool
    new_average: float


class RatingDetail(BaseModel):
    score: float
    comment: str | None
    created_at: str


class EpisodeRatingsResponse(BaseModel):
    season: int
    episode: int
    average: float
    count: int
    ratings: list[RatingDetail]


class GridResponse(BaseModel):
    grid: dict[str, dict[str, float]]
    season_averages: dict[str, float]
    total_ratings: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def semantic_search(query: str, top_k: int = TOP_K) -> list[tuple[dict, float]]:
    """Return top-k episodes sorted by cosine similarity to the query."""
    query_emb = model.encode([query], convert_to_numpy=True)
    query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
    scores = (episode_embeddings @ query_emb.T).squeeze()
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score > 0.1:  # minimum relevance threshold
            results.append((episodes[idx], score))
    return results


def detect_speaker_from_query(query: str) -> str | None:
    """Auto-detect a speaker name from the query text (Spanish patterns).

    Examples:
        "que dijo Luquitas sobre futbol" -> "Luquitas"
        "segun Roberto el partido" -> "Roberto"
    """
    # Build a lookup: lowered name -> canonical name
    name_map = {s.lower(): s for s in KNOWN_SPEAKERS}
    query_lower = query.lower()

    # Try regex patterns first
    for pattern in _SPEAKER_PATTERNS:
        m = re.search(pattern, query_lower)
        if m:
            candidate = m.group(1).lower()
            if candidate in name_map:
                return name_map[candidate]

    # Fallback: check if any known speaker name appears anywhere in the query
    for lower_name, canonical in name_map.items():
        if lower_name in query_lower:
            return canonical

    return None


def rerank(query: str, chunks: list[ChunkResult], top_k: int = TOP_K) -> list[ChunkResult]:
    """Re-rank chunks using the cross-encoder reranker. Falls back to original order if reranker is unavailable."""
    if reranker is None or not chunks:
        return chunks[:top_k]
    try:
        pairs = [(query, chunk.text) for chunk in chunks]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(chunks, scores), key=lambda x: float(x[1]), reverse=True)
        results = []
        for chunk, score in ranked[:top_k]:
            chunk.rerank_score = round(float(score), 4)
            results.append(chunk)
        return results
    except Exception as exc:
        print(f"Reranking failed: {exc}. Returning original order.")
        return chunks[:top_k]


def pgvector_search(query: str, top_k: int = TOP_K, speaker: str | None = None) -> list[ChunkResult] | None:
    """Search transcript chunks via pgvector. Returns None if DB unavailable.

    Retrieves extra candidates for cross-encoder reranking when the reranker is
    available, then returns the top_k results after reranking.
    """
    conn = _get_pg()
    if not conn:
        return None
    try:
        # Fetch more candidates when reranker is available so it can re-sort
        fetch_limit = RERANK_CANDIDATES if reranker is not None else top_k

        query_emb = model.encode([query], convert_to_numpy=True)
        query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
        emb_list = query_emb[0].tolist()
        emb_str = "[" + ",".join(str(x) for x in emb_list) + "]"

        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        if speaker:
            cur.execute(
                """
                SELECT
                    c.text,
                    c.start_time,
                    c.end_time,
                    c.speaker,
                    c.embedding <=> %s::vector AS distance,
                    v.title,
                    v.upload_date,
                    v.video_id
                FROM chunks c
                JOIN videos v ON c.video_id = v.video_id
                WHERE c.embedding IS NOT NULL AND c.speaker = %s
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
                """,
                (emb_str, speaker, emb_str, fetch_limit),
            )
        else:
            cur.execute(
                """
                SELECT
                    c.text,
                    c.start_time,
                    c.end_time,
                    c.speaker,
                    c.embedding <=> %s::vector AS distance,
                    v.title,
                    v.upload_date,
                    v.video_id
                FROM chunks c
                JOIN videos v ON c.video_id = v.video_id
                WHERE c.embedding IS NOT NULL
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
                """,
                (emb_str, emb_str, fetch_limit),
            )
        rows = cur.fetchall()
        cur.close()
        conn.close()

        candidates = []
        for row in rows:
            similarity = 1.0 - float(row["distance"])
            if similarity < 0.1:
                continue
            clean_text = _strip_episode_prefix(row["text"])
            video_id = row["video_id"]
            start = row["start_time"]
            candidates.append(ChunkResult(
                text=clean_text,
                episode_title=row["title"] or "",
                episode_date=str(row["upload_date"]) if row["upload_date"] else None,
                youtube_id=video_id,
                start_time=start,
                end_time=row["end_time"],
                speaker=row.get("speaker"),
                score=round(similarity, 4),
                youtube_url=_youtube_timestamp_url(video_id, start),
            ))

        # Apply cross-encoder reranking
        return rerank(query, candidates, top_k=top_k)
    except Exception as exc:
        print(f"pgvector search error: {exc}")
        try:
            conn.close()
        except Exception:
            pass
        return None


def _strip_episode_prefix(text: str) -> str:
    """Strip 'Episodio: TITLE (DATE)\\n\\n' metadata prefix from chunk text."""
    if text.startswith("Episodio: "):
        idx = text.find("\n\n")
        if idx != -1:
            return text[idx + 2:]
    return text


def _youtube_timestamp_url(youtube_id: str, start_time: float | None) -> str:
    """Build a YouTube URL with optional timestamp."""
    url = f"https://www.youtube.com/watch?v={youtube_id}"
    if start_time is not None:
        url += f"&t={int(start_time)}"
    return url


def _format_timestamp(seconds: float | None) -> str:
    """Convert seconds to HH:MM:SS or MM:SS string."""
    if seconds is None:
        return ""
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def format_answer(query: str, results: list[tuple[dict, float]]) -> str:
    """Build a friendly Spanish answer referencing matching episodes."""
    if not results:
        return (
            "No encontre programas que coincidan con tu consulta. "
            "Proba con otras palabras clave o preguntame sobre futbol, "
            "entrevistas, anecdotas o cualquier tema del programa."
        )

    if len(results) == 1:
        ep = results[0][0]
        return (
            f"Encontre un programa que puede interesarte: "
            f"\"{ep['titulo']}\" ({ep['fecha']}). "
            f"{ep['descripcion']}"
        )

    intro = f"Encontre {len(results)} programas relacionados con tu consulta:\n\n"
    lines = []
    for ep, score in results:
        lines.append(
            f"- \"{ep['titulo']}\" ({ep['fecha']})"
        )
    return intro + "\n".join(lines)


def generate_answer(query: str, chunks: list[ChunkResult]) -> str | None:
    """Use LLM to generate a natural language answer from retrieved chunks.

    Returns None if the LLM call fails for any reason (fallback to format_answer_chunks).
    """
    if llm_client is None or not chunks:
        return None

    # Build context from top 5 chunks
    top_chunks = chunks[:5]
    context_parts = []
    for i, chunk in enumerate(top_chunks, 1):
        ts = _format_timestamp(chunk.start_time)
        ts_tag = f", {ts}" if ts else ""
        speaker_tag = f" [{chunk.speaker}]" if chunk.speaker else ""
        yt_url = chunk.youtube_url or _youtube_timestamp_url(chunk.youtube_id, chunk.start_time)
        context_parts.append(
            f"[{i}] ({chunk.episode_title}{ts_tag}){speaker_tag} {chunk.text}\n    Link: {yt_url}"
        )
    context_block = "\n\n".join(context_parts)

    user_message = (
        f"Pregunta: {query}\n\n"
        f"Fragmentos del programa:\n\n"
        f"{context_block}\n\n"
        f"Responde de forma concisa citando los fragmentos relevantes con su numero [N] "
        f"e incluyendo el link de YouTube cuando sea util."
    )

    try:
        response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Sos un asistente experto del programa Paren La Mano, "
                        "un podcast/stream argentino de Vorterix. "
                        "Responde en espanol argentino informal. "
                        "Basa tu respuesta SOLO en los fragmentos provistos. "
                        "Si no hay informacion suficiente, decilo. "
                        "Cita los momentos del video con timestamps."
                    ),
                },
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
            max_tokens=500,
        )
        return response.choices[0].message.content
    except Exception as exc:
        print(f"LLM generation failed: {exc}")
        return None


def format_answer_chunks(query: str, chunks: list[ChunkResult]) -> str:
    """Build a Spanish answer referencing specific transcript moments."""
    if not chunks:
        return (
            "No encontre fragmentos que coincidan con tu consulta. "
            "Proba con otras palabras clave o preguntame sobre futbol, "
            "entrevistas, anecdotas o cualquier tema del programa."
        )

    # Group chunks by episode
    seen_episodes: dict[str, list[ChunkResult]] = {}
    for chunk in chunks:
        seen_episodes.setdefault(chunk.youtube_id, []).append(chunk)

    n_episodes = len(seen_episodes)
    n_chunks = len(chunks)
    intro = (
        f"Encontre {n_chunks} fragmento{'s' if n_chunks != 1 else ''} "
        f"en {n_episodes} programa{'s' if n_episodes != 1 else ''} "
        f"relacionados con tu consulta:\n\n"
    )

    lines = []
    for vid_id, ep_chunks in seen_episodes.items():
        title = ep_chunks[0].episode_title
        date = ep_chunks[0].episode_date or ""
        # Clean title: strip "Episodio: " prefix if present
        clean_title = title
        if clean_title.lower().startswith("episodio: "):
            clean_title = clean_title[10:]
        lines.append(f"\U0001f399\ufe0f **Programa del {date}** — {clean_title}")
        lines.append("")
        for chunk in ep_chunks:
            ts = _format_timestamp(chunk.start_time)
            snippet = chunk.text[:150].strip()
            if len(chunk.text) > 150:
                snippet += "..."
            speaker_tag = chunk.speaker or "?"
            ts_tag = f", {ts}" if ts else ""
            lines.append(f"[{speaker_tag}{ts_tag}] \"{snippet}\"")
            yt_url = chunk.youtube_url or _youtube_timestamp_url(vid_id, chunk.start_time)
            lines.append(f"\U0001f517 Ver en YouTube: {yt_url}")
            lines.append("")
        lines.append("")

    return intro + "\n".join(lines).rstrip()


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------
@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    query = req.query.strip()
    if not query:
        return ChatResponse(
            answer="Escribi tu pregunta y te busco los programas mas relevantes.",
            episodes=[],
        )

    # Resolve speaker: explicit param takes priority, then auto-detect from query
    speaker = req.speaker
    if not speaker:
        speaker = detect_speaker_from_query(query)

    # Prefer pgvector chunk search when available
    if pg_available:
        chunk_results = pgvector_search(query, speaker=speaker)

        # Fallback: if speaker-filtered search returned empty, retry without speaker
        speaker_fallback = False
        if speaker and (chunk_results is None or len(chunk_results) == 0):
            chunk_results = pgvector_search(query)
            speaker_fallback = True

        if chunk_results is not None and len(chunk_results) > 0:
            # Try LLM generation first
            generated = False
            llm_answer = generate_answer(query, chunk_results)
            if llm_answer:
                answer = llm_answer
                generated = True
                if speaker_fallback:
                    answer = (
                        f"No encontre fragmentos especificos de {speaker}, "
                        f"pero esto es lo que encontre:\n\n{llm_answer}"
                    )
            else:
                # Fallback to formatted chunks
                if speaker_fallback:
                    answer = (
                        f"No encontre fragmentos especificos de {speaker}, "
                        f"pero estos resultados pueden ser relevantes:\n\n"
                        + format_answer_chunks(query, chunk_results)
                    )
                else:
                    answer = format_answer_chunks(query, chunk_results)

            # Also run episode search for episode-level context
            ep_search = semantic_search(query)
            ep_results = [
                EpisodeResult(**ep, score=round(score, 4))
                for ep, score in ep_search
            ]
            return ChatResponse(
                answer=answer,
                episodes=ep_results,
                chunks=chunk_results,
                source="chunks",
                generated=generated,
            )

    # Fallback: in-memory episode search
    results = semantic_search(query)
    answer = format_answer(query, results)
    ep_results = [
        EpisodeResult(**ep, score=round(score, 4)) for ep, score in results
    ]
    return ChatResponse(answer=answer, episodes=ep_results, source="episodes")


# ---------------------------------------------------------------------------
# Rating endpoints
# ---------------------------------------------------------------------------
@app.get("/api/ratings/grid", response_model=GridResponse)
def ratings_grid():
    """Return the full grid of average scores per season/episode."""
    conn = _get_db()
    rows = conn.execute(
        "SELECT season, episode, AVG(score) AS avg_score "
        "FROM ratings GROUP BY season, episode"
    ).fetchall()
    total = conn.execute("SELECT COUNT(*) AS cnt FROM ratings").fetchone()["cnt"]
    conn.close()

    grid: dict[str, dict[str, float]] = {}
    season_scores: dict[str, list[float]] = {}
    for r in rows:
        s = str(r["season"])
        e = str(r["episode"])
        avg = round(r["avg_score"], 1)
        grid.setdefault(s, {})[e] = avg
        season_scores.setdefault(s, []).append(avg)

    season_averages = {
        s: round(sum(vals) / len(vals), 1) for s, vals in season_scores.items()
    }

    return GridResponse(
        grid=grid,
        season_averages=season_averages,
        total_ratings=total,
    )


@app.post("/api/ratings", response_model=RatingSubmitResponse)
def submit_rating(req: RatingRequest):
    """Submit a rating for an episode."""
    conn = _get_db()
    conn.execute(
        "INSERT INTO ratings (season, episode, score, comment) VALUES (?, ?, ?, ?)",
        (req.season, req.episode, req.score, req.comment),
    )
    conn.commit()
    row = conn.execute(
        "SELECT AVG(score) AS avg_score FROM ratings WHERE season = ? AND episode = ?",
        (req.season, req.episode),
    ).fetchone()
    conn.close()
    return RatingSubmitResponse(
        success=True,
        new_average=round(row["avg_score"], 1),
    )


@app.get("/api/ratings/{season}/{episode}", response_model=EpisodeRatingsResponse)
def get_episode_ratings(season: int, episode: int):
    """Get all ratings and comments for a specific episode."""
    conn = _get_db()
    rows = conn.execute(
        "SELECT score, comment, created_at FROM ratings "
        "WHERE season = ? AND episode = ? ORDER BY created_at DESC",
        (season, episode),
    ).fetchall()
    conn.close()

    if not rows:
        return EpisodeRatingsResponse(
            season=season,
            episode=episode,
            average=0.0,
            count=0,
            ratings=[],
        )

    ratings = [
        RatingDetail(score=r["score"], comment=r["comment"], created_at=r["created_at"])
        for r in rows
    ]
    avg = round(sum(r["score"] for r in rows) / len(rows), 1)
    return EpisodeRatingsResponse(
        season=season,
        episode=episode,
        average=avg,
        count=len(rows),
        ratings=ratings,
    )


# ---------------------------------------------------------------------------
# Tagging helpers
# ---------------------------------------------------------------------------
def _load_tags() -> list[dict]:
    """Load existing tags from JSON file."""
    if TAGS_FILE.exists():
        with open(TAGS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_tags(tags: list[dict]) -> None:
    """Save tags to JSON file."""
    TAGGING_DIR.mkdir(parents=True, exist_ok=True)
    with open(TAGS_FILE, "w", encoding="utf-8") as f:
        json.dump(tags, f, ensure_ascii=False, indent=2)


def _sanitize_episode_id(episode_id: str) -> str:
    """Reject episode IDs with path traversal characters."""
    if ".." in episode_id or "/" in episode_id or "\\" in episode_id:
        raise HTTPException(status_code=400, detail="Invalid episode ID")
    return episode_id


def _load_diarization(episode_id: str) -> dict | None:
    """Load a diarization JSON for the given episode."""
    episode_id = _sanitize_episode_id(episode_id)
    path = DIARIZATION_DIR / f"{episode_id}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


DATASET_DIR = DATA_DIR / "tagging" / "clips"

# Persistent MLflow run ID for the tagging session
_mlflow_run_id: str | None = None


def _get_mlflow_client():
    """Get MLflow client and ensure experiment exists."""
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        client = mlflow.tracking.MlflowClient("http://localhost:5000")
        exp = mlflow.set_experiment("plm-speaker-tagging")
        return client, exp.experiment_id
    except Exception as exc:
        print(f"MLflow client failed: {exc}")
        return None, None


def _get_or_create_mlflow_run():
    """Get or create a persistent MLflow run for tagging."""
    global _mlflow_run_id
    client, exp_id = _get_mlflow_client()
    if not client:
        return None, None
    try:
        if _mlflow_run_id:
            return client, _mlflow_run_id
        run = client.create_run(exp_id, run_name="tagging-dataset")
        _mlflow_run_id = run.info.run_id
        return client, _mlflow_run_id
    except Exception as exc:
        print(f"MLflow run creation failed: {exc}")
        return None, None


def _extract_and_save_clip(episode_id: str, start: float, end: float, tag: dict) -> str | None:
    """Extract audio clip and save to dataset directory."""
    audio_path = AUDIO_DIR / f"{episode_id}.wav"
    if not audio_path.exists():
        return None

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    # Filename: {episode}_{segindex}_{speaker}.wav
    speaker_safe = tag["tagged_speaker"].replace("/", "-").replace(" ", "_")
    clip_name = f"{episode_id}_seg{tag['segment_index']}_{speaker_safe}.wav"
    clip_path = DATASET_DIR / clip_name

    pad_start = max(0, start - 0.3)
    duration = (end - pad_start) + 0.3

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(audio_path),
             "-ss", str(pad_start), "-t", str(duration),
             "-ar", "16000", "-ac", "1", str(clip_path)],
            capture_output=True, timeout=15,
        )
        return str(clip_path)
    except Exception:
        return None


def _log_to_mlflow(tags: list[dict], new_tag: dict | None = None) -> None:
    """Log tag + clip to MLflow dataset. One persistent run, incremental."""
    try:
        client, run_id = _get_or_create_mlflow_run()
        if not client or not run_id:
            return

        total = len(tags)
        client.log_metric(run_id, "total_tagged", total, step=total)

        # Per-speaker counts
        speaker_counts: dict[str, int] = {}
        agreements = 0
        for t in tags:
            sp = t.get("tagged_speaker", "Otro")
            base = sp.replace(" +ruido", "")
            speaker_counts[base] = speaker_counts.get(base, 0) + 1
            if t.get("tagged_speaker") == t.get("original_speaker"):
                agreements += 1
        for sp, count in speaker_counts.items():
            safe_sp = sp.replace("/", "_").replace(" ", "_")
            client.log_metric(run_id, f"speaker_{safe_sp}_count", count, step=total)

        agreement_rate = agreements / total if total > 0 else 0
        client.log_metric(run_id, "agreement_rate", round(agreement_rate, 4), step=total)

        has_noise = sum(1 for t in tags if "+ruido" in t.get("tagged_speaker", ""))
        client.log_metric(run_id, "with_noise_count", has_noise, step=total)

        # Log the tags JSON
        client.log_artifact(run_id, str(TAGS_FILE), artifact_path="dataset")

        # If a new clip was extracted, log it organized by speaker
        if new_tag and new_tag.get("clip_path"):
            clip = Path(new_tag["clip_path"])
            if clip.exists():
                speaker_dir = new_tag["tagged_speaker"].replace("/", "-").replace(" ", "_")
                client.log_artifact(run_id, str(clip), artifact_path=f"dataset/clips/{speaker_dir}")

    except Exception as exc:
        print(f"MLflow logging failed (non-critical): {exc}")


# ---------------------------------------------------------------------------
# Tagging request models
# ---------------------------------------------------------------------------
class TagRequest(BaseModel):
    episode_id: str
    segment_index: int
    speaker: str
    start: float
    end: float


# ---------------------------------------------------------------------------
# Tagging endpoints
# ---------------------------------------------------------------------------
@app.get("/api/tagging/episodes")
def tagging_episodes():
    """List available episodes with diarization data."""
    results = []
    for f in sorted(DIARIZATION_DIR.glob("*.json")):
        if f.name == "speaker_profiles_ecapa.json":
            continue
        data = json.loads(f.read_text(encoding="utf-8"))
        results.append({
            "episode_id": f.stem,
            "segment_count": len(data.get("segments", [])),
            "duration": data.get("duration"),
            "num_speakers": data.get("num_speakers"),
            "speaker_stats": data.get("speaker_stats", {}),
        })
    return results


@app.get("/api/tagging/segments")
def tagging_segments(
    episode_id: str = Query(...),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
):
    """Return paginated diarization segments for an episode."""
    data = _load_diarization(episode_id)
    if data is None:
        raise HTTPException(404, f"No diarization found for episode {episode_id}")

    segments = data["segments"]
    total = len(segments)
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    page_segments = segments[start_idx:end_idx]

    # Load existing tags to mark which segments are tagged
    tags = _load_tags()
    tagged_map: dict[str, dict[int, str]] = {}
    for t in tags:
        tagged_map.setdefault(t["episode_id"], {})[t["segment_index"]] = t["tagged_speaker"]

    ep_tags = tagged_map.get(episode_id, {})
    enriched = []
    for i, seg in enumerate(page_segments):
        idx = start_idx + i
        enriched.append({
            **seg,
            "index": idx,
            "tagged_speaker": ep_tags.get(idx),
        })

    return {
        "episode_id": episode_id,
        "segments": enriched,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": (total + per_page - 1) // per_page,
    }


@app.get("/api/tagging/audio/{episode_id}")
def get_audio_clip(
    episode_id: str,
    start: float = Query(...),
    end: float = Query(...),
):
    """Serve an audio clip extracted on-the-fly with ffmpeg."""
    episode_id = _sanitize_episode_id(episode_id)
    audio_path = AUDIO_DIR / f"{episode_id}.wav"
    if not audio_path.exists():
        raise HTTPException(404, f"Audio file not found for episode {episode_id}")

    # Add 0.5s padding before/after
    clip_start = max(0, start - 0.5)
    duration = (end - clip_start) + 0.5

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()

    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(audio_path),
                "-ss", str(clip_start),
                "-t", str(duration),
                "-ar", "16000", "-ac", "1",
                tmp_path,
            ],
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            os.unlink(tmp_path)
            raise HTTPException(500, f"ffmpeg failed: {result.stderr.decode()[:200]}")

        # Use background task to clean up temp file after response is sent
        from starlette.background import BackgroundTask

        def cleanup():
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        return FileResponse(
            tmp_path,
            media_type="audio/wav",
            background=BackgroundTask(cleanup),
        )
    except subprocess.TimeoutExpired:
        os.unlink(tmp_path)
        raise HTTPException(504, "ffmpeg timed out")


@app.post("/api/tagging/tag")
def submit_tag(req: TagRequest):
    """Submit a speaker tag for a diarization segment."""
    # Strip "+ruido" suffix for validation, keep it in the stored tag
    base_speaker = req.speaker.replace(" +ruido", "")
    if base_speaker not in TAGGING_SPEAKERS:
        raise HTTPException(400, f"Unknown speaker: {req.speaker}. Must be one of {TAGGING_SPEAKERS}")

    # Verify episode exists
    data = _load_diarization(req.episode_id)
    if data is None:
        raise HTTPException(404, f"No diarization found for episode {req.episode_id}")

    segments = data["segments"]
    if req.segment_index < 0 or req.segment_index >= len(segments):
        raise HTTPException(400, f"Segment index {req.segment_index} out of range (0-{len(segments)-1})")

    original_speaker = segments[req.segment_index].get("speaker", "unknown")

    tags = _load_tags()

    # Remove existing tag for this segment if any (allow re-tagging)
    tags = [
        t for t in tags
        if not (t["episode_id"] == req.episode_id and t["segment_index"] == req.segment_index)
    ]

    tag_entry = {
        "episode_id": req.episode_id,
        "segment_index": req.segment_index,
        "start": req.start,
        "end": req.end,
        "original_speaker": original_speaker,
        "tagged_speaker": req.speaker,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    # Extract audio clip for the dataset
    clip_path = _extract_and_save_clip(req.episode_id, req.start, req.end, tag_entry)
    if clip_path:
        tag_entry["clip_path"] = clip_path

    tags.append(tag_entry)
    _save_tags(tags)

    # Log every tag to MLflow (persistent run, incremental)
    _log_to_mlflow(tags, new_tag=tag_entry)

    return {"success": True, "total_tagged": len(tags), "tag": tag_entry}


@app.get("/api/tagging/stats")
def tagging_stats():
    """Return tagging progress statistics."""
    tags = _load_tags()

    # Total segments across all episodes
    total_segments = 0
    episode_totals: dict[str, int] = {}
    for f in sorted(DIARIZATION_DIR.glob("*.json")):
        if f.name == "speaker_profiles_ecapa.json":
            continue
        data = json.loads(f.read_text(encoding="utf-8"))
        count = len(data.get("segments", []))
        episode_totals[f.stem] = count
        total_segments += count

    # Per-episode tagged counts
    episode_tagged: dict[str, int] = {}
    speaker_counts: dict[str, int] = {}
    agreements = 0
    for t in tags:
        ep = t["episode_id"]
        episode_tagged[ep] = episode_tagged.get(ep, 0) + 1
        sp = t.get("tagged_speaker", "Otro")
        speaker_counts[sp] = speaker_counts.get(sp, 0) + 1
        if t.get("tagged_speaker") == t.get("original_speaker"):
            agreements += 1

    total_tagged = len(tags)
    agreement_rate = agreements / total_tagged if total_tagged > 0 else 0

    episodes_stats = []
    for ep_id, total in episode_totals.items():
        tagged = episode_tagged.get(ep_id, 0)
        episodes_stats.append({
            "episode_id": ep_id,
            "total_segments": total,
            "tagged": tagged,
            "progress": round(tagged / total * 100, 1) if total > 0 else 0,
        })

    return {
        "total_segments": total_segments,
        "total_tagged": total_tagged,
        "progress_percent": round(total_tagged / total_segments * 100, 1) if total_segments > 0 else 0,
        "agreement_rate": round(agreement_rate, 4),
        "speaker_counts": speaker_counts,
        "episodes": episodes_stats,
    }


# ---------------------------------------------------------------------------
# Allow running directly: python server.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8889, reload=False)
