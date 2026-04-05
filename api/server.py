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
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

try:
    import psycopg2
    import psycopg2.extras
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EPISODES_PATH = Path(__file__).resolve().parent.parent / "data" / "episodes.json"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 5
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://plm_user:plm_pass_2026@localhost:5432/plm_db",
)
KNOWN_SPEAKERS = ["Roberto", "Luquitas", "Joaquin", "German"]
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
    global episodes, episode_texts, episode_embeddings, model, pg_available

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
    youtube_url: str | None = None


class ChatResponse(BaseModel):
    answer: str
    episodes: list[EpisodeResult]
    chunks: list[ChunkResult] = []
    source: str = "episodes"  # "episodes" or "chunks"


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


def pgvector_search(query: str, top_k: int = TOP_K, speaker: str | None = None) -> list[ChunkResult] | None:
    """Search transcript chunks via pgvector. Returns None if DB unavailable."""
    conn = _get_pg()
    if not conn:
        return None
    try:
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
                (emb_str, speaker, emb_str, top_k),
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
                (emb_str, emb_str, top_k),
            )
        rows = cur.fetchall()
        cur.close()
        conn.close()

        results = []
        for row in rows:
            similarity = 1.0 - float(row["distance"])
            if similarity < 0.1:
                continue
            clean_text = _strip_episode_prefix(row["text"])
            video_id = row["video_id"]
            start = row["start_time"]
            results.append(ChunkResult(
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
        return results
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
            if speaker_fallback:
                answer = (
                    f"No encontré fragmentos específicos de {speaker}, "
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
# Allow running directly: python server.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8889, reload=False)
