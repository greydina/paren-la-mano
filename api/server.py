"""
Paren la Mano — RAG Chat API
Semantic search over episode data using sentence-transformers.
Letterboxd-style ratings system for episodes.
Run: uvicorn server:app --host 0.0.0.0 --port 8889
"""

import json
import os
import random
import sqlite3
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EPISODES_PATH = Path(__file__).resolve().parent.parent / "data" / "episodes.json"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 5
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


@app.on_event("startup")
def load_data():
    global episodes, episode_texts, episode_embeddings, model

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


class EpisodeResult(BaseModel):
    id: int
    titulo: str
    fecha: str
    descripcion: str
    youtube_id: str
    duracion: str
    numero: int | None = None
    score: float


class ChatResponse(BaseModel):
    answer: str
    episodes: list[EpisodeResult]


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

    results = semantic_search(query)
    answer = format_answer(query, results)
    ep_results = [
        EpisodeResult(**ep, score=round(score, 4)) for ep, score in results
    ]
    return ChatResponse(answer=answer, episodes=ep_results)


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
