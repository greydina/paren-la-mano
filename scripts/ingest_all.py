#!/usr/bin/env python3
"""End-to-end pipeline orchestrator for Paren La Mano episode ingestion.

Runs all four pipeline stages sequentially:
  1. Download audio   (download_youtube.py)
  2. Transcribe       (transcribe_local.py)
  3. Chunk + embed    (generate_chunks.py)
  4. Load to DB       (load_to_db.py)

Each stage processes all episodes and has built-in skip logic for already-
completed work, providing natural resume support.

Usage:
    python scripts/ingest_all.py              # run full pipeline
    python scripts/ingest_all.py --dry-run    # preview what would be done
    python scripts/ingest_all.py --stage 2    # start from stage N (1-4)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
EPISODES_FILE = PROJECT_DIR / "data" / "episodes.json"
AUDIO_DIR = PROJECT_DIR / "data" / "audio"
TRANSCRIPTS_DIR = PROJECT_DIR / "data" / "transcripts"
EMBEDDINGS_DIR = PROJECT_DIR / "data" / "embeddings"

# Pipeline stages: (name, script, description)
STAGES = [
    ("download", "download_youtube.py", "Download audio"),
    ("transcribe", "transcribe_local.py", "Transcribe with Whisper"),
    ("chunk_embed", "generate_chunks.py", "Chunk transcripts + generate embeddings"),
    ("load_db", "load_to_db.py", "Load to PostgreSQL + pgvector"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_episodes() -> list[dict]:
    with open(EPISODES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def format_duration(seconds: float) -> str:
    """Format seconds as Xh Ym Zs."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    parts = []
    if h:
        parts.append(f"{h}h")
    if m or h:
        parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)


def episode_status(youtube_id: str) -> dict:
    """Check which pipeline outputs exist for a given episode."""
    audio_exists = any(
        (AUDIO_DIR / f"{youtube_id}{ext}").exists()
        for ext in (".wav", ".m4a", ".mp4", ".mp3", ".ogg", ".webm")
    )
    transcript_exists = (TRANSCRIPTS_DIR / f"{youtube_id}.json").exists()
    chunks_exist = (
        (EMBEDDINGS_DIR / f"{youtube_id}_chunks.json").exists()
        and (EMBEDDINGS_DIR / f"{youtube_id}.npy").exists()
    )
    # DB check would require psycopg2; we approximate by checking chunks exist
    # (load_to_db has its own skip logic via already_loaded())
    return {
        "audio": audio_exists,
        "transcript": transcript_exists,
        "chunks": chunks_exist,
    }


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def dry_run(episodes: list[dict], start_stage: int):
    """Preview what work each stage would do."""
    print("=" * 60)
    print("DRY RUN — Pipeline preview")
    print(f"Episodes in catalog: {len(episodes)}")
    print(f"Starting from stage: {start_stage}")
    print("=" * 60)
    print()

    # Gather status for all episodes
    needs_download = []
    needs_transcribe = []
    needs_chunks = []
    needs_load = []

    for ep in episodes:
        yt_id = ep["youtube_id"]
        status = episode_status(yt_id)

        if not status["audio"]:
            needs_download.append(yt_id)
        if not status["transcript"]:
            needs_transcribe.append(yt_id)
        if not status["chunks"]:
            needs_chunks.append(yt_id)
        # For DB loading, we check if chunks exist (load_to_db checks DB internally)
        if status["chunks"]:
            needs_load.append(yt_id)  # candidates; actual skip handled by script

    stage_info = [
        (1, "Download audio", needs_download, len(episodes)),
        (2, "Transcribe", needs_transcribe, len(episodes)),
        (3, "Chunk + embed", needs_chunks, len(episodes)),
        (4, "Load to DB", needs_load, len(episodes)),
    ]

    for stage_num, name, pending, total in stage_info:
        if stage_num < start_stage:
            marker = "SKIP (--stage)"
        else:
            marker = f"{len(pending)} pending"
        done = total - len(pending)
        print(f"  Stage {stage_num}: {name}")
        print(f"    {done}/{total} already done, {marker}")
        if pending and stage_num >= start_stage:
            for yt_id in pending[:5]:
                title = next((e["titulo"] for e in episodes if e["youtube_id"] == yt_id), yt_id)
                print(f"      - {yt_id}  {title[:50]}")
            if len(pending) > 5:
                print(f"      ... and {len(pending) - 5} more")
        print()

    # Summary
    total_work = sum(
        len(info[2]) for info in stage_info if info[0] >= start_stage
    )
    if total_work == 0:
        print("Nothing to do — all episodes are fully ingested.")
    else:
        print(f"Total items to process across stages: {total_work}")
        print("Run without --dry-run to execute.")


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

def run_stage(stage_num: int, name: str, script: str, description: str) -> bool:
    """Run a pipeline stage script with --all. Returns True on success."""
    script_path = SCRIPT_DIR / script
    print(f"{'=' * 60}")
    print(f"STAGE {stage_num}/4: {description}")
    print(f"Script: {script_path}")
    print(f"{'=' * 60}")
    print()

    start = time.time()
    result = subprocess.run(
        [sys.executable, str(script_path), "--all"],
        cwd=str(PROJECT_DIR),
    )
    elapsed = time.time() - start

    print()
    if result.returncode == 0:
        print(f"Stage {stage_num} completed in {format_duration(elapsed)}")
    else:
        print(f"Stage {stage_num} FAILED (exit code {result.returncode}) "
              f"after {format_duration(elapsed)}")
    print()

    return result.returncode == 0


def run_pipeline(episodes: list[dict], start_stage: int):
    """Run the full ingestion pipeline."""
    total_start = time.time()

    print("=" * 60)
    print("PAREN LA MANO — Full Ingestion Pipeline")
    print(f"Episodes: {len(episodes)}")
    print(f"Starting from stage: {start_stage}")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()

    results = {}

    for i, (key, script, description) in enumerate(STAGES, 1):
        if i < start_stage:
            print(f"Skipping stage {i} ({description}) — starting from stage {start_stage}")
            print()
            results[key] = "skipped"
            continue

        # Show pre-stage status
        statuses = [episode_status(ep["youtube_id"]) for ep in episodes]
        status_key = {
            "download": "audio",
            "transcribe": "transcript",
            "chunk_embed": "chunks",
            "load_db": "chunks",
        }[key]
        done = sum(1 for s in statuses if s[status_key])
        pending = len(episodes) - done
        print(f"Pre-check: {done}/{len(episodes)} episodes already have "
              f"{status_key} output ({pending} pending)")
        print()

        if pending == 0 and key != "load_db":
            print(f"Nothing to do for stage {i}, moving on.")
            print()
            results[key] = "nothing_to_do"
            continue

        success = run_stage(i, key, script, description)
        results[key] = "ok" if success else "failed"

        if not success:
            print(f"Pipeline stopped at stage {i} due to failure.")
            print("You can resume with: python scripts/ingest_all.py --stage", i)
            break

    # Final summary
    total_elapsed = time.time() - total_start
    print()
    print("=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    for i, (key, _, description) in enumerate(STAGES, 1):
        status = results.get(key, "not_reached")
        icon = {
            "ok": "OK",
            "failed": "FAILED",
            "skipped": "SKIPPED",
            "nothing_to_do": "OK (nothing to do)",
            "not_reached": "NOT REACHED",
        }.get(status, status)
        print(f"  Stage {i}: {description:40s} [{icon}]")

    print()
    print(f"Total time: {format_duration(total_elapsed)}")
    print(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Final episode status
    fully_done = 0
    for ep in episodes:
        s = episode_status(ep["youtube_id"])
        if s["audio"] and s["transcript"] and s["chunks"]:
            fully_done += 1
    print(f"Episodes fully processed (through embedding): {fully_done}/{len(episodes)}")
    print("=" * 60)

    if any(v == "failed" for v in results.values()):
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end ingestion pipeline for Paren La Mano episodes"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be done without executing",
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2, 3, 4],
        default=1,
        help="Start from this stage (1=download, 2=transcribe, 3=chunk+embed, 4=load)",
    )
    args = parser.parse_args()

    if not EPISODES_FILE.exists():
        print(f"Error: episodes catalog not found: {EPISODES_FILE}")
        sys.exit(1)

    episodes = load_episodes()
    if not episodes:
        print("No episodes found in catalog.")
        sys.exit(1)

    if args.dry_run:
        dry_run(episodes, args.stage)
    else:
        run_pipeline(episodes, args.stage)


if __name__ == "__main__":
    main()
