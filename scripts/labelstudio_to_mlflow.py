#!/usr/bin/env python3
"""Export Label Studio annotations to MLflow as a dataset.

Usage:
    python3 scripts/labelstudio_to_mlflow.py <exported_json>

Or fetch directly from Label Studio API:
    python3 scripts/labelstudio_to_mlflow.py --api --project-id 1 [--ls-url http://localhost:8080] [--token TOKEN]

Logs audio clips + metadata to MLflow experiment "plm-speaker-tagging".
"""
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = PROJECT_ROOT / "data" / "audio"
DATASET_DIR = PROJECT_ROOT / "data" / "tagging" / "clips"


def extract_clip(episode_id: str, start: float, end: float, output_path: Path):
    """Extract audio clip using ffmpeg."""
    audio_file = AUDIO_DIR / f"{episode_id}.wav"
    if not audio_file.exists():
        print(f"  WARNING: Audio file not found: {audio_file}")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration = end - start

    cmd = [
        "ffmpeg", "-y", "-i", str(audio_file),
        "-ss", str(start), "-t", str(duration),
        "-ar", "16000", "-ac", "1",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=30)
    return result.returncode == 0


def parse_annotations(tasks: list) -> list:
    """Parse Label Studio export JSON into annotation records."""
    records = []
    for task in tasks:
        data = task.get("data", {})
        episode_id = data.get("episode_id", "unknown")
        segment_index = data.get("segment_index", 0)
        start = data.get("start", 0)
        end = data.get("end", 0)
        text = data.get("text", "")
        auto_speaker = data.get("auto_speaker", "")

        annotations = task.get("annotations", [])
        if not annotations:
            # Use predictions as fallback
            predictions = task.get("predictions", [])
            if predictions:
                annotations = predictions

        for ann in annotations:
            results = ann.get("result", [])
            speaker = None
            modifiers = []

            for r in results:
                if r.get("from_name") == "speaker":
                    choices = r.get("value", {}).get("choices", [])
                    if choices:
                        speaker = choices[0]
                elif r.get("from_name") == "modifiers":
                    modifiers = r.get("value", {}).get("choices", [])

            if speaker:
                records.append({
                    "episode_id": episode_id,
                    "segment_index": segment_index,
                    "start": start,
                    "end": end,
                    "text": text,
                    "auto_speaker": auto_speaker,
                    "speaker": speaker,
                    "modifiers": modifiers,
                })

    return records


def log_to_mlflow(records: list):
    """Log annotation records to MLflow."""
    import mlflow

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("plm-speaker-tagging")

    with mlflow.start_run(run_name="labelstudio-export") as run:
        # Log summary metrics
        mlflow.log_metric("total_clips", len(records))
        speaker_counts = {}
        for r in records:
            speaker_counts[r["speaker"]] = speaker_counts.get(r["speaker"], 0) + 1
        for speaker, count in speaker_counts.items():
            safe_name = speaker.replace("/", "_").replace(" ", "_")
            mlflow.log_metric(f"clips_{safe_name}", count)

        # Extract and log clips
        for i, rec in enumerate(records):
            episode_id = rec["episode_id"]
            seg_idx = rec["segment_index"]
            speaker = rec["speaker"].replace("/", "_").replace(" ", "_")

            clip_path = DATASET_DIR / speaker / f"{episode_id}_{seg_idx}.wav"

            if extract_clip(episode_id, rec["start"], rec["end"], clip_path):
                artifact_path = f"clips/{speaker}"
                mlflow.log_artifact(str(clip_path), artifact_path)

            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(records)} clips...")

        # Log the full annotations as a JSON artifact
        annotations_path = DATASET_DIR / "annotations.json"
        annotations_path.parent.mkdir(parents=True, exist_ok=True)
        with open(annotations_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        mlflow.log_artifact(str(annotations_path))

        print(f"\nMLflow run ID: {run.info.run_id}")
        print(f"Total clips logged: {len(records)}")
        print(f"Speaker breakdown: {speaker_counts}")


def fetch_from_api(ls_url: str, project_id: int, token: str) -> list:
    """Fetch annotations from Label Studio API."""
    import requests

    headers = {"Authorization": f"Token {token}"}
    url = f"{ls_url}/api/projects/{project_id}/export?exportType=JSON"
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json()


if __name__ == "__main__":
    if "--api" in sys.argv:
        ls_url = "http://localhost:8080"
        project_id = 1
        token = os.environ.get("LABEL_STUDIO_TOKEN", "")

        if "--ls-url" in sys.argv:
            ls_url = sys.argv[sys.argv.index("--ls-url") + 1]
        if "--project-id" in sys.argv:
            project_id = int(sys.argv[sys.argv.index("--project-id") + 1])
        if "--token" in sys.argv:
            token = sys.argv[sys.argv.index("--token") + 1]

        if not token:
            print("ERROR: Provide --token or set LABEL_STUDIO_TOKEN env var")
            sys.exit(1)

        print(f"Fetching annotations from {ls_url} project {project_id}...")
        tasks = fetch_from_api(ls_url, project_id, token)
    elif len(sys.argv) >= 2 and not sys.argv[1].startswith("-"):
        export_file = Path(sys.argv[1])
        if not export_file.exists():
            print(f"ERROR: File not found: {export_file}")
            sys.exit(1)
        with open(export_file, "r", encoding="utf-8") as f:
            tasks = json.load(f)
    else:
        print("Usage:")
        print("  python3 scripts/labelstudio_to_mlflow.py <exported.json>")
        print("  python3 scripts/labelstudio_to_mlflow.py --api --project-id 1 --token TOKEN")
        sys.exit(1)

    records = parse_annotations(tasks)
    if not records:
        print("No annotations found to export.")
        sys.exit(0)

    print(f"Found {len(records)} annotated segments. Logging to MLflow...")
    log_to_mlflow(records)
