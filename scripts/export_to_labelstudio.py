#!/usr/bin/env python3
"""Generate Label Studio import JSON from diarization segments.

Usage:
    python3 scripts/export_to_labelstudio.py <episode_id> [--host HOST]

The output is written to label-studio/import_{episode_id}.json
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIARIZATION_DIR = PROJECT_ROOT / "data" / "diarization"
OUTPUT_DIR = PROJECT_ROOT / "label-studio"

# Speaker mapping from auto-detected names to Label Studio choices
SPEAKER_MAP = {
    "Roberto": "Roberto",
    "Luquitas": "Luquitas",
    "Joaquin": "Joaquin",
    "German": "German",
    "Alfredo": "Alfredo",
    "Jazmin": "Jazmin",
    "UNKNOWN": "Otra persona",
}


def export(episode_id: str, host: str = "http://35.168.141.137:8889"):
    diar_path = DIARIZATION_DIR / f"{episode_id}.json"
    if not diar_path.exists():
        print(f"ERROR: Diarization file not found: {diar_path}")
        sys.exit(1)

    with open(diar_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    tasks = []

    for idx, seg in enumerate(segments):
        start = seg["start"]
        end = seg["end"]
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "")

        audio_url = f"{host}/api/tagging/audio/{episode_id}?start={start}&end={end}"

        # Map auto speaker to a Label Studio choice
        ls_speaker = SPEAKER_MAP.get(speaker)
        if ls_speaker is None:
            # For "Otro_X" style speakers, map to "Otra persona"
            ls_speaker = "Otra persona"

        task = {
            "data": {
                "audio_url": audio_url,
                "text": text if text else "(sin transcripcion)",
                "auto_speaker": speaker,
                "episode_id": episode_id,
                "segment_index": idx,
                "start": start,
                "end": end,
            },
            "predictions": [
                {
                    "model_version": "whisperx-ecapa-tdnn",
                    "result": [
                        {
                            "from_name": "speaker",
                            "to_name": "audio",
                            "type": "choices",
                            "value": {"choices": [ls_speaker]},
                        }
                    ],
                }
            ],
        }
        tasks.append(task)

    output_path = OUTPUT_DIR / f"import_{episode_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    print(f"Exported {len(tasks)} tasks to {output_path}")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/export_to_labelstudio.py <episode_id> [--host HOST]")
        sys.exit(1)

    ep_id = sys.argv[1]
    host = "http://35.168.141.137:8889"
    if "--host" in sys.argv:
        host = sys.argv[sys.argv.index("--host") + 1]

    export(ep_id, host)
