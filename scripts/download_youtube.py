#!/usr/bin/env python3
"""Download audio for Paren La Mano YouTube episodes.

For episodes with full video in S3, downloads from S3 and extracts audio with ffmpeg.
For all other episodes, downloads audio-only via yt-dlp.

Usage:
    python scripts/download_youtube.py <youtube_id>
    python scripts/download_youtube.py --all
    python scripts/download_youtube.py --help
"""

import argparse
import json
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
EPISODES_JSON = os.path.join(PROJECT_DIR, "data", "episodes.json")
AUDIO_DIR = os.path.join(PROJECT_DIR, "data", "audio")
S3_BUCKET = "s3://plm-media-processing-390402545272/videos"

# Episodes available as full MP4 in S3 (youtube_id -> S3 filename)
S3_EPISODES = {
    "50UtSvG80yA": "ARRANCÓ PAREN LA MANO 2025 - DESDE LOS ÁNGELES ｜ #ParenLaMano Completo - 24⧸02 ｜ Vorterix-50UtSvG80yA.mp4",
    "SecttWanihg": "AYUDAMOS EN EL AMOR A NUESTROS OYENTES ｜ #ParenLaMano Completo - 27⧸03 ｜ VORTERIX-SecttWanihg.mp4",
    "-UIAVpa9KgY": "EL TRIBUNERO CON FLAVIO AZZARO ｜ #ParenLaMano Completo - 25⧸03 ｜ VORTERIX--UIAVpa9KgY.mp4",
    "g-BN4_9ORSs": "LA VENGANZA DE MARIO PERGOLINI ｜ #ParenLaMano Completo - 26⧸03 ｜ VORTERIX-g-BN4_9ORSs.mp4",
    "2N4v89apuUY": "PLM CON P DE PRIME ｜ #ParenLaMano Completo - 23⧸03 ｜ VORTERIX-2N4v89apuUY.mp4",
}

# Output format: WAV (16kHz mono) for S3 episodes (Whisper-ready), m4a for YouTube downloads
# We standardize on WAV for all, since the transcription pipeline expects it.
OUTPUT_EXT = ".wav"


def load_episodes():
    """Load episode catalog from data/episodes.json."""
    with open(EPISODES_JSON, "r") as f:
        return json.load(f)


def get_all_youtube_ids():
    """Return list of all youtube_ids from the catalog."""
    episodes = load_episodes()
    return [ep["youtube_id"] for ep in episodes]


def output_path(youtube_id):
    """Return the expected output file path for a youtube_id."""
    return os.path.join(AUDIO_DIR, f"{youtube_id}{OUTPUT_EXT}")


def download_from_s3(youtube_id):
    """Download MP4 from S3 and extract audio as WAV."""
    s3_key = S3_EPISODES[youtube_id]
    tmp_mp4 = f"/tmp/{youtube_id}.mp4"
    out = output_path(youtube_id)

    print(f"  Downloading from S3: {s3_key}")
    result = subprocess.run(
        ["aws", "s3", "cp", f"{S3_BUCKET}/{s3_key}", tmp_mp4],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  ERROR: S3 download failed: {result.stderr.strip()}")
        return False

    print(f"  Extracting audio to {out} (16kHz mono WAV)")
    result = subprocess.run(
        ["ffmpeg", "-i", tmp_mp4, "-vn", "-acodec", "pcm_s16le",
         "-ar", "16000", "-ac", "1", out, "-y", "-loglevel", "warning"],
        capture_output=True, text=True,
    )
    # Clean up MP4 regardless of result
    if os.path.exists(tmp_mp4):
        os.remove(tmp_mp4)

    if result.returncode != 0:
        print(f"  ERROR: ffmpeg extraction failed: {result.stderr.strip()}")
        return False

    size_mb = os.path.getsize(out) / (1024 * 1024)
    print(f"  Done: {size_mb:.1f} MB")
    return True


def download_from_youtube(youtube_id):
    """Download audio from YouTube via yt-dlp, then convert to WAV."""
    url = f"https://www.youtube.com/watch?v={youtube_id}"
    out = output_path(youtube_id)

    # yt-dlp downloads best audio, then we post-process to WAV 16kHz mono
    print(f"  Downloading from YouTube: {url}")
    result = subprocess.run(
        [
            "yt-dlp",
            "--extract-audio",
            "--audio-format", "wav",
            "--postprocessor-args", "ffmpeg:-ar 16000 -ac 1",
            "--output", out,
            "--no-playlist",
            "--quiet",
            "--no-warnings",
            url,
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  ERROR: yt-dlp failed: {result.stderr.strip()}")
        # Clean up partial files
        if os.path.exists(out) and os.path.getsize(out) == 0:
            os.remove(out)
        return False

    if not os.path.exists(out):
        print(f"  ERROR: output file not created")
        return False

    size_mb = os.path.getsize(out) / (1024 * 1024)
    print(f"  Done: {size_mb:.1f} MB")
    return True


def process_episode(youtube_id):
    """Download audio for a single episode. Returns True on success."""
    out = output_path(youtube_id)

    if os.path.exists(out):
        size_mb = os.path.getsize(out) / (1024 * 1024)
        print(f"[SKIP] {youtube_id} — already exists ({size_mb:.1f} MB)")
        return True

    if youtube_id in S3_EPISODES:
        print(f"[S3] {youtube_id}")
        return download_from_s3(youtube_id)
    else:
        print(f"[YT] {youtube_id}")
        return download_from_youtube(youtube_id)


def main():
    parser = argparse.ArgumentParser(
        description="Download audio for Paren La Mano YouTube episodes."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("youtube_id", nargs="?", help="YouTube video ID to download")
    group.add_argument("--all", action="store_true", help="Download all episodes from episodes.json")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without downloading")
    args = parser.parse_args()

    os.makedirs(AUDIO_DIR, exist_ok=True)

    if args.all:
        ids = get_all_youtube_ids()
    else:
        ids = [args.youtube_id]

    if args.dry_run:
        for yt_id in ids:
            out = output_path(yt_id)
            exists = os.path.exists(out)
            source = "S3" if yt_id in S3_EPISODES else "YouTube"
            status = "SKIP (exists)" if exists else f"DOWNLOAD ({source})"
            print(f"  {yt_id}: {status}")
        return

    print(f"Processing {len(ids)} episode(s)...")
    print(f"Output directory: {AUDIO_DIR}")
    print()

    succeeded = []
    failed = []
    skipped = []

    for i, yt_id in enumerate(ids, 1):
        print(f"--- [{i}/{len(ids)}] ---")
        out = output_path(yt_id)

        if os.path.exists(out):
            size_mb = os.path.getsize(out) / (1024 * 1024)
            print(f"[SKIP] {yt_id} — already exists ({size_mb:.1f} MB)")
            skipped.append(yt_id)
        elif yt_id in S3_EPISODES:
            print(f"[S3] {yt_id}")
            if download_from_s3(yt_id):
                succeeded.append(yt_id)
            else:
                failed.append(yt_id)
        else:
            print(f"[YT] {yt_id}")
            if download_from_youtube(yt_id):
                succeeded.append(yt_id)
            else:
                failed.append(yt_id)

        # Rate limit between YouTube downloads (not needed for S3 or last item)
        if i < len(ids) and yt_id not in S3_EPISODES and yt_id not in skipped:
            next_id = ids[i] if i < len(ids) else None
            if next_id and next_id not in S3_EPISODES:
                time.sleep(2.5)

        print()

    # Summary
    print("=" * 50)
    print(f"SUMMARY: {len(succeeded)} downloaded, {len(skipped)} skipped, {len(failed)} failed")
    if failed:
        print(f"FAILED: {', '.join(failed)}")
    print("=" * 50)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
