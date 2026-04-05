#!/usr/bin/env bash
# Extract audio from S3-hosted MP4 videos.
#
# Usage:
#   ./scripts/extract_audio_from_s3.sh <youtube_id>
#   ./scripts/extract_audio_from_s3.sh --all
#
# Downloads MP4 from S3, extracts audio as WAV (16kHz mono for Whisper), deletes the MP4.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
AUDIO_DIR="$PROJECT_DIR/data/audio"
S3_BUCKET="s3://plm-media-processing-390402545272/videos"

mkdir -p "$AUDIO_DIR"

# Episode mappings: youtube_id -> S3 filename
declare -A EPISODES=(
  ["50UtSvG80yA"]="ARRANCÓ PAREN LA MANO 2025 - DESDE LOS ÁNGELES ｜ #ParenLaMano Completo - 24⧸02 ｜ Vorterix-50UtSvG80yA.mp4"
  ["SecttWanihg"]="AYUDAMOS EN EL AMOR A NUESTROS OYENTES ｜ #ParenLaMano Completo - 27⧸03 ｜ VORTERIX-SecttWanihg.mp4"
  ["-UIAVpa9KgY"]="EL TRIBUNERO CON FLAVIO AZZARO ｜ #ParenLaMano Completo - 25⧸03 ｜ VORTERIX--UIAVpa9KgY.mp4"
  ["g-BN4_9ORSs"]="LA VENGANZA DE MARIO PERGOLINI ｜ #ParenLaMano Completo - 26⧸03 ｜ VORTERIX-g-BN4_9ORSs.mp4"
  ["2N4v89apuUY"]="PLM CON P DE PRIME ｜ #ParenLaMano Completo - 23⧸03 ｜ VORTERIX-2N4v89apuUY.mp4"
)

extract_audio() {
  local yt_id="$1"
  local s3_key="${2:-}"
  local output="$AUDIO_DIR/${yt_id}.wav"

  if [[ -f "$output" ]]; then
    echo "Skipping $yt_id (audio already exists)"
    return 0
  fi

  # Resolve S3 key from mapping if not provided
  if [[ -z "$s3_key" ]]; then
    if [[ -v "EPISODES[$yt_id]" ]]; then
      s3_key="${EPISODES[$yt_id]}"
    else
      echo "Error: no S3 key provided and $yt_id not in episode mappings"
      return 1
    fi
  fi

  local tmp_mp4="/tmp/${yt_id}.mp4"

  echo "Downloading: $s3_key"
  aws s3 cp "$S3_BUCKET/$s3_key" "$tmp_mp4"

  echo "Verifying download..."
  ffprobe "$tmp_mp4" 2>&1 | grep Duration || { echo "Error: download corrupt"; rm -f "$tmp_mp4"; return 1; }

  echo "Extracting audio to $output (16kHz mono WAV)"
  ffmpeg -i "$tmp_mp4" -vn -acodec pcm_s16le -ar 16000 -ac 1 "$output" -y -loglevel warning

  echo "Cleaning up MP4"
  rm -f "$tmp_mp4"

  echo "Done: $output ($(du -h "$output" | cut -f1))"
}

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <youtube_id> [s3_key]"
  echo "       $0 --all"
  exit 1
fi

if [[ "$1" == "--all" ]]; then
  echo "Processing all ${#EPISODES[@]} episodes..."
  for yt_id in "${!EPISODES[@]}"; do
    echo ""
    echo "=== $yt_id ==="
    extract_audio "$yt_id"
  done
  echo ""
  echo "All done."
else
  extract_audio "$1" "${2:-}"
fi
