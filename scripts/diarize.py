#!/usr/bin/env python3
"""Speaker diarization pipeline using resemblyzer + webrtcvad.

pyannote-audio is broken on this machine (torchaudio version mismatch with
torch 2.11) and would likely OOM on 3.8GB RAM anyway. This script uses a
lighter-weight approach:

  1. webrtcvad — Voice Activity Detection to find speech regions
  2. resemblyzer — Speaker embedding (256-dim) for each speech segment
  3. sklearn AgglomerativeClustering — Group segments by speaker

Input:  data/audio/<youtube_id>.wav  (16kHz mono)
Output: data/diarization/<youtube_id>.json

Usage:
    python scripts/diarize.py <youtube_id>
    python scripts/diarize.py --all
    python scripts/diarize.py <youtube_id> --num-speakers 4

If voice profiles exist in S3, speakers are labeled by name (Roberto,
Luquitas, Joaquin, German). Otherwise they get generic labels (SPEAKER_0, etc).
"""

import argparse
import gc
import json
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import webrtcvad

AUDIO_DIR = Path(__file__).resolve().parent.parent / "data" / "audio"
DIARIZATION_DIR = Path(__file__).resolve().parent.parent / "data" / "diarization"
VOICE_PROFILES_CACHE = Path(__file__).resolve().parent.parent / "data" / "voice_profiles.json"
SUPPORTED_EXTENSIONS = (".wav", ".m4a", ".mp4", ".mp3", ".ogg", ".webm")

# VAD parameters
VAD_AGGRESSIVENESS = 2  # 0-3, higher = more aggressive filtering
FRAME_DURATION_MS = 30  # 10, 20, or 30 ms
SAMPLE_RATE = 16000

# Segment parameters
MIN_SEGMENT_DURATION = 1.0   # Minimum speech segment in seconds
MAX_SEGMENT_DURATION = 30.0  # Max segment for embedding (longer = more RAM)
MERGE_GAP = 0.5             # Merge speech segments closer than this (seconds)

# Embedding chunk: process audio in N-minute windows to limit RAM
CHUNK_MINUTES = 10


def find_audio_file(youtube_id: str) -> Path | None:
    """Find the audio file for a given youtube_id."""
    for ext in SUPPORTED_EXTENSIONS:
        path = AUDIO_DIR / f"{youtube_id}{ext}"
        if path.exists():
            return path
    return None


def list_audio_ids() -> list[str]:
    """Return youtube_ids for all audio files in the audio directory."""
    ids = []
    if not AUDIO_DIR.exists():
        return ids
    for f in sorted(AUDIO_DIR.iterdir()):
        if f.suffix in SUPPORTED_EXTENSIONS:
            ids.append(f.stem)
    return ids


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def get_audio_duration(audio_path: Path) -> float:
    """Get audio duration in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path),
        ],
        capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def load_audio_chunk(audio_path: Path, start_sec: float, duration: float) -> np.ndarray:
    """Load a chunk of audio as 16kHz mono int16 using ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-ss", str(start_sec), "-t", str(duration),
        "-i", str(audio_path),
        "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "s16le", "-acodec", "pcm_s16le",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        return np.array([], dtype=np.int16)
    return np.frombuffer(result.stdout, dtype=np.int16)


def run_vad(pcm_int16: np.ndarray, sample_rate: int = SAMPLE_RATE) -> list[tuple[float, float]]:
    """Run webrtcvad on PCM int16 audio, return list of (start, end) in seconds."""
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    frame_size = int(sample_rate * FRAME_DURATION_MS / 1000)  # samples per frame
    frame_bytes = frame_size * 2  # 2 bytes per int16 sample

    raw = pcm_int16.tobytes()
    num_frames = len(raw) // frame_bytes

    speech_frames = []
    for i in range(num_frames):
        offset = i * frame_bytes
        frame = raw[offset:offset + frame_bytes]
        is_speech = vad.is_speech(frame, sample_rate)
        speech_frames.append(is_speech)

    # Convert frame-level decisions to segments
    segments = []
    in_speech = False
    start = 0.0
    for i, is_speech in enumerate(speech_frames):
        t = i * FRAME_DURATION_MS / 1000.0
        if is_speech and not in_speech:
            start = t
            in_speech = True
        elif not is_speech and in_speech:
            end = t
            if end - start >= MIN_SEGMENT_DURATION:
                segments.append((start, end))
            in_speech = False
    if in_speech:
        end = len(speech_frames) * FRAME_DURATION_MS / 1000.0
        if end - start >= MIN_SEGMENT_DURATION:
            segments.append((start, end))

    # Merge close segments
    merged = []
    for seg in segments:
        if merged and seg[0] - merged[-1][1] < MERGE_GAP:
            merged[-1] = (merged[-1][0], seg[1])
        else:
            merged.append(seg)

    return merged


def split_long_segments(segments: list[tuple[float, float]],
                        max_dur: float = MAX_SEGMENT_DURATION) -> list[tuple[float, float]]:
    """Split segments longer than max_dur into smaller pieces."""
    result = []
    for start, end in segments:
        dur = end - start
        if dur <= max_dur:
            result.append((start, end))
        else:
            n_parts = int(np.ceil(dur / max_dur))
            part_dur = dur / n_parts
            for i in range(n_parts):
                s = start + i * part_dur
                e = start + (i + 1) * part_dur
                result.append((round(s, 3), round(e, 3)))
    return result


def get_voice_profiles() -> dict | None:
    """Load voice profiles from cache or S3."""
    if VOICE_PROFILES_CACHE.exists():
        with open(VOICE_PROFILES_CACHE) as f:
            data = json.load(f)
        return {name: np.array(p["embedding"]) for name, p in data["speakers"].items()}

    # Try downloading from S3
    try:
        result = subprocess.run(
            [
                "aws", "s3", "cp",
                "s3://plm-datasets-390402545272/plm-voices/plm_final_voice_profiles.json",
                str(VOICE_PROFILES_CACHE),
            ],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0 and VOICE_PROFILES_CACHE.exists():
            with open(VOICE_PROFILES_CACHE) as f:
                data = json.load(f)
            return {name: np.array(p["embedding"]) for name, p in data["speakers"].items()}
    except Exception as e:
        print(f"  Warning: could not load voice profiles from S3: {e}")

    return None


def label_speakers_with_profiles(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    profiles: dict[str, np.ndarray],
) -> dict[int, str]:
    """Map cluster IDs to speaker names using voice profiles (cosine similarity)."""
    unique_clusters = sorted(set(cluster_labels))
    cluster_centroids = {}
    for c in unique_clusters:
        mask = cluster_labels == c
        cluster_centroids[c] = embeddings[mask].mean(axis=0)

    # Compute cosine similarity between each cluster centroid and each profile
    label_map = {}
    used_names = set()
    similarities = []

    for c, centroid in cluster_centroids.items():
        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-10)
        for name, profile_emb in profiles.items():
            profile_norm = profile_emb / (np.linalg.norm(profile_emb) + 1e-10)
            sim = float(np.dot(centroid_norm, profile_norm))
            similarities.append((sim, c, name))

    # Greedy assignment: best similarity first
    similarities.sort(reverse=True)
    used_clusters = set()
    for sim, c, name in similarities:
        if c in used_clusters or name in used_names:
            continue
        if sim > 0.3:  # Minimum threshold for a match
            label_map[c] = name
            used_clusters.add(c)
            used_names.add(name)

    # Assign generic labels to unmatched clusters
    generic_idx = 0
    for c in unique_clusters:
        if c not in label_map:
            while f"SPEAKER_{generic_idx}" in used_names:
                generic_idx += 1
            label_map[c] = f"SPEAKER_{generic_idx}"
            used_names.add(label_map[c])
            generic_idx += 1

    return label_map


def diarize_file(audio_path: Path, youtube_id: str, num_speakers: int | None = None) -> None:
    """Diarize a single audio file."""
    json_path = DIARIZATION_DIR / f"{youtube_id}.json"

    if json_path.exists():
        print(f"  Skipping {youtube_id} (already diarized)")
        return

    print(f"  Diarizing {audio_path.name} ...")
    start_time = time.time()

    total_duration = get_audio_duration(audio_path)
    print(f"  Audio duration: {format_timestamp(total_duration)} ({total_duration:.0f}s)")

    # --- Step 1: Voice Activity Detection (chunked) ---
    print("  Step 1/3: Voice Activity Detection ...", flush=True)
    all_speech_segments = []
    chunk_seconds = CHUNK_MINUTES * 60

    n_chunks = int(np.ceil(total_duration / chunk_seconds))
    for chunk_idx in range(n_chunks):
        chunk_start = chunk_idx * chunk_seconds
        chunk_dur = min(chunk_seconds, total_duration - chunk_start)
        print(
            f"    VAD chunk {chunk_idx + 1}/{n_chunks} "
            f"({format_timestamp(chunk_start)}-{format_timestamp(chunk_start + chunk_dur)})",
            flush=True,
        )
        pcm = load_audio_chunk(audio_path, chunk_start, chunk_dur)
        if len(pcm) == 0:
            continue
        segments = run_vad(pcm)
        # Offset segments to absolute time
        for s, e in segments:
            all_speech_segments.append((round(s + chunk_start, 3), round(e + chunk_start, 3)))
        del pcm
        gc.collect()

    # Merge segments that got split across chunk boundaries
    all_speech_segments.sort()
    merged_segments = []
    for seg in all_speech_segments:
        if merged_segments and seg[0] - merged_segments[-1][1] < MERGE_GAP:
            merged_segments[-1] = (merged_segments[-1][0], seg[1])
        else:
            merged_segments.append(seg)

    # Split long segments for better per-speaker embedding
    segments_for_embedding = split_long_segments(merged_segments)
    total_speech = sum(e - s for s, e in segments_for_embedding)
    print(
        f"  Found {len(segments_for_embedding)} speech segments "
        f"({format_timestamp(total_speech)} of speech)"
    )

    if not segments_for_embedding:
        print("  Warning: no speech segments found!")
        output = {
            "youtube_id": youtube_id,
            "duration": round(total_duration, 2),
            "num_speakers": 0,
            "segments": [],
            "method": "webrtcvad+resemblyzer",
        }
        DIARIZATION_DIR.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    # --- Step 2: Speaker Embeddings ---
    print("  Step 2/3: Computing speaker embeddings (resemblyzer) ...", flush=True)
    from resemblyzer import VoiceEncoder

    encoder = VoiceEncoder("cpu")

    embeddings = []
    batch_size = 50  # Process N segments, then gc
    for i, (seg_start, seg_end) in enumerate(segments_for_embedding):
        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - start_time
            print(
                f"    Embedding segment {i + 1}/{len(segments_for_embedding)} "
                f"(elapsed {elapsed:.0f}s)",
                flush=True,
            )

        # Load segment audio as float32
        dur = seg_end - seg_start
        pcm = load_audio_chunk(audio_path, seg_start, dur)
        if len(pcm) < SAMPLE_RATE * 0.5:  # Skip very short segments (< 0.5s of audio)
            embeddings.append(np.zeros(256, dtype=np.float32))
            continue
        wav_float = pcm.astype(np.float32) / 32768.0
        del pcm

        try:
            emb = encoder.embed_utterance(wav_float)
            embeddings.append(emb)
        except Exception:
            embeddings.append(np.zeros(256, dtype=np.float32))
        del wav_float

        if (i + 1) % batch_size == 0:
            gc.collect()

    embeddings_array = np.stack(embeddings)
    del embeddings
    gc.collect()

    # Remove zero embeddings (failed segments) from clustering
    valid_mask = np.linalg.norm(embeddings_array, axis=1) > 0.1
    valid_indices = np.where(valid_mask)[0]
    valid_embeddings = embeddings_array[valid_mask]

    if len(valid_embeddings) < 2:
        print("  Warning: too few valid embeddings for clustering")
        output = {
            "youtube_id": youtube_id,
            "duration": round(total_duration, 2),
            "num_speakers": 1,
            "segments": [
                {
                    "start": round(s, 3),
                    "end": round(e, 3),
                    "speaker": "SPEAKER_0",
                }
                for s, e in segments_for_embedding
            ],
            "method": "webrtcvad+resemblyzer",
        }
        DIARIZATION_DIR.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    # --- Step 3: Clustering ---
    print("  Step 3/3: Clustering speakers ...", flush=True)
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    if num_speakers is not None:
        n_clusters = num_speakers
        print(f"    Using specified number of speakers: {n_clusters}")
    else:
        # Try different cluster counts and pick the best silhouette score
        # PLM typically has 4 speakers but some episodes may have guests
        best_score = -1
        best_n = 4  # default
        for n in range(2, 7):
            if n >= len(valid_embeddings):
                break
            clustering = AgglomerativeClustering(
                n_clusters=n, metric="cosine", linkage="average"
            )
            labels = clustering.fit_predict(valid_embeddings)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(valid_embeddings, labels, metric="cosine")
            print(f"    n_speakers={n}: silhouette={score:.3f}")
            if score > best_score:
                best_score = score
                best_n = n
        n_clusters = best_n
        print(f"    Selected {n_clusters} speakers (silhouette={best_score:.3f})")

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters, metric="cosine", linkage="average"
    )
    cluster_labels = clustering.fit_predict(valid_embeddings)

    # Map cluster IDs to all segments (including invalid ones)
    full_labels = np.full(len(segments_for_embedding), -1, dtype=int)
    full_labels[valid_indices] = cluster_labels

    # Assign invalid segments to nearest valid neighbor's cluster
    for i in range(len(full_labels)):
        if full_labels[i] == -1:
            # Find nearest valid segment by time proximity
            seg_mid = (segments_for_embedding[i][0] + segments_for_embedding[i][1]) / 2
            best_dist = float("inf")
            best_label = 0
            for j in valid_indices:
                other_mid = (segments_for_embedding[j][0] + segments_for_embedding[j][1]) / 2
                dist = abs(seg_mid - other_mid)
                if dist < best_dist:
                    best_dist = dist
                    best_label = cluster_labels[np.where(valid_indices == j)[0][0]]
            full_labels[i] = best_label

    # Try to label with voice profiles
    profiles = get_voice_profiles()
    if profiles is not None:
        print("  Labeling speakers using voice profiles ...")
        label_map = label_speakers_with_profiles(valid_embeddings, cluster_labels, profiles)
        print(f"    Speaker map: { {v: k for k, v in label_map.items()} }")
    else:
        print("  No voice profiles available, using generic labels")
        label_map = {i: f"SPEAKER_{i}" for i in range(n_clusters)}

    # Build output segments
    output_segments = []
    for i, (seg_start, seg_end) in enumerate(segments_for_embedding):
        cluster_id = int(full_labels[i])
        speaker = label_map.get(cluster_id, f"SPEAKER_{cluster_id}")
        output_segments.append({
            "start": round(seg_start, 3),
            "end": round(seg_end, 3),
            "speaker": speaker,
        })

    # Merge consecutive segments with same speaker
    merged_output = []
    for seg in output_segments:
        if merged_output and merged_output[-1]["speaker"] == seg["speaker"]:
            gap = seg["start"] - merged_output[-1]["end"]
            if gap < 2.0:  # Merge if gap < 2 seconds
                merged_output[-1]["end"] = seg["end"]
                continue
        merged_output.append(dict(seg))

    elapsed = time.time() - start_time

    # Speaker stats
    speaker_times = {}
    for seg in merged_output:
        dur = seg["end"] - seg["start"]
        speaker_times[seg["speaker"]] = speaker_times.get(seg["speaker"], 0) + dur

    output = {
        "youtube_id": youtube_id,
        "duration": round(total_duration, 2),
        "num_speakers": n_clusters,
        "speaker_stats": {
            name: {
                "total_seconds": round(t, 1),
                "percentage": round(100 * t / total_duration, 1),
            }
            for name, t in sorted(speaker_times.items(), key=lambda x: -x[1])
        },
        "segments": merged_output,
        "method": "webrtcvad+resemblyzer",
        "notes": (
            "pyannote-audio unavailable (torchaudio/torch version mismatch). "
            "Using webrtcvad for VAD + resemblyzer for speaker embeddings + "
            "agglomerative clustering. Voice profiles from S3 used for labeling."
        ),
    }

    DIARIZATION_DIR.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n  Done in {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"  {len(merged_output)} segments, {n_clusters} speakers")
    for name, stats in output["speaker_stats"].items():
        print(f"    {name}: {format_timestamp(stats['total_seconds'])} ({stats['percentage']}%)")
    print(f"  Output: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Speaker diarization using resemblyzer + webrtcvad"
    )
    parser.add_argument(
        "youtube_id", nargs="?", help="YouTube video ID to diarize"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Diarize all audio files in data/audio/",
    )
    parser.add_argument(
        "--num-speakers", type=int, default=None,
        help="Number of speakers (auto-detected if not specified)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-diarize even if output already exists",
    )
    args = parser.parse_args()

    if not args.youtube_id and not args.all:
        parser.print_help()
        sys.exit(1)

    # Check available memory
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    avail_mb = int(line.split()[1]) / 1024
                    print(f"Available memory: {avail_mb:.0f} MB")
                    if avail_mb < 500:
                        print("WARNING: Less than 500 MB available. Diarization may be slow or fail.")
                    break
    except Exception:
        pass

    # Determine which IDs to process
    if args.all:
        ids = list_audio_ids()
        if not ids:
            print(f"No audio files found in {AUDIO_DIR}")
            sys.exit(1)
        print(f"Found {len(ids)} audio file(s) to process")
    else:
        audio = find_audio_file(args.youtube_id)
        if not audio:
            print(f"Error: no audio file found for {args.youtube_id} in {AUDIO_DIR}")
            sys.exit(1)
        ids = [args.youtube_id]

    DIARIZATION_DIR.mkdir(parents=True, exist_ok=True)

    for i, vid_id in enumerate(ids, 1):
        print(f"\nEpisode {i} of {len(ids)}: {vid_id}")
        audio = find_audio_file(vid_id)
        if not audio:
            print("  Warning: no audio file found, skipping")
            continue

        json_path = DIARIZATION_DIR / f"{vid_id}.json"
        if json_path.exists() and not args.force:
            print(f"  Skipping {vid_id} (already diarized, use --force to redo)")
            continue

        if args.force and json_path.exists():
            json_path.unlink()

        diarize_file(audio, vid_id, num_speakers=args.num_speakers)
        gc.collect()

    print("\nAll done.")


if __name__ == "__main__":
    main()
