#!/usr/bin/env python3
"""
Speaker diarization using SpeechBrain ECAPA-TDNN + spectral clustering.
No gated models required. Runs on c5.2xlarge (16GB RAM, CPU).

Pipeline:
1. Energy-based VAD to find speech segments
2. SpeechBrain ECAPA-TDNN embeddings (192-dim) per segment
3. Spectral clustering to group speakers
4. Cosine similarity against voice profiles to identify speakers

Usage:
  python3 diarize_speechbrain.py --all
  python3 diarize_speechbrain.py --build-profiles
  python3 diarize_speechbrain.py --episode ID
"""

import argparse
import gc
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio

AUDIO_DIR = Path.home() / "audio"
OUTPUT_DIR = Path.home() / "output"
VOICE_SAMPLES_DIR = Path.home() / "voice_samples"
PROFILES_FILE = Path.home() / "speaker_profiles_ecapa.json"

KNOWN_SPEAKERS = ["Luquitas", "Roberto", "Joaquin", "German", "Alfredo"]

SAMPLE_MAP = {
    "Luquitas": ["luquitas_sample1.mp3", "luquitas_sample2.mp3"],
    "Roberto": ["roberto_sample1.mp3"],
    "Joaquin": ["joaquin_sample1.mp3"],
    "German": ["german_sample1.mp3", "german_sample2.mp3", "german_sample3.mp3"],
}

# Diarization parameters
SEGMENT_MIN_DURATION = 1.5   # min speech segment duration (seconds)
SEGMENT_MAX_DURATION = 15.0  # max segment for embedding (longer ones are split)
VAD_ENERGY_THRESHOLD = 0.02  # energy threshold for VAD
VAD_MIN_SILENCE = 0.4        # min silence gap to split segments
MERGE_GAP = 0.8              # merge same-speaker segments closer than this


def load_embedding_model():
    """Load SpeechBrain ECAPA-TDNN speaker verification model."""
    from speechbrain.inference.speaker import EncoderClassifier
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": "cpu"},
    )
    print("ECAPA-TDNN model loaded")
    return model


def embed_audio(model, waveform, sample_rate):
    """Get speaker embedding for an audio waveform."""
    # Resample to 16kHz if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)

    # Ensure mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    with torch.no_grad():
        embedding = model.encode_batch(waveform)
    return embedding.squeeze().numpy()


def energy_vad(waveform, sample_rate, frame_ms=30,
               energy_threshold=VAD_ENERGY_THRESHOLD,
               min_speech=SEGMENT_MIN_DURATION,
               min_silence=VAD_MIN_SILENCE):
    """Simple energy-based Voice Activity Detection."""
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    audio = waveform.squeeze().numpy()

    frame_size = int(sample_rate * frame_ms / 1000)
    n_frames = len(audio) // frame_size

    # Compute frame energy
    energies = []
    for i in range(n_frames):
        frame = audio[i * frame_size:(i + 1) * frame_size]
        energies.append(np.sqrt(np.mean(frame ** 2)))

    energies = np.array(energies)

    # Adaptive threshold: use percentile if fixed threshold is too high/low
    p30 = np.percentile(energies, 30)
    p70 = np.percentile(energies, 70)
    threshold = max(energy_threshold, p30 + 0.3 * (p70 - p30))

    # Find speech frames
    is_speech = energies > threshold

    # Convert to segments
    segments = []
    in_speech = False
    start = 0
    silence_frames = 0
    min_silence_frames = int(min_silence * 1000 / frame_ms)

    for i, speech in enumerate(is_speech):
        if speech:
            if not in_speech:
                start = i
                in_speech = True
            silence_frames = 0
        else:
            if in_speech:
                silence_frames += 1
                if silence_frames >= min_silence_frames:
                    end = i - silence_frames
                    seg_start = start * frame_ms / 1000
                    seg_end = end * frame_ms / 1000
                    if seg_end - seg_start >= min_speech:
                        segments.append((seg_start, seg_end))
                    in_speech = False
                    silence_frames = 0

    # Handle last segment
    if in_speech:
        seg_start = start * frame_ms / 1000
        seg_end = n_frames * frame_ms / 1000
        if seg_end - seg_start >= min_speech:
            segments.append((seg_start, seg_end))

    return segments


def split_long_segments(segments, max_duration=SEGMENT_MAX_DURATION):
    """Split segments longer than max_duration into smaller pieces."""
    result = []
    for start, end in segments:
        duration = end - start
        if duration <= max_duration:
            result.append((start, end))
        else:
            # Split into pieces of ~max_duration
            n_pieces = int(np.ceil(duration / max_duration))
            piece_dur = duration / n_pieces
            for i in range(n_pieces):
                s = start + i * piece_dur
                e = start + (i + 1) * piece_dur
                result.append((s, min(e, end)))
    return result


def spectral_clustering(embeddings, max_speakers=8):
    """Cluster speaker embeddings using spectral clustering with auto num_speakers."""
    from sklearn.cluster import SpectralClustering
    from sklearn.metrics import silhouette_score

    n = len(embeddings)
    if n < 2:
        return np.zeros(n, dtype=int), 1

    # Compute affinity matrix (cosine similarity)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normalized = embeddings / norms
    affinity = normalized @ normalized.T

    # Clip to [0, 1] for spectral clustering
    affinity = np.clip(affinity, 0, 1)
    np.fill_diagonal(affinity, 1.0)

    # Try different number of speakers, pick best silhouette
    best_score = -1
    best_labels = None
    best_k = 2

    max_k = min(max_speakers, n - 1, 8)
    for k in range(2, max_k + 1):
        try:
            sc = SpectralClustering(
                n_clusters=k,
                affinity="precomputed",
                random_state=42,
                n_init=10,
            )
            labels = sc.fit_predict(affinity)
            score = silhouette_score(embeddings, labels, metric="cosine")
            print(f"    k={k}: silhouette={score:.3f}")
            if score > best_score:
                best_score = score
                best_labels = labels
                best_k = k
        except Exception:
            continue

    print(f"    Selected {best_k} speakers (silhouette={best_score:.3f})")
    return best_labels, best_k


def build_profiles(model):
    """Build speaker profiles from voice samples."""
    profiles = {}
    for speaker, sample_files in SAMPLE_MAP.items():
        embeddings = []
        for fname in sample_files:
            fpath = VOICE_SAMPLES_DIR / fname
            if not fpath.exists():
                print(f"  WARNING: {fpath} not found")
                continue
            print(f"  Embedding {speaker}: {fname}")
            waveform, sr = torchaudio.load(str(fpath))
            emb = embed_audio(model, waveform, sr)
            embeddings.append(emb)

        if embeddings:
            avg = np.mean(embeddings, axis=0)
            profiles[speaker] = {
                "embedding": avg.tolist(),
                "num_samples": len(embeddings),
                "dim": len(avg),
            }
            print(f"  {speaker}: {len(embeddings)} samples, {len(avg)}-dim")

    with open(PROFILES_FILE, "w") as f:
        json.dump(profiles, f, indent=2)
    print(f"\nSaved {len(profiles)} profiles to {PROFILES_FILE}")
    return profiles


def load_profiles():
    if PROFILES_FILE.exists():
        with open(PROFILES_FILE) as f:
            return json.load(f)
    return {}


def identify_clusters(cluster_embeddings, profiles):
    """Match cluster average embeddings to known speaker profiles."""
    if not profiles:
        return {}

    profile_embs = {
        name: np.array(p["embedding"])
        for name, p in profiles.items()
    }

    label_map = {}
    used = set()

    # Compute all similarities
    sims = []
    for label, cemb in cluster_embeddings.items():
        for name, pemb in profile_embs.items():
            norm_c = np.linalg.norm(cemb)
            norm_p = np.linalg.norm(pemb)
            if norm_c > 0 and norm_p > 0:
                sim = np.dot(cemb, pemb) / (norm_c * norm_p)
                sims.append((sim, label, name))

    sims.sort(reverse=True)

    for sim, label, name in sims:
        if label in label_map or name in used:
            continue
        if sim > 0.25:
            label_map[label] = name
            used.add(name)
            print(f"    Cluster {label} → {name} (sim={sim:.3f})")

    # Unmatched clusters
    for label in cluster_embeddings:
        if label not in label_map:
            label_map[label] = f"Otro_{label}"
            print(f"    Cluster {label} → Otro_{label}")

    return label_map


def merge_consecutive_segments(segments, gap=MERGE_GAP):
    """Merge consecutive segments with same speaker and small gap."""
    if not segments:
        return segments

    merged = [segments[0].copy()]
    for seg in segments[1:]:
        prev = merged[-1]
        if (seg["speaker"] == prev["speaker"] and
                seg["start"] - prev["end"] <= gap):
            prev["end"] = seg["end"]
        else:
            merged.append(seg.copy())
    return merged


def process_episode(episode_id, model, profiles):
    """Full diarization pipeline for one episode."""
    audio_path = AUDIO_DIR / f"{episode_id}.wav"
    output_path = OUTPUT_DIR / f"{episode_id}.json"

    if not audio_path.exists():
        print(f"  Audio not found: {audio_path}")
        return False

    if output_path.exists():
        print(f"  Already done: {output_path}")
        return True

    print(f"\n{'='*60}")
    print(f"Processing: {episode_id}")
    print(f"{'='*60}")
    t0 = time.time()

    # Load audio
    print("  Loading audio...")
    waveform, sample_rate = torchaudio.load(str(audio_path))
    duration = waveform.shape[1] / sample_rate
    print(f"  Duration: {duration/60:.1f} min, SR: {sample_rate}")

    # VAD
    print("  Running VAD...")
    vad_segments = energy_vad(waveform, sample_rate)
    vad_segments = split_long_segments(vad_segments)
    speech_time = sum(e - s for s, e in vad_segments)
    print(f"  Found {len(vad_segments)} speech segments ({speech_time/60:.1f} min of speech)")

    # Embed each segment
    print(f"  Computing embeddings for {len(vad_segments)} segments...")
    embeddings = []
    valid_segments = []
    for i, (start, end) in enumerate(vad_segments):
        if i % 50 == 0 and i > 0:
            print(f"    Segment {i}/{len(vad_segments)} (elapsed {time.time()-t0:.0f}s)")

        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)
        seg_audio = waveform[:, start_sample:end_sample]

        if seg_audio.shape[1] < sample_rate * 0.5:  # < 0.5s
            continue

        try:
            emb = embed_audio(model, seg_audio, sample_rate)
            if np.any(np.isnan(emb)) or np.linalg.norm(emb) < 1e-6:
                continue
            embeddings.append(emb)
            valid_segments.append((start, end))
        except Exception:
            continue

    print(f"  Got {len(embeddings)} valid embeddings")

    if len(embeddings) < 2:
        print("  Too few embeddings, skipping")
        return False

    emb_array = np.array(embeddings)

    # Free audio memory
    del waveform
    gc.collect()

    # Cluster
    print("  Clustering speakers...")
    labels, n_speakers = spectral_clustering(emb_array)

    # Compute cluster centroids
    cluster_embs = {}
    for label in range(n_speakers):
        mask = labels == label
        if np.any(mask):
            cluster_embs[label] = np.mean(emb_array[mask], axis=0)

    # Identify speakers
    print("  Identifying speakers...")
    label_map = identify_clusters(cluster_embs, profiles)

    # Build output segments
    segments = []
    for (start, end), label in zip(valid_segments, labels):
        speaker = label_map.get(int(label), f"Speaker_{label}")
        segments.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "speaker": speaker,
            "cluster": int(label),
        })

    # Merge consecutive same-speaker segments
    segments = merge_consecutive_segments(segments)

    # Stats
    speaker_times = {}
    for seg in segments:
        spk = seg["speaker"]
        d = seg["end"] - seg["start"]
        speaker_times[spk] = speaker_times.get(spk, 0) + d

    total = sum(speaker_times.values())
    stats = {
        spk: {
            "total_seconds": round(t, 1),
            "percentage": round(100 * t / total, 1) if total > 0 else 0,
        }
        for spk, t in sorted(speaker_times.items(), key=lambda x: -x[1])
    }

    output = {
        "youtube_id": episode_id,
        "duration": round(duration, 1),
        "num_speakers": n_speakers,
        "method": "speechbrain-ecapa-tdnn+spectral-clustering",
        "processing_time_seconds": round(time.time() - t0, 1),
        "speaker_stats": stats,
        "label_map": {str(k): v for k, v in label_map.items()},
        "segments": segments,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed/60:.1f} min")
    for spk, s in stats.items():
        print(f"    {spk}: {s['total_seconds']:.0f}s ({s['percentage']:.1f}%)")
    print(f"  Segments: {len(segments)}")
    print(f"  Output: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", help="Single episode ID")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--build-profiles", action="store_true")
    args = parser.parse_args()

    print("Loading ECAPA-TDNN model...")
    model = load_embedding_model()

    if args.build_profiles:
        build_profiles(model)
        return

    profiles = load_profiles()
    if not profiles:
        print("No profiles found, building...")
        profiles = build_profiles(model)

    if args.episode:
        process_episode(args.episode, model, profiles)
    elif args.all:
        episodes = sorted([f.stem for f in AUDIO_DIR.glob("*.wav")])
        print(f"\n{len(episodes)} episodes to process")
        for i, ep in enumerate(episodes, 1):
            print(f"\n[{i}/{len(episodes)}]")
            process_episode(ep, model, profiles)

        # Upload results to S3
        print("\nUploading results to S3...")
        for f in OUTPUT_DIR.glob("*.json"):
            subprocess.run([
                "/usr/local/bin/aws", "s3", "cp", str(f),
                f"s3://plm-datasets-390402545272/diarization/{f.name}",
                "--quiet",
            ])
        if PROFILES_FILE.exists():
            subprocess.run([
                "/usr/local/bin/aws", "s3", "cp", str(PROFILES_FILE),
                f"s3://plm-datasets-390402545272/diarization/speaker_profiles_ecapa.json",
                "--quiet",
            ])
        print("Uploaded to s3://plm-datasets-390402545272/diarization/")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
