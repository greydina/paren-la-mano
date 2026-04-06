#!/usr/bin/env python3
"""Run pyannote speaker diarization 3.1 on the 20-minute sample."""
import warnings
warnings.filterwarnings("ignore")
import json, time, gc, os, sys

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)  # line-buffered

print("Step 1: Loading pipeline...")
from pyannote.audio import Pipeline
import torch

HF_TOKEN = os.environ.get("HF_TOKEN", "")
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=HF_TOKEN)
pipeline.to(torch.device("cpu"))
pipeline._segmentation.batch_size = 1
print("Pipeline loaded (batch_size=1)")

print("Step 2: Loading audio...")
import soundfile as sf
AUDIO_FILE = "/home/ubuntu/paren-la-mano/data/audio/SecttWanihg_sample20m.wav"
data, sample_rate = sf.read(AUDIO_FILE, dtype='float32')

if data.ndim == 1:
    waveform = torch.from_numpy(data).unsqueeze(0)
else:
    waveform = torch.from_numpy(data.T)
del data
gc.collect()

audio_input = {"waveform": waveform, "sample_rate": sample_rate}
print(f"Audio: {waveform.shape}, sr={sample_rate}")
os.system("free -m")

print("Step 3: Running diarization...")
start = time.time()
diarization = pipeline(audio_input)
elapsed = time.time() - start
print(f"Diarization complete in {elapsed:.1f}s")

del pipeline, waveform, audio_input
gc.collect()

# Extract results - pyannote 4.0 returns DiarizeOutput dataclass
if hasattr(diarization, 'speaker_diarization'):
    annotation = diarization.speaker_diarization
    exclusive = diarization.exclusive_speaker_diarization
else:
    annotation = diarization
    exclusive = diarization

# Save RTTM immediately as backup
RTTM_PATH = "/home/ubuntu/paren-la-mano/data/diarization/SecttWanihg_sample20m_pyannote.rttm"
with open(RTTM_PATH, "w") as f:
    annotation.write_rttm(f)
print(f"RTTM saved to {RTTM_PATH}")

segments = []
speakers = set()
for turn, _, speaker in annotation.itertracks(yield_label=True):
    segments.append({
        "start": round(turn.start, 3),
        "end": round(turn.end, 3),
        "speaker": speaker
    })
    speakers.add(speaker)

# Detect overlaps by finding time regions with 2+ speakers
from pyannote.core import Timeline
overlap_timeline = Timeline()
# Use get_overlap method if available, else compute manually
if hasattr(annotation, 'get_overlap'):
    overlap_timeline = annotation.get_overlap()
else:
    # Build overlap from segments that share time
    from itertools import combinations
    tracks_list = list(annotation.itertracks(yield_label=True))
    for i, (s1, _, l1) in enumerate(tracks_list):
        for j, (s2, _, l2) in enumerate(tracks_list):
            if j <= i:
                continue
            if l1 == l2:
                continue
            intersection = s1 & s2
            if intersection:
                overlap_timeline.add(intersection)
    overlap_timeline = overlap_timeline.support()

overlaps = []
for segment in overlap_timeline:
    overlaps.append({
        "start": round(segment.start, 3),
        "end": round(segment.end, 3)
    })

from collections import Counter
speaker_dur = Counter()
for s in segments:
    speaker_dur[s["speaker"]] += s["end"] - s["start"]

result = {
    "source": "pyannote/speaker-diarization-3.1",
    "audio_file": AUDIO_FILE,
    "processing_time_seconds": round(elapsed, 1),
    "num_speakers": len(speakers),
    "speakers": sorted(list(speakers)),
    "num_segments": len(segments),
    "num_overlaps": len(overlaps),
    "total_overlap_duration": round(sum(o["end"] - o["start"] for o in overlaps), 3),
    "segments": segments,
    "overlaps": overlaps
}

OUT_PATH = "/home/ubuntu/paren-la-mano/data/diarization/SecttWanihg_sample20m_pyannote.json"
with open(OUT_PATH, "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"\nSaved to {OUT_PATH}")
print(f"Speakers: {len(speakers)} - {sorted(list(speakers))}")
print(f"Segments: {len(segments)}")
print(f"Overlaps: {len(overlaps)} ({result['total_overlap_duration']}s)")
total_dur = sum(speaker_dur.values())
print("\nSpeaker distribution:")
for spk, dur in sorted(speaker_dur.items()):
    print(f"  {spk}: {dur:.1f}s ({dur/total_dur*100:.1f}%)")
