"""Highlight detection from audio energy.

Idea: loud/energetic moments (laughter, shouting, excited reactions, big
sound effects) tend to correlate with clip-worthy moments, and unlike chat
this works for any source -- YouTube, Twitch, plain podcasts with no chat at
all. We extract mono audio via ffmpeg, compute short-time RMS energy per
fixed-size window with librosa, and z-score each window against the video's
own baseline, same pattern as chat_spikes.py.
"""
import shutil
import subprocess
import tempfile
from typing import Dict, List

import librosa
import numpy as np

from .base import HighlightWindow, merge_scores_into_windows, zscore_series

DEFAULT_BUCKET_SECONDS = 20.0
SAMPLE_RATE = 22050


def _extract_audio(video_path: str) -> str:
    if shutil.which("ffmpeg") is None:
        raise EnvironmentError(
            "ffmpeg is required to extract audio for highlight scoring. "
            "Install it from https://ffmpeg.org/ and ensure it is on your PATH."
        )
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        audio_path = tmp.name
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
        audio_path,
    ], check=True, capture_output=True)
    return audio_path


def _bucket_rms(video_path: str, bucket_seconds: float) -> List[float]:
    audio_path = _extract_audio(video_path)
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

    hop_length = int(bucket_seconds * sr)
    if hop_length <= 0 or len(y) == 0:
        return []

    rms = librosa.feature.rms(y=y, frame_length=hop_length, hop_length=hop_length)[0]
    return [float(v) for v in rms]


def score_series(video_path: str, bucket_seconds: float = DEFAULT_BUCKET_SECONDS) -> Dict[int, float]:
    """Returns {bucket_index: z-score} for audio RMS energy across the video."""
    counts = _bucket_rms(video_path, bucket_seconds)
    return zscore_series(counts)


def score_audio_spikes(
    video_path: str,
    bucket_seconds: float = DEFAULT_BUCKET_SECONDS,
    min_score: float = 2.0,
    merge_gap_buckets: int = 1,
) -> List[HighlightWindow]:
    """Scores time buckets by audio-energy z-score and returns merged
    candidate highlight windows sorted by descending score.
    """
    rms_values = _bucket_rms(video_path, bucket_seconds)
    scores = zscore_series(rms_values)
    if not scores:
        return []

    def detail(run_indices):
        peak_rms = max(rms_values[i] for i in run_indices)
        return {"peak_rms": peak_rms}

    return merge_scores_into_windows(scores, bucket_seconds, min_score, merge_gap_buckets, detail_fn=detail)
