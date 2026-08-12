"""Combines chat, audio, and semantic highlight signals into one ranked list.

Each signal module exposes a dense `score_series(...)` -> {bucket_index:
z-score}, already normalized against that video's own baseline. Fusion here
is a weighted sum of whatever signals are available for a given source (chat
only applies to Twitch; audio and semantic apply to any source), renormalized
so missing signals don't silently zero out a bucket's score. The combined
series is then thresholded into merged HighlightWindows the same way each
individual signal is, via base.merge_scores_into_windows.
"""
from typing import Dict, List, Optional

from .base import HighlightWindow, merge_scores_into_windows

DEFAULT_WEIGHTS = {
    "chat": 0.5,
    "audio": 0.3,
    "semantic": 0.2,
}


def fuse_scores(
    signal_series: Dict[str, Dict[int, float]],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[int, float]:
    """Combines named per-bucket z-score series into one weighted-sum series.

    - signal_series: {"chat": {...}, "audio": {...}, "semantic": {...}},
      any subset of these keys may be present/absent per source availability.
    - weights: per-signal weight; missing signals have their weight excluded
      and the remaining weights renormalized to sum to 1, so e.g. a YouTube
      video with only audio+semantic isn't penalized for lacking chat.
    """
    weights = weights or DEFAULT_WEIGHTS
    active_signals = {name: series for name, series in signal_series.items() if series}
    if not active_signals:
        return {}

    active_weight_total = sum(weights.get(name, 0.0) for name in active_signals) or 1.0
    normalized_weights = {
        name: weights.get(name, 0.0) / active_weight_total for name in active_signals
    }

    all_buckets = set()
    for series in active_signals.values():
        all_buckets.update(series.keys())

    fused = {}
    for bucket in all_buckets:
        fused[bucket] = sum(
            normalized_weights[name] * series.get(bucket, 0.0)
            for name, series in active_signals.items()
        )
    return fused


def fuse_highlights(
    signal_series: Dict[str, Dict[int, float]],
    bucket_seconds: float,
    min_score: float = 1.0,
    merge_gap_buckets: int = 1,
    weights: Optional[Dict[str, float]] = None,
) -> List[HighlightWindow]:
    """Fuses signal series and returns merged, ranked candidate highlight
    windows. `detail` on each window records the per-signal contribution at
    its peak bucket, for debugging/explainability.
    """
    fused = fuse_scores(signal_series, weights)
    if not fused:
        return []

    def detail(run_indices):
        peak_idx = max(run_indices, key=lambda i: fused.get(i, 0.0))
        return {
            name: series.get(peak_idx, 0.0)
            for name, series in signal_series.items() if series
        }

    return merge_scores_into_windows(fused, bucket_seconds, min_score, merge_gap_buckets, detail_fn=detail)
