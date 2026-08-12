"""Shared types for highlight-scoring signals (chat, audio, semantic, fusion).

Each signal module exposes a `score_series(...)` that buckets the video into
fixed-size time windows and returns a per-bucket z-score against that video's
own baseline (dense, one entry per bucket, not just the spikes) so that
fusion.py can combine multiple signals before thresholding into windows.
`merge_scores_into_windows` turns any such dense score series into merged
HighlightWindow candidates on its own, for standalone/single-signal use.
"""
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class HighlightWindow:
    start: float
    end: float
    score: float
    detail: dict = field(default_factory=dict)


def merge_scores_into_windows(
    scores: Dict[int, float],
    bucket_seconds: float,
    min_score: float,
    merge_gap_buckets: int = 1,
    detail_fn: Optional[callable] = None,
) -> List[HighlightWindow]:
    """Merges buckets whose score >= min_score into contiguous HighlightWindows,
    bridging gaps of at most `merge_gap_buckets` non-spiking buckets.

    - scores: {bucket_index: score}, dense or sparse.
    - detail_fn(run_indices) -> dict: optional per-window extra detail payload.
    """
    spiking = sorted(idx for idx, score in scores.items() if score >= min_score)
    if not spiking:
        return []

    windows: List[HighlightWindow] = []
    run_start = spiking[0]
    run_end = spiking[0]

    def flush(start_idx: int, end_idx: int):
        run_indices = list(range(start_idx, end_idx + 1))
        avg_score = statistics.mean(scores.get(i, 0.0) for i in run_indices)
        detail = detail_fn(run_indices) if detail_fn else {}
        windows.append(HighlightWindow(
            start=start_idx * bucket_seconds,
            end=(end_idx + 1) * bucket_seconds,
            score=avg_score,
            detail=detail,
        ))

    for idx in spiking[1:]:
        if idx - run_end <= merge_gap_buckets + 1:
            run_end = idx
        else:
            flush(run_start, run_end)
            run_start = idx
            run_end = idx
    flush(run_start, run_end)

    windows.sort(key=lambda w: w.score, reverse=True)
    return windows


def zscore_series(counts: List[float]) -> Dict[int, float]:
    """Z-scores a list of per-bucket values against their own mean/stdev.
    Returns {} if there's not enough variance to be meaningful (e.g. flat or
    all-zero series), so callers can skip a signal cleanly rather than divide
    by zero.
    """
    if len(counts) < 2:
        return {}
    mean = statistics.mean(counts)
    stdev = statistics.pstdev(counts)
    if stdev == 0:
        return {}
    return {idx: (value - mean) / stdev for idx, value in enumerate(counts)}
