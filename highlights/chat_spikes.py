"""Highlight detection from Twitch chat activity.

Idea: viewers chat more (and use more emotes) during exciting/funny moments.
Bucket the VOD's chat replay into fixed-size time windows and z-score each
bucket's message rate against the VOD's own baseline. This is a cheap,
first-pass signal meant to be combined with audio-energy and transcript-based
scoring (see fusion.py and roadmap.txt, Phase 2).
"""
from typing import Dict, List

from .base import HighlightWindow, merge_scores_into_windows, zscore_series

DEFAULT_BUCKET_SECONDS = 20.0


def _comment_offset(comment: dict) -> float:
    """TwitchDownloaderCLI chat JSON uses `content_offset_seconds`; fall back
    to `contentOffsetSeconds` for older/alternate export formats.
    """
    if "content_offset_seconds" in comment:
        return float(comment["content_offset_seconds"])
    return float(comment["contentOffsetSeconds"])


def _is_emote_heavy(comment: dict) -> bool:
    fragments = comment.get("message", {}).get("fragments", [])
    if not fragments:
        return False
    return any(frag.get("emoticon") for frag in fragments)


def _bucket_comments(comments: List[dict], bucket_seconds: float) -> dict:
    buckets = {}
    for comment in comments:
        try:
            offset = _comment_offset(comment)
        except (KeyError, TypeError, ValueError):
            continue
        bucket_idx = int(offset // bucket_seconds)
        bucket = buckets.setdefault(bucket_idx, {"count": 0, "emotes": 0})
        bucket["count"] += 1
        if _is_emote_heavy(comment):
            bucket["emotes"] += 1
    return buckets


def score_series(comments: List[dict], bucket_seconds: float = DEFAULT_BUCKET_SECONDS) -> Dict[int, float]:
    """Returns {bucket_index: z-score} for chat message rate across the whole
    VOD (dense: every bucket up to the last comment, zero-filled if empty).
    """
    if not comments:
        return {}
    buckets = _bucket_comments(comments, bucket_seconds)
    if len(buckets) < 2:
        return {}
    max_bucket_idx = max(buckets.keys())
    counts = [buckets.get(i, {"count": 0})["count"] for i in range(max_bucket_idx + 1)]
    return zscore_series(counts)


def score_chat_spikes(
    comments: List[dict],
    bucket_seconds: float = DEFAULT_BUCKET_SECONDS,
    min_score: float = 2.0,
    merge_gap_buckets: int = 1,
) -> List[HighlightWindow]:
    """Scores time buckets by chat-rate z-score and returns merged candidate
    highlight windows sorted by descending score.

    - bucket_seconds: granularity of the analysis window.
    - min_score: minimum z-score for a bucket to be considered a spike.
    - merge_gap_buckets: merge spiking buckets separated by at most this many
      non-spiking buckets, so one highlight moment doesn't get split into
      several tiny clips.
    """
    scores = score_series(comments, bucket_seconds)
    if not scores:
        return []

    buckets = _bucket_comments(comments, bucket_seconds)

    def detail(run_indices):
        total_messages = sum(buckets.get(i, {"count": 0})["count"] for i in run_indices)
        peak_rate = max(buckets.get(i, {"count": 0})["count"] / bucket_seconds for i in run_indices)
        return {"message_count": total_messages, "peak_messages_per_sec": peak_rate}

    return merge_scores_into_windows(scores, bucket_seconds, min_score, merge_gap_buckets, detail_fn=detail)
