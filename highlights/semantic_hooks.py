"""Highlight detection from transcript "hook" structure.

Local heuristics score each transcript segment for things that tend to
correlate with clip-worthy moments: questions, exclamations, emphatic
language, and sharp topic/sentiment shifts (measured as embedding distance
from the running local average). This needs no external service and is cheap
enough to run on every video.

On top of that, `rank_with_ollama` can optionally send the local shortlist to
a locally-running Ollama model and ask it to re-rank/justify clip-worthiness.
This is off by default: it's a refinement over the heuristic shortlist, not a
replacement, and keeps the pipeline fully local-only (no API keys) while
still usable for testing/iteration per the user's preference for Ollama over
a hosted LLM here.
"""
import json
import re
from typing import Dict, List, Optional

import numpy as np
import requests

from .base import HighlightWindow, merge_scores_into_windows, zscore_series

DEFAULT_BUCKET_SECONDS = 20.0
OLLAMA_URL = "http://localhost:11434/api/generate"

_QUESTION_RE = re.compile(r"\?")
_EXCLAMATION_RE = re.compile(r"!")
_EMPHASIS_WORDS = {
    "insane", "crazy", "unbelievable", "wow", "what", "no way", "oh my god",
    "wait", "actually", "literally", "seriously", "huge", "amazing", "wild",
}


def _heuristic_hook_score(text: str) -> float:
    lower = text.lower()
    score = 0.0
    score += 1.0 * len(_QUESTION_RE.findall(text))
    score += 1.0 * len(_EXCLAMATION_RE.findall(text))
    score += sum(0.5 for word in _EMPHASIS_WORDS if word in lower)
    return score


def _embedding_shift_scores(embeddings: np.ndarray) -> List[float]:
    """Cosine distance of each segment's embedding from a running average of
    prior segments -- large shifts suggest a topic/tone swing, which can be a
    hook signal (e.g. a punchline or a sudden turn in the conversation).
    """
    if len(embeddings) < 2:
        return [0.0] * len(embeddings)

    shifts = [0.0]
    running_sum = embeddings[0].copy()
    running_count = 1
    for i in range(1, len(embeddings)):
        running_avg = running_sum / running_count
        denom = (np.linalg.norm(running_avg) * np.linalg.norm(embeddings[i])) or 1e-9
        cosine_sim = float(np.dot(running_avg, embeddings[i]) / denom)
        shifts.append(1.0 - cosine_sim)
        running_sum += embeddings[i]
        running_count += 1
    return shifts


def score_series(
    transcript: List[dict],
    embeddings: Optional[np.ndarray] = None,
    bucket_seconds: float = DEFAULT_BUCKET_SECONDS,
) -> Dict[int, float]:
    """Returns {bucket_index: z-score} for transcript "hook" strength across
    the video. `embeddings` (one per transcript segment, e.g. from
    sentence-transformers) is optional -- without it, scoring falls back to
    question/exclamation/emphasis heuristics only.
    """
    if not transcript:
        return {}

    max_end = max(seg["start"] + seg["duration"] for seg in transcript)
    num_buckets = int(max_end // bucket_seconds) + 1
    bucket_scores = [0.0] * num_buckets

    shifts = _embedding_shift_scores(embeddings) if embeddings is not None else None

    for idx, seg in enumerate(transcript):
        hook = _heuristic_hook_score(seg["text"])
        if shifts is not None:
            hook += 3.0 * shifts[idx]
        bucket_idx = int(seg["start"] // bucket_seconds)
        bucket_scores[bucket_idx] += hook

    return zscore_series(bucket_scores)


def score_semantic_hooks(
    transcript: List[dict],
    embeddings: Optional[np.ndarray] = None,
    bucket_seconds: float = DEFAULT_BUCKET_SECONDS,
    min_score: float = 1.5,
    merge_gap_buckets: int = 1,
) -> List[HighlightWindow]:
    """Scores time buckets by transcript hook strength and returns merged
    candidate highlight windows sorted by descending score.
    """
    scores = score_series(transcript, embeddings, bucket_seconds)
    if not scores:
        return []
    return merge_scores_into_windows(scores, bucket_seconds, min_score, merge_gap_buckets)


def rank_with_ollama(
    windows: List[HighlightWindow],
    transcript_text_by_window: List[str],
    model: str = "gpt-oss:latest",
    ollama_url: str = OLLAMA_URL,
    timeout: float = 90.0,
) -> List[HighlightWindow]:
    """Re-ranks a shortlist of candidate windows using a locally-running
    Ollama model, asking it to judge clip-worthiness from the transcript
    text. Returns a new list of HighlightWindows with `score` replaced by the
    model's rating (1-10) and the rationale stored in `detail["rationale"]`.

    Falls back to the original windows, unmodified, if Ollama isn't reachable
    or returns something unparseable -- this is a refinement pass, not a hard
    dependency, so a local Ollama outage shouldn't break clip selection.
    """
    if not windows:
        return windows

    prompt = (
        "You are helping a video editor pick the best short clips from a "
        "longer video for social media. For each numbered transcript "
        "excerpt below, rate how likely it is to make a compelling "
        "standalone short clip, from 1 (boring) to 10 (must-clip). Respond "
        "with ONLY a JSON array of objects like "
        '[{"index": 0, "rating": 7, "reason": "..."}], one entry per excerpt, '
        "no other text.\n\n"
    )
    for i, text in enumerate(transcript_text_by_window):
        prompt += f"{i}. {text.strip()[:500]}\n\n"

    try:
        response = requests.post(
            ollama_url,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        response.raise_for_status()
        raw = response.json().get("response", "")
        ratings = json.loads(_extract_json_array(raw))
    except (requests.RequestException, ValueError, KeyError, json.JSONDecodeError):
        return windows

    ranked = []
    for entry in ratings:
        idx = entry.get("index")
        if idx is None or not (0 <= idx < len(windows)):
            continue
        original = windows[idx]
        ranked.append(HighlightWindow(
            start=original.start,
            end=original.end,
            score=float(entry.get("rating", original.score)),
            detail={**original.detail, "rationale": entry.get("reason", "")},
        ))

    if not ranked:
        return windows

    ranked.sort(key=lambda w: w.score, reverse=True)
    return ranked


def _extract_json_array(text: str) -> str:
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON array found in Ollama response")
    return text[start:end + 1]
