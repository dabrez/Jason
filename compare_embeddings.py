"""Side-by-side comparison of embedding backends for highlight selection.

The embedding model reaches clip selection through exactly one path:
pipeline._compute_embeddings -> highlights.semantic_hooks.score_series ->
_embedding_shift_scores. Audio and chat scoring are embedder-independent, so
isolating the semantic signal is what actually measures the embedder -- and
it means this runs off a cached transcript alone, with no source video and
no Whisper pass.

Usage:
    python compare_embeddings.py                       # all default models
    python compare_embeddings.py --models a b          # specific models
    python compare_embeddings.py --json out.json       # machine-readable too

Each model produces a ranked list of highlight windows; the report shows
which windows the models agree on (by time overlap), which are unique to one
model, and the transcript text for divergent picks so the choice can be
judged on content rather than on scores alone.
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

from embeddings import get_embedder
from highlights.base import HighlightWindow
from highlights.semantic_hooks import DEFAULT_BUCKET_SECONDS, score_semantic_hooks

TRANSCRIPT_CACHE_DIR = "transcript_cache"

DEFAULT_MODELS = [
    "all-MiniLM-L6-v2",
    "ollama:qwen3-embedding",
    "ollama:nomic-embed-text",
]

# Two windows count as "the same pick" when they overlap by at least this
# fraction of their union (IoU). Highlight boundaries get snapped to
# transcript segments downstream, so near-identical picks rarely match to the
# second -- this tolerates that without merging genuinely different moments.
OVERLAP_THRESHOLD = 0.5


# -- transcript loading ---------------------------------------------------

def load_cached_transcript(path: Optional[str] = None) -> Tuple[List[dict], str]:
    """Loads a transcript from the pipeline's on-disk cache. With no path,
    picks the largest cached transcript (the most substantial video cached).
    """
    if path:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), path

    if not os.path.isdir(TRANSCRIPT_CACHE_DIR):
        raise SystemExit(
            f"No {TRANSCRIPT_CACHE_DIR}/ directory. Run the pipeline once to "
            "populate the transcript cache, or pass --transcript."
        )
    candidates = [
        os.path.join(TRANSCRIPT_CACHE_DIR, name)
        for name in os.listdir(TRANSCRIPT_CACHE_DIR)
        if name.endswith(".json")
    ]
    if not candidates:
        raise SystemExit(f"No cached transcripts in {TRANSCRIPT_CACHE_DIR}/.")

    chosen = max(candidates, key=os.path.getsize)
    with open(chosen, "r", encoding="utf-8") as f:
        return json.load(f), chosen


def preprocess_texts(transcript: List[dict]) -> List[str]:
    """Mirrors ClipPipeline._preprocess_text so embeddings are computed over
    exactly the same strings the real pipeline would feed them.
    """
    import re
    return [re.sub(r"[\W_]+", " ", entry["text"]).lower() for entry in transcript]


# -- per-model run --------------------------------------------------------

def run_model(
    model_name: str,
    transcript: List[dict],
    texts: List[str],
    bucket_seconds: float,
    top_k: int,
) -> dict:
    """Embeds the transcript with one backend and returns its ranked
    highlight windows plus timing//shift diagnostics.
    """
    import time

    started = time.time()
    embedder = get_embedder(model_name)
    embeddings = np.asarray(embedder.encode(texts))
    encode_seconds = time.time() - started

    windows = score_semantic_hooks(
        transcript, embeddings, bucket_seconds=bucket_seconds
    )
    windows = windows[:top_k]

    # Diagnostics on the shift signal itself -- a signal that is flat or
    # saturated produces uninformative rankings regardless of clip content.
    from highlights.semantic_hooks import _embedding_shift_scores
    shifts = np.asarray(_embedding_shift_scores(embeddings))

    return {
        "model": model_name,
        "dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        "encode_seconds": encode_seconds,
        "windows": windows,
        "shift_mean": float(shifts.mean()),
        "shift_std": float(shifts.std()),
        "shift_p95": float(np.percentile(shifts, 95)),
    }


# -- window comparison ----------------------------------------------------

def overlap_ratio(a: HighlightWindow, b: HighlightWindow) -> float:
    """Intersection-over-union of two time windows."""
    intersection = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    return intersection / union if union > 0 else 0.0


def match_windows(
    left: List[HighlightWindow], right: List[HighlightWindow]
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """Greedily pairs windows between two models by best overlap.

    Returns (matched pairs as (left_idx, right_idx, iou), left-only indices,
    right-only indices). Greedy-by-best-overlap is sufficient here because
    highlight windows within one model are non-overlapping, so there is no
    ambiguity about which pairing is best.
    """
    pairs = sorted(
        (
            (overlap_ratio(l, r), i, j)
            for i, l in enumerate(left)
            for j, r in enumerate(right)
        ),
        reverse=True,
    )

    matched: List[Tuple[int, int, float]] = []
    used_left, used_right = set(), set()
    for iou, i, j in pairs:
        if iou < OVERLAP_THRESHOLD:
            break
        if i in used_left or j in used_right:
            continue
        matched.append((i, j, iou))
        used_left.add(i)
        used_right.add(j)

    left_only = [i for i in range(len(left)) if i not in used_left]
    right_only = [j for j in range(len(right)) if j not in used_right]
    return matched, left_only, right_only


def text_between(transcript: List[dict], start: float, end: float) -> str:
    """Mirrors ClipPipeline._text_between."""
    return " ".join(
        seg["text"] for seg in transcript
        if seg["start"] >= start and seg["start"] < end
    )


def fmt_time(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes:d}:{secs:02d}"


def fmt_window(w: HighlightWindow) -> str:
    return f"{fmt_time(w.start)}-{fmt_time(w.end)} (z={w.score:.2f})"


# -- reporting ------------------------------------------------------------

def print_report(
    results: List[dict],
    transcript: List[dict],
    excerpt_chars: int,
) -> None:
    print("=" * 78)
    print("EMBEDDING COMPARISON -- semantic highlight signal")
    print("=" * 78)
    total_min = max(s["start"] + s["duration"] for s in transcript) / 60
    print(f"Transcript: {len(transcript)} segments, {total_min:.1f} min")
    print()

    print("Per-model summary")
    print("-" * 78)
    header = f"{'model':<28} {'dim':>5} {'encode':>8} {'windows':>8} {'shift mean/std':>16}"
    print(header)
    for r in results:
        print(
            f"{r['model']:<28} {r['dim']:>5} {r['encode_seconds']:>7.1f}s "
            f"{len(r['windows']):>8} "
            f"{r['shift_mean']:>7.3f}/{r['shift_std']:.3f}"
        )
    print()

    baseline = results[0]
    for other in results[1:]:
        print("=" * 78)
        print(f"{baseline['model']}  vs  {other['model']}")
        print("-" * 78)

        matched, left_only, right_only = match_windows(
            baseline["windows"], other["windows"]
        )

        total = len(baseline["windows"]) + len(other["windows"])
        agree_pct = (2 * len(matched) / total * 100) if total else 0.0
        print(
            f"Agreement: {len(matched)} shared picks "
            f"({agree_pct:.0f}% of all windows)   "
            f"unique to {baseline['model']}: {len(left_only)}   "
            f"unique to {other['model']}: {len(right_only)}"
        )
        print()

        if matched:
            print("SHARED PICKS (both models found these)")
            for i, j, iou in sorted(matched, key=lambda m: baseline["windows"][m[0]].start):
                lw, rw = baseline["windows"][i], other["windows"][j]
                print(
                    f"  {fmt_window(lw):<26} | {fmt_window(rw):<26} "
                    f"| rank {i + 1} vs {j + 1}, IoU {iou:.2f}"
                )
            print()

        for label, model_result, indices in (
            (f"ONLY {baseline['model']}", baseline, left_only),
            (f"ONLY {other['model']}", other, right_only),
        ):
            if not indices:
                continue
            print(f"{label} -- these are the picks to judge")
            for idx in sorted(indices, key=lambda k: model_result["windows"][k].start):
                w = model_result["windows"][idx]
                excerpt = text_between(transcript, w.start, w.end).strip()
                excerpt = " ".join(excerpt.split())[:excerpt_chars]
                print(f"  [rank {idx + 1}] {fmt_window(w)}")
                print(f"      {excerpt}...")
                print()
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help="Embedder names; first is treated as the baseline.")
    parser.add_argument("--transcript", help="Path to a cached transcript JSON.")
    parser.add_argument("--bucket-seconds", type=float, default=DEFAULT_BUCKET_SECONDS)
    parser.add_argument("--top-k", type=int, default=15,
                        help="Windows to keep per model (default 15).")
    parser.add_argument("--excerpt-chars", type=int, default=320)
    parser.add_argument("--json", help="Also write full results to this JSON path.")
    args = parser.parse_args()

    transcript, source = load_cached_transcript(args.transcript)
    texts = preprocess_texts(transcript)
    print(f"Using cached transcript: {source}\n")

    results = []
    for model in args.models:
        print(f"Encoding with {model} ...", flush=True)
        try:
            results.append(
                run_model(model, transcript, texts, args.bucket_seconds, args.top_k)
            )
        except Exception as exc:  # noqa: BLE001 - one bad backend shouldn't abort the rest
            print(f"  SKIPPED {model}: {type(exc).__name__}: {exc}\n", flush=True)
    print()

    if len(results) < 2:
        print("Need at least two working models to compare.")
        return 1

    print_report(results, transcript, args.excerpt_chars)

    if args.json:
        payload = [
            {
                **{k: v for k, v in r.items() if k != "windows"},
                "windows": [
                    {
                        "start": w.start,
                        "end": w.end,
                        "score": w.score,
                        "text": text_between(transcript, w.start, w.end),
                    }
                    for w in r["windows"]
                ],
            }
            for r in results
        ]
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
