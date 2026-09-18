"""Builds a review set: candidate clips cut from a source video, ready to be
judged in the browser (see review/server.py).

The point of the review set is to measure clip-selection quality against
human judgment, so it deliberately includes more than just the winners:

- `selected`: what the pipeline would actually ship (top-ranked fused
  windows). These measure precision -- how many shipped clips are good.
- `nearmiss`: windows that scored well but fell below the selection cutoff.
  These show what a slightly looser threshold would let through.
- `random`: uniformly sampled windows from anywhere in the video, scored or
  not. These are the only way to catch recall failures -- a great moment the
  scorer never surfaced will never appear in `selected` or `nearmiss`, so
  without random samples the blind spot stays invisible by construction.

Labels are keyed by (video_id, start, end) rounded to the second, so a clip
re-cut by a later run is still recognizably the same pick and keeps its
label.
"""
import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embeddings import get_embedder  # noqa: E402
from highlights import audio_score_series, fuse_highlights, semantic_score_series  # noqa: E402
from highlights.base import HighlightWindow  # noqa: E402
from highlights.chat_spikes import DEFAULT_BUCKET_SECONDS  # noqa: E402
from pipeline import MAX_CLIP_SECONDS, ClipPipeline  # noqa: E402

REVIEW_DIR = "review_clips"
DEFAULT_TRANSCRIPT = "transcript_cache/ee0ebfd311aaf9f8.json"


def clip_key(video_id: str, start: float, end: float) -> str:
    return f"{video_id}:{int(round(start))}-{int(round(end))}"


def load_transcript(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def preprocess_texts(transcript: List[dict]) -> List[str]:
    import re
    return [re.sub(r"[\W_]+", " ", e["text"]).lower() for e in transcript]


def text_between(transcript: List[dict], start: float, end: float) -> str:
    return " ".join(
        seg["text"] for seg in transcript
        if seg["start"] >= start and seg["start"] < end
    ).strip()


def make_pipeline(transcript: List[dict]) -> "ClipPipeline":
    """A ClipPipeline bound to a transcript only, so the review set is cut
    with the real boundary/cap logic rather than a copy of it -- a second
    implementation here would drift from pipeline.py and the review set
    would stop reflecting what actually ships.
    """
    p = ClipPipeline.__new__(ClipPipeline)
    p.transcript = transcript
    return p


def build_windows(
    transcript: List[dict],
    video_path: str,
    embedding_model: str,
    bucket_seconds: float,
    num_selected: int,
    num_nearmiss: int,
    num_random: int,
    seed: int,
) -> List[dict]:
    """Scores the video and assembles the three candidate groups."""
    texts = preprocess_texts(transcript)
    print(f"  embedding with {embedding_model} ...", flush=True)
    embeddings = np.asarray(get_embedder(embedding_model).encode(texts))

    print("  scoring audio ...", flush=True)
    audio = audio_score_series(video_path, bucket_seconds)
    print("  scoring semantic ...", flush=True)
    semantic = semantic_score_series(transcript, embeddings, bucket_seconds)

    signals = {"chat": {}, "audio": audio, "semantic": semantic}
    windows = fuse_highlights(signals, bucket_seconds)
    print(f"  fused -> {len(windows)} candidate windows", flush=True)

    pipe = make_pipeline(transcript)
    sentence_starts, sentence_ends = pipe.sentence_boundaries()
    duration = max(s["start"] + s["duration"] for s in transcript)

    def to_record(w: HighlightWindow, group: str) -> dict:
        start = pipe._snap_back(max(0.0, w.start - 5.0), sentence_starts)
        end = pipe._snap_forward(w.end + 5.0, sentence_ends)
        if end - start > MAX_CLIP_SECONDS:
            start, end = pipe._trim_to_max(
                start, end, MAX_CLIP_SECONDS, sentence_starts, sentence_ends
            )
        return {
            "group": group,
            "start": start,
            "end": end,
            "score": float(w.score),
            "signals": {k: float(v) for k, v in (w.detail or {}).items()
                        if isinstance(v, (int, float))},
            "text": text_between(transcript, start, end),
        }

    records = [to_record(w, "selected") for w in windows[:num_selected]]
    records += [to_record(w, "nearmiss")
                for w in windows[num_selected:num_selected + num_nearmiss]]

    # Random windows: uniformly sampled starts, avoiding overlap with anything
    # already chosen so the reviewer isn't shown the same moment twice.
    rng = random.Random(seed)
    taken = [(r["start"], r["end"]) for r in records]
    attempts = 0
    while len([r for r in records if r["group"] == "random"]) < num_random and attempts < num_random * 60:
        attempts += 1
        length = rng.uniform(25.0, 70.0)
        start = rng.uniform(0.0, max(1.0, duration - length))
        end = start + length
        if any(start < te and end > ts for ts, te in taken):
            continue
        start = pipe._snap_back(start, sentence_starts)
        end = pipe._snap_forward(end, sentence_ends)
        if end - start < 10.0:
            continue
        body = text_between(transcript, start, end)
        if not body:
            continue
        taken.append((start, end))
        records.append({
            "group": "random", "start": start, "end": end,
            "score": 0.0, "signals": {}, "text": body,
        })

    records.sort(key=lambda r: r["start"])
    return records


def cut_clips(records: List[dict], video_path: str, video_id: str, out_dir: str) -> None:
    """Cuts each record to an mp4. Re-encodes rather than stream-copying:
    `-c copy` snaps to the nearest keyframe, which can shift a clip's real
    start by seconds and would mean judging a different moment than the one
    being scored.
    """
    os.makedirs(out_dir, exist_ok=True)
    for i, rec in enumerate(records, 1):
        rec["id"] = clip_key(video_id, rec["start"], rec["end"])
        filename = f"{rec['id'].replace(':', '_')}.mp4"
        path = os.path.join(out_dir, filename)
        rec["file"] = filename
        if os.path.exists(path) and os.path.getsize(path) > 0:
            print(f"  [{i}/{len(records)}] cached {filename}", flush=True)
            continue
        print(f"  [{i}/{len(records)}] cutting {filename}", flush=True)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-ss", str(rec["start"]), "-to", str(rec["end"]),
             "-i", video_path,
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
             "-c:a", "aac", "-movflags", "+faststart", path],
            check=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Path to the source video.")
    parser.add_argument("--video-id", help="Stable id for label keys (default: filename stem).")
    parser.add_argument("--transcript", default=DEFAULT_TRANSCRIPT)
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--bucket-seconds", type=float, default=DEFAULT_BUCKET_SECONDS)
    parser.add_argument("--selected", type=int, default=12)
    parser.add_argument("--nearmiss", type=int, default=6)
    parser.add_argument("--random", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=REVIEW_DIR)
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required.")
    if not os.path.exists(args.video):
        raise SystemExit(f"No such video: {args.video}")

    video_id = args.video_id or os.path.splitext(os.path.basename(args.video))[0]
    transcript = load_transcript(args.transcript)
    print(f"Transcript: {len(transcript)} segments")

    records = build_windows(
        transcript, args.video, args.embedding_model, args.bucket_seconds,
        args.selected, args.nearmiss, args.random, args.seed,
    )
    counts: Dict[str, int] = {}
    for r in records:
        counts[r["group"]] = counts.get(r["group"], 0) + 1
    print(f"Review set: {counts}")

    cut_clips(records, args.video, video_id, args.out)

    manifest = {
        "video_id": video_id,
        "video": os.path.abspath(args.video),
        "embedding_model": args.embedding_model,
        "bucket_seconds": args.bucket_seconds,
        "clips": records,
    }
    manifest_path = os.path.join(args.out, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {manifest_path} ({len(records)} clips)")
    print("Now run:  python3 review/server.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
