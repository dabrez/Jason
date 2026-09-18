"""Scores a selection run against human labels from the review UI.

This is the regression check: once clips are labeled, any change to the
selection code (ad filtering, shift-window changes, fusion weights, a
different embedder) can be re-scored against the same labels to show whether
it actually helped, rather than relying on re-reading transcripts by hand.

    python3 review/score.py                       # score the current manifest
    python3 review/score.py --baseline before.json  # compare two runs

Precision here means: of the windows this run would ship, how many did you
call good. Recall means: of everything you called good anywhere in the
review set (including near-miss and random samples the scorer did not pick),
how many this run would ship. Recall is the number that catches a scorer
which is confidently shipping mediocre clips while missing the real ones.
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Optional

REVIEW_DIR = "review_clips"
LABELS_PATH = "clip_labels.json"
OVERLAP_THRESHOLD = 0.5


def overlap(a: dict, b: dict) -> float:
    inter = max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    union = max(a["end"], b["end"]) - min(a["start"], b["start"])
    return inter / union if union > 0 else 0.0


def find_label(clip: dict, labels: Dict[str, dict]) -> Optional[dict]:
    """Exact id match first; otherwise the best-overlapping labeled window, so
    a re-cut clip with slightly shifted boundaries still finds its label.
    """
    if clip["id"] in labels:
        return labels[clip["id"]]
    best, best_iou = None, 0.0
    for lab in labels.values():
        if lab.get("start") is None:
            continue
        iou = overlap(clip, lab)
        if iou > best_iou:
            best, best_iou = lab, iou
    return best if best_iou >= OVERLAP_THRESHOLD else None


def score_run(clips: List[dict], labels: Dict[str, dict], shipped_group: str = "selected") -> dict:
    shipped = [c for c in clips if c["group"] == shipped_group]

    judged_shipped = [(c, find_label(c, labels)) for c in shipped]
    judged_shipped = [(c, l) for c, l in judged_shipped if l]

    good_shipped = [c for c, l in judged_shipped if l["verdict"] == "good"]
    bad_shipped = [c for c, l in judged_shipped if l["verdict"] == "bad"]

    all_good = [l for l in labels.values() if l["verdict"] == "good"]
    missed = []
    for lab in all_good:
        if not any(overlap(c, lab) >= OVERLAP_THRESHOLD for c in shipped):
            missed.append(lab)

    precision = len(good_shipped) / len(judged_shipped) if judged_shipped else 0.0
    recall = len(good_shipped) / len(all_good) if all_good else 0.0

    reason_counts: Dict[str, int] = {}
    for _, lab in judged_shipped:
        for r in lab.get("reasons", []):
            reason_counts[r] = reason_counts.get(r, 0) + 1

    return {
        "shipped": len(shipped),
        "judged": len(judged_shipped),
        "good": len(good_shipped),
        "bad": len(bad_shipped),
        "precision": precision,
        "recall": recall,
        "missed_good": missed,
        "reasons": reason_counts,
    }


def fmt_time(s: float) -> str:
    return f"{int(s // 60)}:{int(s % 60):02d}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=os.path.join(REVIEW_DIR, "manifest.json"))
    parser.add_argument("--labels", default=LABELS_PATH)
    parser.add_argument("--baseline", help="A previous manifest to compare against.")
    args = parser.parse_args()

    if not os.path.exists(args.labels):
        raise SystemExit(f"No labels at {args.labels}. Judge some clips first.")
    with open(args.labels, "r", encoding="utf-8") as f:
        labels = json.load(f)
    if not labels:
        raise SystemExit("No labels recorded yet.")

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    verdicts: Dict[str, int] = {}
    for lab in labels.values():
        verdicts[lab["verdict"]] = verdicts.get(lab["verdict"], 0) + 1
    print(f"Labels: {len(labels)} total -- " +
          ", ".join(f"{v} {k}" for k, v in sorted(verdicts.items())))
    print()

    result = score_run(manifest["clips"], labels)
    print(f"=== {os.path.basename(args.manifest)} "
          f"({manifest.get('embedding_model', '?')}) ===")
    print(f"  shipped windows : {result['shipped']} ({result['judged']} of them judged)")
    print(f"  precision       : {result['precision']:.0%} "
          f"({result['good']} good, {result['bad']} bad)")
    print(f"  recall          : {result['recall']:.0%} "
          f"(missed {len(result['missed_good'])} good moments)")

    if result["reasons"]:
        print("  failure reasons :")
        for reason, n in sorted(result["reasons"].items(), key=lambda kv: -kv[1]):
            print(f"      {n:>2}x {reason}")

    if result["missed_good"]:
        print("  good moments NOT shipped:")
        for lab in sorted(result["missed_good"], key=lambda l: l["start"]):
            note = f" -- {lab['note']}" if lab.get("note") else ""
            print(f"      {fmt_time(lab['start'])}-{fmt_time(lab['end'])} "
                  f"[{lab.get('group', '?')}]{note}")

    if args.baseline:
        with open(args.baseline, "r", encoding="utf-8") as f:
            base_manifest = json.load(f)
        base = score_run(base_manifest["clips"], labels)
        print()
        print("=== vs baseline ===")
        for key, label in (("precision", "precision"), ("recall", "recall")):
            delta = result[key] - base[key]
            arrow = "improved" if delta > 0 else ("worse" if delta < 0 else "unchanged")
            print(f"  {label:<10} {base[key]:.0%} -> {result[key]:.0%}  ({arrow})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
