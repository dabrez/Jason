"""Tests for sentence-boundary clip alignment (pipeline.ClipPipeline).

Covers the behavior that keeps clips from starting or ending mid-thought,
plus the degenerate cases real transcripts actually produce -- notably
stretches where Whisper emits no sentence punctuation at all, which is what
broke the first version of this code (a 769-second "clip").

    python3 -m pytest tests/ -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import MAX_SNAP_SHIFT, ClipPipeline  # noqa: E402


def make_pipeline(transcript):
    p = ClipPipeline.__new__(ClipPipeline)
    p.transcript = transcript
    return p


def seg(text, start, duration, words=None):
    return {"text": text, "start": start, "duration": duration, "words": words or []}


def words_from(pairs):
    """[(word, start, end), ...] -> whisper-style word dicts."""
    return [{"word": w, "start": s, "end": e} for w, s, e in pairs]


# -- sentence_boundaries -------------------------------------------------

def test_sentence_boundaries_splits_on_punctuation():
    transcript = [seg("Hello there. How are you?", 0.0, 4.0, words_from([
        ("Hello", 0.0, 0.5), ("there.", 0.5, 1.0),
        ("How", 1.2, 1.5), ("are", 1.5, 1.8), ("you?", 1.8, 2.2),
    ]))]
    starts, ends = make_pipeline(transcript).sentence_boundaries()
    assert starts == [0.0, 1.2]
    assert ends == [1.0, 2.2]


def test_sentence_boundaries_handles_trailing_quote():
    transcript = [seg('He said "stop." Then left.', 0.0, 3.0, words_from([
        ("He", 0.0, 0.2), ("said", 0.2, 0.5), ('"stop."', 0.5, 1.0),
        ("Then", 1.1, 1.4), ("left.", 1.4, 1.8),
    ]))]
    starts, ends = make_pipeline(transcript).sentence_boundaries()
    assert starts == [0.0, 1.1]
    assert ends == [1.0, 1.8]


def test_sentence_boundaries_closes_unterminated_tail():
    """Trailing words with no final punctuation still yield an end, so the
    last sentence is reachable rather than silently dropped."""
    transcript = [seg("and then we", 0.0, 2.0, words_from([
        ("and", 0.0, 0.3), ("then", 0.3, 0.6), ("we", 0.6, 1.0),
    ]))]
    starts, ends = make_pipeline(transcript).sentence_boundaries()
    assert starts == [0.0]
    assert ends == [1.0]


def test_sentence_boundaries_falls_back_to_segments_without_words():
    """Older cached transcripts predate word_timestamps=True."""
    transcript = [seg("no words here", 5.0, 3.0), seg("nor here", 10.0, 2.0)]
    starts, ends = make_pipeline(transcript).sentence_boundaries()
    assert starts == [5.0, 10.0]
    assert ends == [8.0, 12.0]


# -- directional snapping ------------------------------------------------

def test_snap_back_never_moves_forward():
    boundaries = [0.0, 10.0, 20.0]
    # 19.0 is nearest to 20.0, but snapping forward would cut into speech.
    assert ClipPipeline._snap_back(19.0, boundaries) == 10.0


def test_snap_forward_never_moves_backward():
    boundaries = [0.0, 10.0, 20.0]
    assert ClipPipeline._snap_forward(11.0, boundaries) == 20.0


def test_snap_exact_boundary_is_unchanged():
    boundaries = [0.0, 10.0, 20.0]
    assert ClipPipeline._snap_back(10.0, boundaries) == 10.0
    assert ClipPipeline._snap_forward(10.0, boundaries) == 10.0


def test_snap_respects_max_shift():
    """The bug that produced a 769s clip: an unpunctuated stretch makes the
    next boundary minutes away, so the snap must decline to travel."""
    boundaries = [0.0, 500.0]
    assert ClipPipeline._snap_forward(100.0, boundaries) == 100.0
    assert ClipPipeline._snap_back(400.0, boundaries) == 400.0


def test_snap_travels_up_to_but_not_past_limit():
    target = 100.0
    inside = target + MAX_SNAP_SHIFT - 0.1
    outside = target + MAX_SNAP_SHIFT + 0.1
    assert ClipPipeline._snap_forward(target, [inside]) == inside
    assert ClipPipeline._snap_forward(target, [outside]) == target


def test_snap_with_no_boundaries_returns_target():
    assert ClipPipeline._snap_back(42.0, []) == 42.0
    assert ClipPipeline._snap_forward(42.0, []) == 42.0


# -- length cap ----------------------------------------------------------

def test_trim_prefers_latest_sentence_end_within_cap():
    p = make_pipeline([])
    start, end = p._trim_to_max(0.0, 200.0, 75.0, [], [10.0, 70.0, 90.0])
    assert (start, end) == (0.0, 70.0)


def test_trim_falls_back_to_segment_edge_without_sentence_ends():
    p = make_pipeline([seg("x", 0.0, 60.0)])
    start, end = p._trim_to_max(0.0, 300.0, 75.0, [], [])
    assert (start, end) == (0.0, 60.0)


def test_trim_hard_cuts_when_nothing_fits():
    """The cap must hold even with no usable boundary of any kind --
    otherwise an over-long clip escapes, which is what originally happened."""
    p = make_pipeline([seg("x", 0.0, 500.0)])
    start, end = p._trim_to_max(0.0, 500.0, 75.0, [], [])
    assert (start, end) == (0.0, 75.0)


def test_trim_leaves_short_clips_alone():
    p = make_pipeline([])
    # A clip already within the cap should be untouched by the caller; verify
    # trim itself picks the latest fitting end rather than shrinking harder.
    start, end = p._trim_to_max(0.0, 50.0, 75.0, [], [20.0, 50.0])
    assert (start, end) == (0.0, 50.0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
