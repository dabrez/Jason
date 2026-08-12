"""Word-by-word burned-in captions, TikTok/Reels/Shorts style.

Groups Whisper word timestamps into short bursts (1-4 words), renders them as
an ASS subtitle file with large bold centered text, and burns them into the
video via ffmpeg's `ass` filter. ASS (not SRT) is used because it supports
the styling (font size/weight, position, color) this style needs -- SRT is
plain text with no styling control.
"""
import os
import shutil
import subprocess
import tempfile
from typing import List, Optional

import cv2

WORDS_PER_BURST = 3
MAX_BURST_SECONDS = 1.5

# Font size as a fraction of video height -- TikTok/Reels-style captions read
# as roughly 1/12th of frame height. Kept resolution-relative rather than a
# fixed point size since PlayResY is set to the actual output resolution
# below (libass scales font size relative to PlayResY, so a fixed size would
# look tiny on a 4K source and huge on a 480p one).
FONT_SIZE_FRACTION = 1 / 12


def _build_style(video_height: int) -> str:
    font_size = max(24, int(video_height * FONT_SIZE_FRACTION))
    return (
        f"Style: Default,Arial Black,{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
        "-1,0,0,0,100,100,0,0,1,3,2,2,60,60,120,1"
    )


def group_words_into_bursts(
    words: List[dict],
    words_per_burst: int = WORDS_PER_BURST,
    max_burst_seconds: float = MAX_BURST_SECONDS,
) -> List[dict]:
    """Groups a flat word-timestamp list into caption bursts of up to
    `words_per_burst` words, also splitting early if a burst would otherwise
    span more than `max_burst_seconds` (keeps captions snappy even when
    Whisper's word timing has a long pause inside a would-be burst).
    """
    bursts = []
    current = []

    def flush():
        if not current:
            return
        bursts.append({
            "text": " ".join(w["word"] for w in current).strip(),
            "start": current[0]["start"],
            "end": current[-1]["end"],
        })

    for word in words:
        if not word.get("word", "").strip():
            continue
        if current and (
            len(current) >= words_per_burst
            or (word["end"] - current[0]["start"]) > max_burst_seconds
        ):
            flush()
            current = []
        current.append(word)
    flush()

    return bursts


def _format_ass_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}:{minutes:02d}:{secs:05.2f}"


def _escape_ass_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def build_ass_subtitles(bursts: List[dict], video_width: int, video_height: int) -> str:
    """Renders caption bursts into ASS subtitle file content. PlayResX/Y are
    set to the actual output video resolution so libass's font-size scaling
    matches what actually gets burned in, regardless of source resolution.
    """
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "WrapStyle: 0\n"
        f"PlayResX: {video_width}\n"
        f"PlayResY: {video_height}\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,"
        "Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,"
        "Alignment,MarginL,MarginR,MarginV,Encoding\n"
        f"{_build_style(video_height)}\n"
        "\n"
        "[Events]\n"
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
    )

    lines = []
    for burst in bursts:
        start = _format_ass_timestamp(burst["start"])
        end = _format_ass_timestamp(burst["end"])
        text = _escape_ass_text(burst["text"].upper())
        lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")

    return header + "\n".join(lines) + "\n"


def write_ass_file(
    bursts: List[dict],
    video_width: int,
    video_height: int,
    output_path: Optional[str] = None,
) -> str:
    if output_path is None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ass") as tmp:
            output_path = tmp.name
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(build_ass_subtitles(bursts, video_width, video_height))
    return output_path


def burn_in_captions(
    video_path: str,
    words: List[dict],
    output_path: Optional[str] = None,
    words_per_burst: int = WORDS_PER_BURST,
    max_burst_seconds: float = MAX_BURST_SECONDS,
) -> str:
    """Burns word-by-word captions into a video and returns the output path."""
    if shutil.which("ffmpeg") is None:
        raise EnvironmentError(
            "ffmpeg is required for caption burn-in. "
            "Install it from https://ffmpeg.org/ and ensure it is on your PATH."
        )
    if output_path is None:
        with tempfile.NamedTemporaryFile(delete=False, suffix="_captioned.mp4") as tmp:
            output_path = tmp.name

    bursts = group_words_into_bursts(words, words_per_burst, max_burst_seconds)
    if not bursts:
        shutil.copyfile(video_path, output_path)
        return output_path

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    ass_path = write_ass_file(bursts, video_width, video_height)
    try:
        # ffmpeg's filtergraph treats ':' as an option separator, so the
        # subtitle file path needs escaping if it contains one (e.g. a
        # Windows drive letter, or if we ever end up with a colon in a temp
        # dir path).
        escaped_ass_path = ass_path.replace("\\", "/").replace(":", "\\:")
        result = subprocess.run([
            "ffmpeg", "-y", "-i", video_path,
            "-vf", f"ass={escaped_ass_path}",
            "-c:a", "copy",
            output_path,
        ], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg caption burn-in failed:\n{result.stderr[-2000:]}")
    finally:
        os.remove(ass_path)

    return output_path
