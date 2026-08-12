"""Source-agnostic pipeline: ingest -> transcribe -> segment/score -> cut.

Two clip-selection strategies feed the same cutting step:
- Fused highlight scoring (works for any source): combines chat-spike
  (Twitch only), audio-energy, and transcript semantic-hook signals into one
  ranked list of candidate windows, then snaps each window to the nearest
  transcript boundary so clips don't start/end mid-sentence. See
  highlights/fusion.py.
- Topic segmentation (works for any source): splits the transcript into
  topically coherent chapters. Used as the fallback when fusion finds no
  clear highlights (e.g. a quiet video with no obvious spikes anywhere).

Both produce a list of clip dicts: {start_time, end_time, text, ...}. Adding
another highlight signal means adding another scorer module with a
score_series(...) function and wiring it into select_highlights below, not
touching the cutting code.
"""
import os
import re
import shutil
import subprocess
from typing import List, Optional

import numpy as np
import whisper
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from highlights import audio_score_series, chat_score_series, fuse_highlights, semantic_score_series
from highlights.chat_spikes import DEFAULT_BUCKET_SECONDS
from reformat import burn_in_captions, crop_to_vertical
from sources import VideoSource


class ClipPipeline:
    def __init__(self, source: VideoSource):
        self.source = source
        self.transcript = None
        self.texts = None
        self.time_stamps = None
        self.embeddings = None
        self.labels = None

    # -- ingestion -----------------------------------------------------
    def ingest(self):
        self.source.resolve()
        self.source.download()
        return self.source.video_path

    # -- transcription ---------------------------------------------------
    def transcribe(self):
        """Transcribes the video's audio using Whisper. Requests word-level
        timestamps too (used by reformat/captions.py for word-by-word
        burned-in captions) -- stored per-segment as `words`, so existing
        segment-level consumers (topic segmentation, highlight scoring) are
        unaffected.
        """
        if self.source.video_path is None:
            self.ingest()
        model = whisper.load_model("base")
        result = model.transcribe(self.source.video_path, word_timestamps=True)
        self.transcript = []
        for seg in result.get("segments", []):
            self.transcript.append({
                "text": seg["text"].strip(),
                "start": seg["start"],
                "duration": seg["end"] - seg["start"],
                "words": [
                    {"word": w["word"].strip(), "start": w["start"], "end": w["end"]}
                    for w in seg.get("words", [])
                ],
            })
        return self.transcript

    def word_timestamps(self):
        """Flattens the transcript's per-segment word lists into one
        chronological list of {word, start, end}, for caption generation.
        """
        if self.transcript is None:
            self.transcribe()
        words = []
        for seg in self.transcript:
            words.extend(seg.get("words", []))
        return words

    def _preprocess_text(self):
        if self.transcript is None:
            self.transcribe()
        self.texts = []
        self.time_stamps = []
        for entry in self.transcript:
            text = re.sub(r"[\W_]+", " ", entry["text"]).lower()
            self.texts.append(text)
            self.time_stamps.append((entry["start"], entry["start"] + entry["duration"]))

    # -- topic segmentation (fallback path, any source) -------------------
    def _compute_embeddings(self):
        self._preprocess_text()
        model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = model.encode(self.texts)

    def _estimate_optimal_topics(self):
        max_topics = min(10, len(self.embeddings))
        scores = []
        K = range(2, max_topics)
        for k in K:
            clustering = AgglomerativeClustering(n_clusters=k, affinity="cosine", linkage="average")
            labels = clustering.fit_predict(self.embeddings)
            scores.append(silhouette_score(self.embeddings, labels, metric="cosine"))
        return K[np.argmax(scores)]

    def segment_by_topic(self, num_topics=None):
        """Splits the transcript into topically coherent chapters."""
        self._compute_embeddings()
        if num_topics is None:
            num_topics = self._estimate_optimal_topics()
        clustering = AgglomerativeClustering(n_clusters=num_topics, affinity="cosine", linkage="average")
        self.labels = clustering.fit_predict(self.embeddings)

        chapters = []
        current = {
            "start_time": self.time_stamps[0][0],
            "end_time": self.time_stamps[0][1],
            "text": self.texts[0],
            "label": self.labels[0],
        }
        for idx in range(1, len(self.labels)):
            if self.labels[idx] == current["label"]:
                current["end_time"] = self.time_stamps[idx][1]
                current["text"] += " " + self.texts[idx]
            else:
                chapters.append(current)
                current = {
                    "start_time": self.time_stamps[idx][0],
                    "end_time": self.time_stamps[idx][1],
                    "text": self.texts[idx],
                    "label": self.labels[idx],
                }
        chapters.append(current)
        return chapters

    # -- fused highlight scoring (any source) ------------------------------
    def select_highlights(
        self,
        pad_seconds: float = 5.0,
        bucket_seconds: float = DEFAULT_BUCKET_SECONDS,
        use_ollama: bool = False,
    ):
        """Combines whatever highlight signals are available for this source
        (chat spikes on Twitch, audio energy and semantic hooks on any
        source) into one ranked list of candidate clips, snapped to the
        nearest transcript segment boundary.

        Chat and audio are computed unconditionally when available; semantic
        scoring needs sentence embeddings, so it reuses _compute_embeddings
        (also used by topic segmentation) rather than re-encoding text twice.
        """
        if self.transcript is None:
            self.transcribe()
        if self.embeddings is None:
            self._compute_embeddings()

        signal_series = {
            "chat": chat_score_series(self.source.fetch_chat(), bucket_seconds)
                if self.source.supports_chat() else {},
            "audio": audio_score_series(self.source.video_path, bucket_seconds),
            "semantic": semantic_score_series(self.transcript, self.embeddings, bucket_seconds),
        }

        windows = fuse_highlights(signal_series, bucket_seconds)
        if use_ollama and windows:
            from highlights import rank_with_ollama
            texts = [self._text_between(w.start, w.end) for w in windows]
            windows = rank_with_ollama(windows, texts)

        boundaries = sorted({t for pair in self._segment_boundaries() for t in pair})
        clips = []
        for window in windows:
            start = self._snap_to_boundary(max(0.0, window.start - pad_seconds), boundaries)
            end = self._snap_to_boundary(window.end + pad_seconds, boundaries)
            if end <= start:
                continue
            clips.append({
                "start_time": start,
                "end_time": end,
                "text": self._text_between(start, end),
                "score": window.score,
                "signals": window.detail,
            })
        return clips

    def _segment_boundaries(self):
        if self.transcript is None:
            self.transcribe()
        return [(seg["start"], seg["start"] + seg["duration"]) for seg in self.transcript]

    @staticmethod
    def _snap_to_boundary(target: float, boundaries):
        if not boundaries:
            return target
        return min(boundaries, key=lambda b: abs(b - target))

    def _text_between(self, start: float, end: float) -> str:
        if self.transcript is None:
            return ""
        return " ".join(
            seg["text"] for seg in self.transcript
            if seg["start"] >= start and seg["start"] < end
        )

    # -- titling -----------------------------------------------------------
    @staticmethod
    def generate_titles(clips):
        titles = []
        for clip in clips:
            text = clip.get("text", "").strip()
            if not text:
                titles.append("clip")
                continue
            vectorizer = TfidfVectorizer(stop_words="english")
            try:
                X = vectorizer.fit_transform([text])
            except ValueError:
                titles.append("clip")
                continue
            indices = np.argsort(vectorizer.idf_)[::-1]
            features = vectorizer.get_feature_names_out()
            top_features = [features[i] for i in indices[:3]]
            titles.append(" ".join(top_features) if top_features else "clip")
        return titles

    # -- cutting -------------------------------------------------------
    def save_clips(self, clips, titles=None, output_dir="segments"):
        if shutil.which("ffmpeg") is None:
            raise EnvironmentError(
                "ffmpeg is required to save video segments. "
                "Install it from https://ffmpeg.org/ and ensure it is on your PATH."
            )
        if self.source.video_path is None:
            self.ingest()

        os.makedirs(output_dir, exist_ok=True)
        clip_paths = []
        for idx, clip in enumerate(clips):
            filename = f"segment_{idx + 1}.mp4"
            if titles:
                safe_title = re.sub(r"[^a-zA-Z0-9_-]+", "_", titles[idx]).strip("_")
                filename = f"{idx + 1:02d}_{safe_title}.mp4"
            output_path = os.path.join(output_dir, filename)
            subprocess.run([
                "ffmpeg", "-y", "-i", self.source.video_path,
                "-ss", str(clip["start_time"]), "-to", str(clip["end_time"]),
                "-c", "copy", output_path,
            ], check=True)
            clip_paths.append(output_path)
        return output_dir, clip_paths

    # -- vertical reformatting (Shorts/Reels/TikTok) ------------------------
    def reformat_clip(
        self,
        clip_path: str,
        clip: dict,
        output_path: Optional[str] = None,
        multi_speaker: bool = True,
    ) -> str:
        """Crops a cut clip to 9:16 (following the active speaker's face --
        switches between tracked faces using mouth-movement + audio when
        `multi_speaker` is set, otherwise always follows the single largest
        detected face) and burns in word-by-word captions. `clip` must be
        the same dict produced by select_highlights/segment_by_topic (needs
        start_time) so word timestamps can be shifted from full-video time
        to clip-relative time.
        """
        if output_path is None:
            base, ext = os.path.splitext(clip_path)
            output_path = f"{base}_vertical{ext}"

        cropped_path = crop_to_vertical(clip_path, multi_speaker=multi_speaker)
        try:
            clip_start = clip["start_time"]
            words = [
                {"word": w["word"], "start": w["start"] - clip_start, "end": w["end"] - clip_start}
                for w in self.word_timestamps()
                if clip_start <= w["start"] < clip["end_time"]
            ]
            burn_in_captions(cropped_path, words, output_path=output_path)
        finally:
            if cropped_path != output_path:
                os.remove(cropped_path)

        return output_path

    def reformat_all(self, clips, clip_paths, suffix: str = "_vertical", multi_speaker: bool = True) -> List[str]:
        """Runs reformat_clip over every saved clip; returns the list of
        vertical output paths in the same order as clip_paths.
        """
        outputs = []
        for clip, clip_path in zip(clips, clip_paths):
            outputs.append(self.reformat_clip(clip_path, clip, multi_speaker=multi_speaker))
        return outputs

    # -- top-level entrypoint --------------------------------------------
    def run(self, output_dir="segments", use_ollama: bool = False, vertical: bool = False, multi_speaker: bool = True):
        """Picks fused highlight windows (chat/audio/semantic, whichever
        signals the source supports); falls back to topic segmentation if
        fusion finds no clear highlights at all. If `vertical` is set, also
        produces a 9:16 face-tracked, captioned version of each clip
        (reformat/crop.py + reformat/captions.py) alongside the normal cut.
        `multi_speaker` controls whether the crop follows whichever tracked
        face is actively talking (mouth-movement + audio) or always the
        single largest detected face; ignored when only one face is ever
        detected in a clip.
        """
        self.ingest()
        self.transcribe()

        clips = self.select_highlights(use_ollama=use_ollama)
        if not clips:
            clips = self.segment_by_topic()

        titles = self.generate_titles(clips)
        output_dir, clip_paths = self.save_clips(clips, titles, output_dir)

        vertical_paths = self.reformat_all(clips, clip_paths, multi_speaker=multi_speaker) if vertical else []
        return output_dir, clips, clip_paths, vertical_paths
