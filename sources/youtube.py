"""YouTube video source, backed by yt-dlp."""
import os
import re
import shutil
import tempfile
from typing import Optional
from urllib.parse import urlparse, parse_qs

import yt_dlp

from .base import SourceMetadata, VideoSource


class YouTubeSource(VideoSource):
    def is_valid(self) -> bool:
        return self.video_id() is not None

    def video_id(self) -> Optional[str]:
        parsed_url = urlparse(self.url)

        if parsed_url.netloc == "youtu.be":
            return parsed_url.path[1:] or None

        if parsed_url.netloc in ("www.youtube.com", "youtube.com"):
            query_params = parse_qs(parsed_url.query)
            if "v" in query_params:
                return query_params["v"][0]
            if parsed_url.path.startswith("/embed/"):
                return parsed_url.path.split("/")[2]
            if parsed_url.path.startswith("/v/"):
                return parsed_url.path.split("/")[2]

        regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
        match = re.search(regex, self.url)
        return match.group(1) if match else None

    def resolve(self) -> SourceMetadata:
        with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True}) as ydl:
            info = ydl.extract_info(self.url, download=False)
        self.metadata = SourceMetadata(
            source_id=info.get("id", self.video_id() or ""),
            title=info.get("title", ""),
            url=self.url,
        )
        return self.metadata

    def download(self, dest_path: Optional[str] = None) -> str:
        if dest_path is None:
            # Reserve a path, not a file: yt-dlp treats an already-existing
            # output path as "already downloaded" and skips downloading, so
            # NamedTemporaryFile (which creates the file immediately) would
            # silently leave an empty video. mkdtemp only creates the
            # directory; the mp4 itself is written by yt-dlp.
            dest_path = os.path.join(tempfile.mkdtemp(), "video.mp4")

        ydl_opts = {
            "quiet": True,
            # The default web/android-vr client fallback yt-dlp picks
            # currently trips YouTube's signature/DRM checks for many
            # videos (403s on the separate video+audio streams, or the
            # tv client's formats coming back DRM-protected). The android
            # client's progressive (single-file, pre-merged) formats sidestep
            # this -- no signature deciphering or stream merge needed -- at
            # the cost of being capped at 360p. Good enough for transcription
            # / highlight scoring / face tracking, which don't need source
            # resolution beyond what the final vertical crop uses anyway.
            "format": "best[ext=mp4]/best",
            "extractor_args": {"youtube": {"player_client": ["android"]}},
            "outtmpl": dest_path,
            "merge_output_format": "mp4",
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([self.url])

        self.video_path = dest_path
        return self.video_path
