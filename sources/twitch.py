"""Twitch VOD video source, backed by yt-dlp for download and the Helix API
for metadata. Chat replay (used for highlight scoring) is fetched separately
via TwitchDownloaderCLI since Twitch has no public REST endpoint for it.
"""
import json
import os
import re
import shutil
import subprocess
import tempfile
from typing import List, Optional

import requests
import yt_dlp

from .base import SourceMetadata, VideoSource

HELIX_BASE = "https://api.twitch.tv/helix"
OAUTH_TOKEN_URL = "https://id.twitch.tv/oauth2/token"

VOD_URL_RE = re.compile(r"twitch\.tv/videos/(\d+)")


class TwitchAuthError(RuntimeError):
    """Raised when TWITCH_CLIENT_ID / TWITCH_CLIENT_SECRET are missing or invalid."""


class TwitchSource(VideoSource):
    def __init__(self, url: str, client_id: Optional[str] = None, client_secret: Optional[str] = None):
        super().__init__(url)
        self.client_id = client_id or os.environ.get("TWITCH_CLIENT_ID")
        self.client_secret = client_secret or os.environ.get("TWITCH_CLIENT_SECRET")
        self._app_access_token: Optional[str] = None

    def supports_chat(self) -> bool:
        return True

    def vod_id(self) -> Optional[str]:
        match = VOD_URL_RE.search(self.url)
        return match.group(1) if match else None

    def is_valid(self) -> bool:
        return self.vod_id() is not None

    def _get_app_access_token(self) -> str:
        if self._app_access_token:
            return self._app_access_token
        if not self.client_id or not self.client_secret:
            raise TwitchAuthError(
                "TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET must be set (env vars or "
                "constructor args) to look up VOD metadata. Register an app at "
                "https://dev.twitch.tv/console/apps to obtain them."
            )
        response = requests.post(
            OAUTH_TOKEN_URL,
            params={
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "grant_type": "client_credentials",
            },
            timeout=15,
        )
        response.raise_for_status()
        self._app_access_token = response.json()["access_token"]
        return self._app_access_token

    def _helix_headers(self) -> dict:
        return {
            "Client-Id": self.client_id,
            "Authorization": f"Bearer {self._get_app_access_token()}",
        }

    def resolve(self) -> SourceMetadata:
        vod_id = self.vod_id()
        if vod_id is None:
            raise ValueError(f"Not a Twitch VOD URL: {self.url}")

        response = requests.get(
            f"{HELIX_BASE}/videos",
            params={"id": vod_id},
            headers=self._helix_headers(),
            timeout=15,
        )
        response.raise_for_status()
        data = response.json().get("data", [])
        if not data:
            raise ValueError(f"Twitch VOD {vod_id} not found or unavailable")

        video = data[0]
        self.metadata = SourceMetadata(
            source_id=vod_id,
            title=video.get("title", ""),
            url=self.url,
        )
        return self.metadata

    def download(self, dest_path: Optional[str] = None) -> str:
        if dest_path is None:
            # See YouTubeSource.download: reserve a path, not a file --
            # yt-dlp skips downloading if the output path already exists.
            dest_path = os.path.join(tempfile.mkdtemp(), "video.mp4")

        ydl_opts = {
            "quiet": True,
            "format": "best[ext=mp4]/best",
            "outtmpl": dest_path,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([self.url])

        self.video_path = dest_path
        return self.video_path

    def fetch_chat(self, output_path: Optional[str] = None) -> List[dict]:
        """Downloads VOD chat replay via TwitchDownloaderCLI and returns the
        parsed comment list (each with a relative `content_offset_seconds`
        timestamp), used as input to chat-spike highlight scoring.
        """
        vod_id = self.vod_id()
        if vod_id is None:
            raise ValueError(f"Not a Twitch VOD URL: {self.url}")

        if shutil.which("TwitchDownloaderCLI") is None:
            raise EnvironmentError(
                "TwitchDownloaderCLI is required to fetch chat replay. Download it from "
                "https://github.com/lay295/TwitchDownloader and ensure it is on your PATH."
            )

        if output_path is None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
                output_path = tmp.name

        subprocess.run(
            [
                "TwitchDownloaderCLI", "chatdownload",
                "--id", vod_id,
                "-o", output_path,
                "--embed-images", "false",
            ],
            check=True,
        )

        with open(output_path, "r", encoding="utf-8") as f:
            chat_log = json.load(f)

        return chat_log.get("comments", [])
