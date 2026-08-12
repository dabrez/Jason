"""Common interface for pulling a video (plus optional chat data) from a platform."""
import abc
from dataclasses import dataclass
from typing import Optional


@dataclass
class SourceMetadata:
    """Basic info about a resolved video, independent of platform."""
    source_id: str
    title: str
    url: str


class VideoSource(abc.ABC):
    """A platform-specific way to resolve a URL to a downloadable video.

    Implementations: YouTubeSource, TwitchSource. Each produces a local media
    file plus metadata; everything downstream (transcription, segmentation,
    highlight scoring, cutting) works against that, not the platform API.
    """

    def __init__(self, url: str):
        self.url = url
        self.metadata: Optional[SourceMetadata] = None
        self.video_path: Optional[str] = None

    @abc.abstractmethod
    def is_valid(self) -> bool:
        """Whether `url` belongs to and is reachable on this platform."""

    @abc.abstractmethod
    def resolve(self) -> SourceMetadata:
        """Fetch metadata (id/title) without downloading the full video."""

    @abc.abstractmethod
    def download(self, dest_path: Optional[str] = None) -> str:
        """Download the video, store the local path, and return it."""

    def supports_chat(self) -> bool:
        """Whether this source can provide chat replay data for highlight scoring."""
        return False
