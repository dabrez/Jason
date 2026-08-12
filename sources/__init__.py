from .base import SourceMetadata, VideoSource
from .youtube import YouTubeSource
from .twitch import TwitchSource, TwitchAuthError


def resolve_source(url: str) -> VideoSource:
    """Picks the right VideoSource implementation for a given URL."""
    for source_cls in (TwitchSource, YouTubeSource):
        source = source_cls(url)
        if source.is_valid():
            return source
    raise ValueError(f"Unrecognized or unsupported video URL: {url}")


__all__ = [
    "SourceMetadata",
    "VideoSource",
    "YouTubeSource",
    "TwitchSource",
    "TwitchAuthError",
    "resolve_source",
]
