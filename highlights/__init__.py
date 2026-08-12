from .base import HighlightWindow
from .chat_spikes import score_chat_spikes, score_series as chat_score_series
from .audio_energy import score_audio_spikes, score_series as audio_score_series
from .semantic_hooks import score_semantic_hooks, score_series as semantic_score_series, rank_with_ollama
from .fusion import fuse_scores, fuse_highlights, DEFAULT_WEIGHTS

__all__ = [
    "HighlightWindow",
    "score_chat_spikes",
    "chat_score_series",
    "score_audio_spikes",
    "audio_score_series",
    "score_semantic_hooks",
    "semantic_score_series",
    "rank_with_ollama",
    "fuse_scores",
    "fuse_highlights",
    "DEFAULT_WEIGHTS",
]
