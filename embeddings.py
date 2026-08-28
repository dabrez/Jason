"""Pluggable text-embedding backends for transcript topic segmentation and
semantic highlight scoring (see pipeline.py's _compute_embeddings).

Two backends:
- sentence-transformers (local, in-process): the default, all-MiniLM-L6-v2.
- Ollama (local server, /api/embed): any embedding model already pulled via
  `ollama pull <name>`, e.g. qwen3-embedding, nomic-embed-text. Lets larger
  embedding models be swapped in for comparison without adding them as a
  hard pip dependency -- Ollama already manages the model weights/runtime.

get_embedder(name) picks the backend by name: "ollama:<model>" routes to
Ollama, anything else is loaded as a sentence-transformers model id.
"""
import abc
from typing import List

import numpy as np
import requests

OLLAMA_URL = "http://localhost:11434"


class Embedder(abc.ABC):
    @abc.abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """Returns an (n_texts, dim) array of embeddings."""


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        return np.asarray(self._model.encode(texts))


class OllamaEmbedder(Embedder):
    """Batches through Ollama's /api/embed endpoint. Falls back to one
    request per text if the batch endpoint isn't available (older Ollama
    versions only expose the singular /api/embeddings).
    """

    def __init__(self, model_name: str, ollama_url: str = OLLAMA_URL, timeout: float = 60.0):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.timeout = timeout

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0))
        try:
            response = requests.post(
                f"{self.ollama_url}/api/embed",
                json={"model": self.model_name, "input": texts},
                timeout=self.timeout,
            )
            response.raise_for_status()
            embeddings = response.json()["embeddings"]
            return np.asarray(embeddings)
        except (requests.RequestException, KeyError):
            return self._encode_one_by_one(texts)

    def _encode_one_by_one(self, texts: List[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            response = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json={"model": self.model_name, "prompt": text},
                timeout=self.timeout,
            )
            response.raise_for_status()
            vectors.append(response.json()["embedding"])
        return np.asarray(vectors)


_embedder_cache = {}


def get_embedder(name: str) -> Embedder:
    """name: a sentence-transformers model id (e.g. "all-MiniLM-L6-v2"), or
    "ollama:<model>" (e.g. "ollama:qwen3-embedding") to route through a
    locally-running Ollama server instead. Embedders are cached by name so
    repeated calls (e.g. across pipeline stages) don't reload the model or
    reopen a session each time.
    """
    if name in _embedder_cache:
        return _embedder_cache[name]

    if name.startswith("ollama:"):
        model_name = name.split(":", 1)[1]
        embedder: Embedder = OllamaEmbedder(model_name)
    else:
        embedder = SentenceTransformerEmbedder(name)

    _embedder_cache[name] = embedder
    return embedder
