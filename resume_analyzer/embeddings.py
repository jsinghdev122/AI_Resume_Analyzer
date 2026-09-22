"""Sentence embeddings, computed locally (no API key, no GPU)."""
import threading
from typing import Dict, Optional, Sequence

import numpy as np


class Embedder:
    """Wraps fastembed's ONNX models. `embed()` returns L2-normalised vectors, so a dot product is cosine similarity."""

    def __init__(self, model_name: str, cache_dir: Optional[str] = None):
        from fastembed import TextEmbedding  # imported here so the rest of the package works without it

        kwargs = {"cache_dir": cache_dir} if cache_dir else {}
        self._model = TextEmbedding(model_name=model_name, **kwargs)
        self._lock = threading.Lock()
        self._cache: Dict[str, np.ndarray] = {}

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            with self._lock:
                vectors = list(self._model.embed(missing))
            for text, vector in zip(missing, vectors):
                vector = np.asarray(vector, dtype=np.float32)
                norm = np.linalg.norm(vector)
                self._cache[text] = vector / norm if norm else vector
            if len(self._cache) > 5000:  # keep memory bounded: drop the oldest half
                for key in list(self._cache)[:2500]:
                    del self._cache[key]
        return np.vstack([self._cache[t] for t in texts])
