"""
embed_utils.py — Thin wrapper around fastembed that mimics the SentenceTransformer API.
Import this instead of fastembed directly.
"""
from fastembed import TextEmbedding as _TextEmbedding
import numpy as np

class Embedder:
    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        self._model = _TextEmbedding(model)

    def encode(self, texts, batch_size: int = 64, show_progress_bar: bool = False) -> list:
        if isinstance(texts, str):
            return list(self._model.embed([texts]))[0].tolist()
        result = list(self._model.embed(texts))
        return [e.tolist() if isinstance(e, np.ndarray) else e for e in result]

embedder = Embedder()
