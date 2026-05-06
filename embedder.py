from __future__ import annotations
import json
import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import config


class EmbedderFAISS:
    """
    Manages a FAISS flat index + a parallel metadata list.

    Supports two namespaces stored in separate index files:
      • "summaries"  – topic / 100-msg checkpoint summaries
      • "chunks"     – raw message window chunks
    """

    def __init__(self, namespace: str):
        self.namespace = namespace
        self._model: SentenceTransformer | None = None
        self._index: faiss.IndexFlatIP | None = None
        self._meta: list[dict] = []  # parallel list to FAISS rows
        self._index_path = os.path.join(
            config.VECTOR_STORE_DIR, f"{namespace}.faiss"
        )
        self._meta_path = os.path.join(
            config.VECTOR_STORE_DIR, f"{namespace}_meta.pkl"
        )

    #lazy-load embedding model
    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            print(f"[embedder:{self.namespace}] Loading {config.EMBED_MODEL} ...")
            self._model = SentenceTransformer(config.EMBED_MODEL)
        return self._model

    #index lifecycle
    def _new_index(self) -> faiss.IndexFlatIP:
        return faiss.IndexFlatIP(config.EMBED_DIM)

    def load_or_create(self) -> "EmbedderFAISS":
        os.makedirs(config.VECTOR_STORE_DIR, exist_ok=True)
        if os.path.exists(self._index_path) and os.path.exists(self._meta_path):
            self._index = faiss.read_index(self._index_path)
            with open(self._meta_path, "rb") as f:
                self._meta = pickle.load(f)
            print(f"[embedder:{self.namespace}] Loaded {len(self._meta)} vectors")
        else:
            self._index = self._new_index()
            self._meta = []
        return self

    def save(self):
        os.makedirs(config.VECTOR_STORE_DIR, exist_ok=True)
        faiss.write_index(self._index, self._index_path)
        with open(self._meta_path, "wb") as f:
            pickle.dump(self._meta, f)
        print(f"[embedder:{self.namespace}] Saved {len(self._meta)} vectors")

    #add items
    def _embed(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, show_progress_bar=False,
                                 batch_size=64, normalize_embeddings=True)
        return vecs.astype("float32")

    def add(self, texts: list[str], metas: list[dict]):
        """Embed texts and add to FAISS. metas must be same length as texts."""
        assert len(texts) == len(metas)
        vecs = self._embed(texts)
        self._index.add(vecs)
        self._meta.extend(metas)

    #query
    def query(self, query_text: str, top_k: int = 5) -> list[dict]:
        """Return top_k metadata dicts sorted by cosine similarity."""
        if self._index.ntotal == 0:
            return []
        vec = self._embed([query_text])
        scores, indices = self._index.search(vec, min(top_k, self._index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            item = dict(self._meta[idx])
            item["_score"] = float(score)
            results.append(item)
        return results

    #cosine distance helper (used for topic detection)
    def cosine_distance(self, text_a: str, text_b: str) -> float:
        """Returns 1 - cosine_similarity (0 = identical, 2 = opposite)."""
        vecs = self._embed([text_a, text_b])
        similarity = float(np.dot(vecs[0], vecs[1]))  # already L2-normalised
        return 1.0 - similarity

    @property
    def size(self) -> int:
        return self._index.ntotal if self._index else 0
