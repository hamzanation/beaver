"""
Phase 2 embedding infrastructure: model loading, disk-backed caching keyed by text
hash, sentence embeddings, mean-pooled trace embeddings, and a separate whole-trace
long-context embedding (bge-m3) as a robustness check.

GPU confirmed available (RTX 3060 Ti, torch sees cuda:0) -- default device is cuda,
override with device="cpu" if needed (e.g. if VRAM is under pressure from something
else running concurrently).
"""
import hashlib
import os
import random
from pathlib import Path

import numpy as np

CACHE_DIR = Path(__file__).resolve().parents[1] / "outputs" / "embeddings_cache"

DEFAULT_SENTENCE_MODEL = "BAAI/bge-large-en-v1.5"
CPU_FALLBACK_MODEL = "all-mpnet-base-v2"
LONG_CONTEXT_MODEL = "BAAI/bge-m3"

SEED = 42


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


class EmbeddingCache:
    """One .npy file per (model_slug, text_hash) under outputs/embeddings_cache/.
    Simple, incremental, no rewrite-the-whole-archive cost per new item -- fine at
    our scale (thousands of sentences, not millions)."""

    def __init__(self, model_slug: str, cache_dir: Path = CACHE_DIR):
        self.dir = cache_dir / model_slug
        self.dir.mkdir(parents=True, exist_ok=True)

    def get(self, text: str):
        p = self.dir / f"{_hash_text(text)}.npy"
        if p.exists():
            return np.load(p)
        return None

    def put(self, text: str, vec: np.ndarray):
        p = self.dir / f"{_hash_text(text)}.npy"
        np.save(p, vec)


_MODEL_CACHE = {}


def _get_model(model_name: str, device: str = "cuda"):
    key = (model_name, device)
    if key not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer
        _MODEL_CACHE[key] = SentenceTransformer(model_name, device=device)
    return _MODEL_CACHE[key]


def embed_texts(texts, model_name: str = DEFAULT_SENTENCE_MODEL, device: str = "cuda",
                 batch_size: int = 32, show_progress_bar: bool = False):
    """Embeds a list of texts, using the on-disk cache for anything already computed.
    Returns an (n, dim) float32 array in the same order as `texts`."""
    model_slug = model_name.replace("/", "__")
    cache = EmbeddingCache(model_slug)

    results = [None] * len(texts)
    to_compute_idx = []
    to_compute_text = []
    for i, t in enumerate(texts):
        cached = cache.get(t)
        if cached is not None:
            results[i] = cached
        else:
            to_compute_idx.append(i)
            to_compute_text.append(t)

    if to_compute_text:
        model = _get_model(model_name, device=device)
        vecs = model.encode(
            to_compute_text, batch_size=batch_size, normalize_embeddings=True,
            show_progress_bar=show_progress_bar, convert_to_numpy=True,
        )
        for i, t, v in zip(to_compute_idx, to_compute_text, vecs):
            v = v.astype(np.float32)
            cache.put(t, v)
            results[i] = v

    return np.stack(results)


def embed_trace_mean_pooled(sentences, model_name: str = DEFAULT_SENTENCE_MODEL, device: str = "cuda"):
    """Trace embedding = mean of its (unit-normalized) sentence embeddings, per spec.
    Re-normalized to unit length after averaging -- mean-pooling normalized vectors
    does NOT preserve unit norm, and downstream code (model_similarity_matrix) treats
    a plain dot product as cosine similarity, which is only valid for unit vectors."""
    if not sentences:
        return None
    sent_embs = embed_texts(sentences, model_name=model_name, device=device)
    mean_vec = sent_embs.mean(axis=0)
    norm = np.linalg.norm(mean_vec)
    return mean_vec / norm if norm > 0 else mean_vec


def embed_trace_long_context(text, model_name: str = LONG_CONTEXT_MODEL, device: str = "cuda"):
    """Whole-trace embedding with a long-context model, as the Phase 2 robustness
    check against the mean-pooled sentence embedding above."""
    if not text or not text.strip():
        return None
    return embed_texts([text], model_name=model_name, device=device)[0]
