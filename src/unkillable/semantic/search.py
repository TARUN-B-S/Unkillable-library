import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from unkillable.utils.logging_config import get_logger

log = get_logger(__name__)

_DEFAULT_EMBED_MODEL = "clip-ViT-B-32"
_HASH_DIM = 512


@dataclass
class SearchResult:
    id: str
    score: float
    metadata: dict


def _token_features(text: str) -> list[str]:
    """Lexical features of a text: word tokens plus char-bigrams.

    Bigrams make the fallback embedding tolerant to word shape variance
    ("trucks" still partially matches "truck").
    """
    words = re.findall(r"[a-z]+", text.lower())
    feats: list[str] = []
    for w in words:
        feats.append("$" + w)
        if len(w) > 2:
            feats.extend(w[i : i + 2] for i in range(len(w) - 1))
    return feats


def _hash_embedding(text: str, dim: int = _HASH_DIM) -> list[float]:
    """Deterministic bag-of-features embedding (used when no real encoder).

    Documents sharing vocabulary get a higher cosine, so the fallback actually
    ranks by lexical overlap instead of hashing each string to noise.
    """
    vec: list[float] = [0.0] * dim
    for feat in _token_features(text):
        digest = hashlib.md5(feat.encode()).digest()
        bucket = int.from_bytes(digest[:4], "big") % dim
        vec[bucket] += 1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def cosine(a: list[float], b: list[float]) -> float:
    """True cosine similarity in [-1, 1].

    CLIP/SBERT embeddings are not L2-normalized, so a raw dot product can
    exceed 1.0 and saturate downstream score mixing.  Normalizing keeps the
    semantic term comparable with the other score components.

    NumPy-backed; a pure-Python fallback is unnecessary since numpy is a hard
    dependency (pulled in by opencv/ultralytics).
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(va @ vb / (na * nb))


def _stat_key(path: Path) -> tuple[int, int] | None:
    """Cache-invalidation key for a file: (mtime_ns, size), or None if unreadable."""
    try:
        st = path.stat()
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


class SemanticSearch:
    """Embedding search with a real encoder and a deterministic hash fallback.

    Mode is selected at construction time:
      * ``UNKILLABLE_EMBEDDING_MODE=auto`` (default): use sentence-transformers
        (CLIP by default so text queries match image frames); any import,
        download, or encode failure falls back to hash embeddings.
      * ``UNKILLABLE_EMBEDDING_MODE=hash``: always use hash embeddings (fast,
        offline, deterministic — used by the unit test suite).
    ``UNKILLABLE_EMBEDDING_MODEL`` overrides the model name.

    The JSONL db is parsed once and cached in memory; the cache is refreshed
    whenever the file's mtime or size changes (so external appends are picked
    up), and this instance's own ``index()`` calls update it incrementally.
    A L2-normalized NumPy matrix of the vectors is cached alongside, so each
    search is a single matrix-vector product instead of a per-row Python loop.
    """

    def __init__(self, model: str | None = None, db_path: Path | None = None) -> None:
        self.model = os.environ.get("UNKILLABLE_EMBEDDING_MODEL") or model or _DEFAULT_EMBED_MODEL
        self.mode = os.environ.get("UNKILLABLE_EMBEDDING_MODE", "auto")
        self.db_path = Path(db_path) if db_path else Path("storage/index/entries.jsonl")
        self._encoder = None
        self._record_cache: list[dict] | None = None
        self._record_cache_key: tuple[int, int] | None = None
        self._matrix_cache: tuple[tuple[int, int] | None, int, np.ndarray, list[int]] | None = None
        self._init_encoder()

    def _init_encoder(self) -> None:
        if self.mode == "hash":
            log.info("Semantic embeddings: hash mode (UNKILLABLE_EMBEDDING_MODE=hash)")
            return
        try:
            import importlib.util

            if importlib.util.find_spec("sentence_transformers") is None:
                log.warning("sentence_transformers not installed, using hash embeddings")
                return
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.model)
            log.info("Semantic encoder loaded: %s (%s)", self.model, self.mode)
        except Exception as exc:
            log.warning("Semantic encoder fallback to hash: %s", exc)
            self._encoder = None

    # ── Embedding ────────────────────────────────────────────────────

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts in a single encoder pass.

        Falls back to per-text hash embeddings when no encoder is available
        (or encoding fails).  Raises ValueError if any text is empty, matching
        :meth:`embed_text`.
        """
        for t in texts:
            if not t or not t.strip():
                raise ValueError("Text queries must be non-empty")
        if not texts:
            return []
        if self._encoder is not None:
            try:
                vecs = self._encoder.encode(texts)
                return [np.asarray(v, dtype=float).tolist() for v in vecs]
            except Exception as exc:
                log.error("Batch text embedding failed: %s", exc)
        return [_hash_embedding(t) for t in texts]

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_images(self, image_paths: list[Path | str]) -> list[list[float]]:
        """Embed a batch of images in a single encoder pass.

        All paths are validated before encoding; raises FileNotFoundError for
        the first missing image.  Hash fallback embeds the file path (same
        behavior as the single-image path without an encoder).
        """
        paths = [Path(p) for p in image_paths]
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(f"Image not found: {p}")
        if not paths:
            return []
        if self._encoder is not None:
            try:
                from PIL import Image

                imgs = [Image.open(p).convert("RGB") for p in paths]
                vecs = self._encoder.encode(imgs)
                return [np.asarray(v, dtype=float).tolist() for v in vecs]
            except Exception as exc:
                log.error("Batch image embedding failed: %s", exc)
        return [_hash_embedding(str(p)) for p in paths]

    def embed_image(self, image_path: Path | str) -> list[float]:
        return self.embed_images([image_path])[0]

    # ── Index store ──────────────────────────────────────────────────

    def index(self, doc_id: str, vector: list[float], metadata: dict) -> None:
        key_before = _stat_key(self.db_path)
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            record = {"id": doc_id, "vector": vector, "metadata": metadata}
            with open(self.db_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as exc:
            log.error("Index write failed: %s", exc)
            raise
        # Keep the in-memory caches in sync without a full re-read — but only
        # when the cache was current before this append; otherwise drop it so
        # the next search re-reads the file (another writer may have added lines).
        if self._record_cache is not None and self._record_cache_key == key_before:
            self._record_cache.append(record)
            self._record_cache_key = _stat_key(self.db_path)
            self._matrix_cache = None
        else:
            self._record_cache = None
            self._record_cache_key = None
            self._matrix_cache = None
        log.info("Indexed %s", doc_id)

    def index_entry(
        self,
        doc_id: str,
        image_vector: list[float],
        text_vector: list[float] | None,
        metadata: dict,
    ) -> None:
        """Index a frame with dual vectors: image embedding + text embedding.

        ``text_vector`` is the embedding of the frame's tags/labels, enabling
        text↔text matching (stronger than CLIP text↔image for tagged frames)
        and lexical fallback matching in hash mode.  ``None`` keeps legacy
        single-vector layout.
        """
        metadata = dict(metadata or {})
        if text_vector:
            metadata["text_vector"] = text_vector
        self.index(doc_id, image_vector, metadata)

    def _load_records(self) -> list[dict]:
        """Parsed db records, cached until the file's (mtime, size) changes."""
        if not self.db_path.exists():
            return []
        key = _stat_key(self.db_path)
        if self._record_cache is not None and key == self._record_cache_key:
            return self._record_cache
        records: list[dict] = []
        try:
            with open(self.db_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        log.warning("Skipping corrupt line in %s", self.db_path)
        except OSError as exc:
            log.error("Search failed: %s", exc)
            raise
        self._record_cache = records
        self._record_cache_key = key
        self._matrix_cache = None
        return records

    def _normalized_matrix(self, dim: int) -> tuple[np.ndarray | None, list[int]]:
        """Row-normalized vector matrix for records whose vector is ``dim`` long.

        Returns the matrix (or None) and the record indices each row
        corresponds to.  Cached with the records.
        """
        records = self._load_records()
        key = self._record_cache_key
        if self._matrix_cache is not None:
            cached_key, cached_dim, matrix, idxs = self._matrix_cache
            if cached_key == key and cached_dim == dim:
                return matrix, idxs
        rows: list[list[float]] = []
        idxs: list[int] = []
        for i, rec in enumerate(records):
            vec = rec.get("vector") or rec.get("embedding")
            if vec and len(vec) == dim:
                rows.append(vec)
                idxs.append(i)
        if rows:
            matrix = np.asarray(rows, dtype=np.float32)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            matrix = matrix / norms
        else:
            matrix = None
        self._matrix_cache = (key, dim, matrix, idxs)
        return matrix, idxs

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if not self.db_path.exists():
            log.warning("No embedding DB at %s", self.db_path)
            return []
        records = self._load_records()
        if not records:
            return []
        qvec = self.embed_text(query)

        scores = [0.0] * len(records)
        qn = np.asarray(qvec, dtype=np.float32)
        qnorm = float(np.linalg.norm(qn))
        if qnorm > 0.0:
            qv = qn / qnorm
            # Image-vector scores
            image_scores = [0.0] * len(records)
            matrix, idxs = self._normalized_matrix(len(qvec))
            if matrix is not None:
                sims = matrix @ qv
                for pos, i in enumerate(idxs):
                    image_scores[i] = float(sims[pos])
            else:
                # Mixed vector dims in the db: score mismatches individually.
                for i, rec in enumerate(records):
                    image_scores[i] = cosine(qvec, rec.get("vector") or rec.get("embedding") or [])
            # Text-vector scores (frames indexed with tags via index_entry)
            text_scores = self._text_scores(records, qvec)
            # Fuse: max of image and text similarity — a strong match on
            # either representation ranks the frame.
            scores = [max(im, tx) for im, tx in zip(image_scores, text_scores)]

        order = sorted(range(len(records)), key=lambda i: scores[i], reverse=True)
        return [
            SearchResult(id=records[i]["id"], score=scores[i], metadata=records[i].get("metadata", {}))
            for i in order[:top_k]
        ]

    def _text_scores(self, records: list[dict], qvec: list[float]) -> list[float]:
        """Cosine of the query vs each record's stored text vector (0 if none)."""
        scored: list[float] = [0.0] * len(records)
        by_dim: dict[int, list[tuple[int, list[float]]]] = {}
        for i, rec in enumerate(records):
            tvec = rec.get("metadata", {}).get("text_vector") or rec.get("text_embedding")
            if tvec:
                by_dim.setdefault(len(tvec), []).append((i, tvec))
        for dim, items in by_dim.items():
            if dim != len(qvec):
                continue
            matrix = np.asarray([v for _, v in items], dtype=np.float32)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            sims = (matrix / norms) @ (np.asarray(qvec, dtype=np.float32) / max(np.linalg.norm(qvec), 1e-12))
            for (i, _), s in zip(items, sims):
                scored[i] = float(s)
        return scored
