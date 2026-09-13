import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

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
    return sum(x * y for x, y in zip(a, b))


class SemanticSearch:
    """Embedding search with a real encoder and a deterministic hash fallback.

    Mode is selected at construction time:
      * ``UNKILLABLE_EMBEDDING_MODE=auto`` (default): use sentence-transformers
        (CLIP by default so text queries match image frames); any import,
        download, or encode failure falls back to hash embeddings.
      * ``UNKILLABLE_EMBEDDING_MODE=hash``: always use hash embeddings (fast,
        offline, deterministic — used by the unit test suite).
    ``UNKILLABLE_EMBEDDING_MODEL`` overrides the model name.
    """

    def __init__(self, model: str | None = None, db_path: Path | None = None) -> None:
        self.model = os.environ.get("UNKILLABLE_EMBEDDING_MODEL") or model or _DEFAULT_EMBED_MODEL
        self.mode = os.environ.get("UNKILLABLE_EMBEDDING_MODE", "auto")
        self.db_path = Path(db_path) if db_path else Path("storage/embeddings/db.jsonl")
        self._encoder = None
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

    def embed_text(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("Text query must be non-empty")
        if self._encoder is not None:
            try:
                vec = self._encoder.encode([text]).tolist()[0]
                return vec
            except Exception as exc:
                log.error("Text embedding failed: %s", exc)
        return _hash_embedding(text)

    def embed_image(self, image_path: Path | str) -> list[float]:
        p = Path(image_path)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")
        if self._encoder is not None:
            try:
                from PIL import Image

                img = Image.open(p).convert("RGB")
                vec = self._encoder.encode(img).tolist()[0]
                return vec
            except Exception as exc:
                log.error("Image embedding failed: %s", exc)
        return _hash_embedding(str(p))

    def index(self, doc_id: str, vector: list[float], metadata: dict) -> None:
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            record = {"id": doc_id, "vector": vector, "metadata": metadata}
            with open(self.db_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            log.info("Indexed %s", doc_id)
        except OSError as exc:
            log.error("Index write failed: %s", exc)
            raise

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if not self.db_path.exists():
            log.warning("No embedding DB at %s", self.db_path)
            return []
        qvec = self.embed_text(query)
        results: list[SearchResult] = []
        try:
            with open(self.db_path, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    score = cosine(qvec, rec["vector"])
                    results.append(SearchResult(id=rec["id"], score=score, metadata=rec["metadata"]))
        except (OSError, json.JSONDecodeError) as exc:
            log.error("Search failed: %s", exc)
            raise
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]