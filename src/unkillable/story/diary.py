"""Nightwatch Diary: compose the day's text diary and index it for search.

The diary lives under ``storage/story/diary/YYYY-MM-DD.txt`` and is re-embedded
into ``storage/story/diary_index.jsonl`` (same JSONL schema as the main index)
so the existing cosine pipeline can query it.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from unkillable.utils.logging_config import get_logger

log = get_logger(__name__)

NO_EVENTS_LINE = "No events recorded."


def build_diary_text(date_obj: date, ordered_lines: list, per_camera_total: dict) -> str:
    """Assemble the diary: a per-camera header plus chronological lines."""
    header = [f"# Nightwatch Diary — {date_obj.isoformat()}"]
    for camera in sorted(per_camera_total):
        n = per_camera_total[camera]
        header.append(f"- {camera}: {n} event{'s' if n != 1 else ''}")
    header.append("")
    body = [line for _event_id, line in ordered_lines] or [NO_EVENTS_LINE]
    return "\n".join(header + body) + "\n"


def write_diary(diary_dir: Path, date_obj: date, text: str) -> Path:
    """Write (atomically) one day's diary; returns its path."""
    diary_dir.mkdir(parents=True, exist_ok=True)
    path = diary_dir / f"{date_obj.isoformat()}.txt"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return path


def diary_lines(diary_dir: Path) -> list[tuple[str, str]]:
    """``(doc_id, text)`` for every body line of every diary file, in order.

    Header lines (``#``/``- ``) are skipped; each line gets a stable doc id
    so a full re-index is deterministic.
    """
    out: list[tuple[str, str]] = []
    if not diary_dir.exists():
        return out
    for path in sorted(diary_dir.glob("*.txt")):
        date_str = path.stem
        content = path.read_text(encoding="utf-8")
        for i, line in enumerate(content.splitlines()):
            if not line.strip() or line.startswith(("#", "- ")):
                continue
            out.append((f"diary:{date_str}:{i:04d}", line))
    return out


def rebuild_index(diary_dir: Path, index_path: Path, search=None) -> int:
    """Re-embed all diary lines into a fresh JSONL (idempotent per run).

    Reuses the existing record schema (``id``/``vector``/``metadata`` with a
    ``text_vector``) so the same SemanticSearch pipeline can query the diary.
    """
    from unkillable.semantic.search import SemanticSearch

    search = search or SemanticSearch(db_path=index_path)
    lines = diary_lines(diary_dir)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    if not lines:
        index_path.write_text("", encoding="utf-8")
        return 0

    doc_ids = [doc_id for doc_id, _ in lines]
    texts = [text for _, text in lines]
    vectors = search.embed_texts(texts)

    tmp = index_path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for doc_id, text, vec in zip(doc_ids, texts, vectors):
            record = {
                "id": doc_id,
                "vector": vec,
                "metadata": {"text_vector": vec, "kind": "diary", "text": text},
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp.replace(index_path)
    return len(doc_ids)