from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory

from unkillable.index.engine import IndexEngine
from unkillable.query.engine import QueryEngine
from unkillable.utils.logging_config import get_logger

log = get_logger(__name__)

app = Flask(__name__, static_folder=None)

# Initialized on first request
_index_engine: IndexEngine | None = None
_query_engine: QueryEngine | None = None
_storage_root: Path = Path("storage")


def init_engines(storage_root: Path | str = "storage") -> None:
    global _index_engine, _query_engine, _storage_root
    _storage_root = Path(storage_root)
    _index_engine = IndexEngine(storage_root=_storage_root)
    _query_engine = QueryEngine(index_engine=_index_engine, storage_root=_storage_root)


def _get_query_engine() -> QueryEngine:
    if _query_engine is None:
        init_engines()
    return _query_engine


def _get_index_engine() -> IndexEngine:
    if _index_engine is None:
        init_engines()
    return _index_engine


# ─── Dashboard ───────────────────────────────────────────────────────

@app.route("/")
def dashboard():
    """Serve the dashboard HTML."""
    dashboard_path = Path(__file__).parent.parent.parent / "dashboard" / "index.html"
    if dashboard_path.exists():
        return send_file(str(dashboard_path))
    return "<h1>Dashboard not found</h1>", 404


# ─── Search API ──────────────────────────────────────────────────────

@app.route("/api/search")
def api_search():
    """
    GET /api/search?q=red+car&labels=car,person&top_k=10&clips=true
    """
    query_text = request.args.get("q", "").strip()
    if not query_text:
        return jsonify({"error": "Missing query parameter 'q'"}), 400

    labels_raw = request.args.get("labels", "")
    labels = [l.strip() for l in labels_raw.split(",") if l.strip()] or None
    top_k = int(request.args.get("top_k", 10))
    generate_clips = request.args.get("clips", "true").lower() == "true"

    qe = _get_query_engine()
    results = qe.query(query_text, top_k=top_k, labels=labels, generate_clips=generate_clips)

    return jsonify({
        "query": query_text,
        "count": len(results),
        "results": [r.to_dict() for r in results],
    })


@app.route("/api/search/labels")
def api_search_labels():
    """
    GET /api/search/labels?labels=person,car&top_k=10
    """
    labels_raw = request.args.get("labels", "")
    labels = [l.strip() for l in labels_raw.split(",") if l.strip()]
    if not labels:
        return jsonify({"error": "Missing 'labels' parameter"}), 400

    top_k = int(request.args.get("top_k", 10))
    qe = _get_query_engine()
    results = qe.query_by_labels(labels, top_k=top_k)

    return jsonify({
        "labels": labels,
        "count": len(results),
        "results": [r.to_dict() for r in results],
    })


# ─── Clip API ────────────────────────────────────────────────────────

@app.route("/api/clip/<entry_id>")
def api_clip(entry_id: str):
    """
    GET /api/clip/<entry_id>?duration=5
    Generate and serve a short clip around the matched frame.
    """
    duration = float(request.args.get("duration", 5.0))
    ie = _get_index_engine()
    entry = ie.get_entry(entry_id)
    if not entry:
        return jsonify({"error": "Entry not found"}), 404

    qe = _get_query_engine()
    clip_path = qe._generate_clip(entry, duration=duration)
    if not clip_path or not Path(clip_path).exists():
        return jsonify({"error": "Clip generation failed"}), 500

    return send_file(str(Path(clip_path).resolve()), mimetype="video/mp4")


# ─── Frame API ───────────────────────────────────────────────────────

@app.route("/api/frame/<entry_id>")
def api_frame(entry_id: str):
    """GET /api/frame/<entry_id> — serve the keyframe image."""
    ie = _get_index_engine()
    entry = ie.get_entry(entry_id)
    if not entry:
        return jsonify({"error": "Entry not found"}), 404

    thumb = Path(entry.thumbnail_path)
    if not thumb.is_absolute():
        thumb = Path(_get_index_engine().storage_root).parent / thumb
    if not thumb.exists():
        return jsonify({"error": "Thumbnail not found"}), 404

    return send_file(str(thumb.resolve()), mimetype="image/jpeg")


# ─── Describe API ────────────────────────────────────────────────────

@app.route("/api/describe/<entry_id>", methods=["POST"])
def api_describe(entry_id: str):
    """POST /api/describe/<entry_id> — GenAI description on-demand."""
    ie = _get_index_engine()
    entry = ie.get_entry(entry_id)
    if not entry:
        return jsonify({"error": "Entry not found"}), 404

    ollama_url = request.json.get("ollama_url", "http://localhost:11434") if request.is_json else "http://localhost:11434"
    qe = _get_query_engine()
    description = qe.describe_entry(entry, ollama_url=ollama_url)

    return jsonify({"id": entry_id, "description": description})


# ─── Entries API ─────────────────────────────────────────────────────

@app.route("/api/entries")
def api_entries():
    """
    GET /api/entries?limit=50&offset=0
    List indexed entries.
    """
    limit = int(request.args.get("limit", 50))
    offset = int(request.args.get("offset", 0))

    ie = _get_index_engine()
    all_entries = ie.load_entries(limit=limit + offset)
    page = all_entries[offset : offset + limit]

    return jsonify({
        "total": ie.index_count(),
        "offset": offset,
        "limit": limit,
        "count": len(page),
        "entries": [
            {
                "id": e.id,
                "timestamp": e.timestamp,
                "timestamp_str": e.timestamp_str,
                "source_clip": e.source_clip,
                "thumbnail_path": e.thumbnail_path,
                "labels": e.labels,
                "objects": e.objects,
                "counts": e.counts,
                "tags": e.tags,
                "colors": sorted({o.get("color_name") for o in e.objects if o.get("color_name")}),
                "motion_score": e.motion_score,
                "detection_confidence": e.detection_confidence,
                "metadata": e.metadata,
            }
            for e in page
        ],
    })


# ─── Stats API ───────────────────────────────────────────────────────

@app.route("/api/stats")
def api_stats():
    """GET /api/stats — index statistics."""
    ie = _get_index_engine()
    entries = ie.load_entries()

    all_counts: dict[str, int] = {}
    all_colors: dict[str, int] = {}
    all_tags: dict[str, int] = {}
    conf_sum = 0.0
    conf_frames = 0
    total_objects = 0
    for e in entries:
        for o in e.objects:
            color = o.get("color_name")
            if color:
                all_colors[color] = all_colors.get(color, 0) + 1
        for tag in e.tags:
            all_tags[tag] = all_tags.get(tag, 0) + 1
        if e.detection_confidence:
            conf_sum += e.detection_confidence
            conf_frames += 1
        if e.counts:
            for label, count in e.counts.items():
                all_counts[label] = all_counts.get(label, 0) + count
                total_objects += count
        else:
            for label in e.labels:
                all_counts[label] = all_counts.get(label, 0) + 1

    clips = {e.source_clip for e in entries}

    return jsonify({
        "total_frames": len(entries),
        "total_clips": len(clips),
        "total_objects": total_objects,
        "avg_detection_confidence": round(conf_sum / conf_frames, 4) if conf_frames else 0.0,
        "label_counts": all_counts,
        "color_counts": all_colors,
        "tag_counts": all_tags,
        "clip_counts": IndexEngine.clip_counts(entries),
        "index_file": str(ie.entries_file),
    })


# ─── Static files ────────────────────────────────────────────────────

@app.route("/dashboard/<path:filename>")
def serve_dashboard_static(filename: str):
    """Serve dashboard static assets."""
    dash_dir = Path(__file__).parent.parent.parent / "dashboard"
    return send_from_directory(str(dash_dir), filename)
