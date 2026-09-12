from pathlib import Path

import typer
from rich.console import Console

from unkillable.backup.qr_backup import QRBackup
from unkillable.backup.qr_restore import QRRestore
from unkillable.detection.detector import Detector
from unkillable.genai.describer import Describer
from unkillable.motion.engine import MotionEngine
from unkillable.semantic.search import SemanticSearch
from unkillable.utils.ffmpeg import FFmpegWrapper
from unkillable.utils.logging_config import get_logger, setup_logging
from unkillable.utils.storage import Storage

app = typer.Typer(help="The Unkillable Library — local NVR toolkit")
console = Console()
log = get_logger(__name__)


@app.callback()
def main(verbose: bool = False) -> None:
    setup_logging(level="DEBUG" if verbose else "INFO")


@app.command()
def motion(input: str = typer.Argument(..., help="RTSP URL or file path"), duration: int = 10, threshold: float = 0.02) -> None:
    eng = MotionEngine(threshold=threshold)
    try:
        events = eng.detect_via_ffmpeg(input, duration)
        console.print(f"[green]Motion events: {len(events)}[/green]")
        for e in events:
            console.print(f"  score={e.score:.3f} ts={e.start_ts}")
    except Exception as exc:
        console.print(f"[red]Motion failed: {exc}[/red]")
        raise typer.Exit(1)


@app.command()
def thumbnail(input: str, output: str = "storage/thumbnails/thumb.jpg", timestamp: str = "00:00:01") -> None:
    fw = FFmpegWrapper()
    try:
        p = fw.extract_thumbnail(input, Path(output), timestamp)
        console.print(f"[green]Thumbnail: {p}[/green]")
    except Exception as exc:
        console.print(f"[red]Thumbnail failed: {exc}[/red]")
        raise typer.Exit(1)


@app.command()
def detect(
    image: str,
    labels: str = "person,car,dog",
    model: str = typer.Option(None, help="YOLO model file name (default: yolov8n.pt)"),
) -> None:
    det = Detector(track_labels=labels.split(","), model_name=model)
    try:
        results = det.detect(Path(image))
        counts = Detector.counts(results)
        console.print(f"[green]Detections: {len(results)}[/green]")
        for r in results:
            color = f" {r.color_name} ({r.color_hex})" if r.color_name else ""
            console.print(f"  {r.label} {r.confidence:.2f} {r.bbox}{color}")
        if counts:
            console.print(f"[cyan]Counts: {counts}[/cyan]")
    except Exception as exc:
        console.print(f"[red]Detect failed: {exc}[/red]")
        raise typer.Exit(1)


@app.command()
def search(query: str, top_k: int = 5, db: str = "storage/embeddings/db.jsonl") -> None:
    ss = SemanticSearch(db_path=Path(db))
    try:
        results = ss.search(query, top_k=top_k)
        if not results:
            console.print("[yellow]No results (DB empty or no match)[/yellow]")
            return
        for r in results:
            console.print(f"[cyan]{r.id}[/cyan] score={r.score:.3f} {r.metadata}")
    except Exception as exc:
        console.print(f"[red]Search failed: {exc}[/red]")
        raise typer.Exit(1)


@app.command()
def describe(image: str, ollama_url: str = "http://localhost:11434") -> None:
    d = Describer(base_url=ollama_url)
    try:
        text = d.describe(Path(image))
        console.print(f"[green]{text}[/green]")
    except Exception as exc:
        console.print(f"[red]Describe failed: {exc}[/red]")
        raise typer.Exit(1)


@app.command()
def backup(sources: str = typer.Argument(..., help="Comma-separated paths to backup"), output: str = "storage/backup.pdf") -> None:
    paths = [Path(s.strip()) for s in sources.split(",")]
    tmp_archive = Path("storage/tmp_backup.tar.gz")
    qb = QRBackup()
    try:
        archive = qb.create_archive(paths, tmp_archive)
        pdf = qb.encode_to_qr_pdf(archive, Path(output))
        console.print(f"[green]Backup PDF: {pdf}[/green]")
    except Exception as exc:
        console.print(f"[red]Backup failed: {exc}[/red]")
        raise typer.Exit(1)


@app.command()
def restore(images: str = typer.Argument(..., help="Comma-separated image paths or glob"), output: str = "storage/restored.tar.gz") -> None:
    import glob as globmod

    expanded: list[Path] = []
    for pat in images.split(","):
        pat = pat.strip()
        expanded.extend(Path(p) for p in globmod.glob(pat))
        if Path(pat).exists():
            expanded.append(Path(pat))
    qr = QRRestore()
    try:
        archive = qr.decode_from_images(expanded, Path(output))
        console.print(f"[green]Restored: {archive}[/green]")
    except Exception as exc:
        console.print(f"[red]Restore failed: {exc}[/red]")
        raise typer.Exit(1)


@app.command()
def storage_info(root: str = "storage") -> None:
    s = Storage(Path(root))
    s.ensure_dirs()
    console.print(s.disk_usage())
    console.print(f"Events: {len(s.load_events(limit=1000))}")


# ── Video Search Commands ──────────────────────────────────────────

@app.command()
def index(
    clip: str = typer.Argument(..., help="Path to video clip or RTSP URL to index"),
    camera: str = "default",
    storage_root: str = "storage",
    motion_threshold: float = 0.02,
) -> None:
    """Index a video clip: extract keyframes, detect objects, embed for search."""
    from unkillable.index.engine import IndexEngine

    ie = IndexEngine(storage_root=Path(storage_root), motion_threshold=motion_threshold)
    try:
        entries = ie.index_clip(Path(clip), camera=camera)
        console.print(f"[green]Indexed {len(entries)} frames from {clip}[/green]")
        for e in entries[:10]:
            counts_str = ", ".join(f"{k}={v}" for k, v in e.counts.items()) or "none"
            console.print(f"  [cyan]{e.id}[/cyan] t={e.timestamp_str} counts=[{counts_str}]")
        if len(entries) > 10:
            console.print(f"  ... and {len(entries) - 10} more")
        if entries:
            clip_totals = IndexEngine.clip_counts(entries)[str(Path(clip).resolve())]
            if clip_totals:
                console.print(f"[green]Clip totals: {clip_totals}[/green]")
    except Exception as exc:
        console.print(f"[red]Index failed: {exc}[/red]")
        raise typer.Exit(1)


@app.command()
def query(
    text: str = typer.Argument(..., help="Natural language query"),
    top_k: int = 5,
    labels: str = "",
    storage_root: str = "storage",
    clip_duration: float = 5.0,
    no_clips: bool = False,
) -> None:
    """Search indexed video frames by natural language query."""
    from unkillable.index.engine import IndexEngine
    from unkillable.query.engine import QueryEngine

    ie = IndexEngine(storage_root=Path(storage_root))
    qe = QueryEngine(index_engine=ie, storage_root=Path(storage_root), clip_duration=clip_duration)
    label_list = [l.strip() for l in labels.split(",") if l.strip()] or None

    try:
        results = qe.query(text, top_k=top_k, labels=label_list, generate_clips=not no_clips)
        if not results:
            console.print("[yellow]No results found[/yellow]")
            return
        console.print(f"[green]{len(results)} results for '{text}'[/green]")
        for r in results:
            labels_str = ", ".join(r.entry.labels) if r.entry.labels else "none"
            clip_info = f" clip={r.clip_path}" if r.clip_path else ""
            console.print(
                f"  [cyan]{r.entry.id}[/cyan] t={r.entry.timestamp_str} "
                f"score={r.score:.3f} labels=[{labels_str}]{clip_info}"
            )
    except Exception as exc:
        console.print(f"[red]Query failed: {exc}[/red]")
        raise typer.Exit(1)


@app.command()
def serve(
    host: str = "0.0.0.0",
    port: int = 5001,
    storage_root: str = "storage",
    debug: bool = False,
) -> None:
    """Start the search API server for the dashboard."""
    from unkillable.api import app, init_engines

    init_engines(storage_root=Path(storage_root))
    console.print(f"[green]Starting API server on {host}:{port}[/green]")
    console.print(f"Dashboard: http://localhost:{port}")
    console.print(f"Search API: http://localhost:{port}/api/search?q=your+query")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    app()
