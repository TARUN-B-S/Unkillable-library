import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from unkillable.utils.logging_config import get_logger

log = get_logger(__name__)


class StorageError(RuntimeError):
    pass


class Storage:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.clips = self.root / "clips"
        self.thumbnails = self.root / "thumbnails"
        self.embeddings = self.root / "embeddings"
        self.events_file = self.root / "events.jsonl"

    def ensure_dirs(self) -> None:
        try:
            for p in [self.clips, self.thumbnails, self.embeddings]:
                p.mkdir(parents=True, exist_ok=True)
            log.info("Storage dirs ready at %s", self.root)
        except OSError as exc:
            raise StorageError(f"Failed to create storage dirs: {exc}") from exc

    def save_event(self, event: dict) -> None:
        try:
            event.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
            with open(self.events_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except OSError as exc:
            raise StorageError(f"Failed to save event: {exc}") from exc

    def load_events(self, limit: int = 100) -> list[dict]:
        if not self.events_file.exists():
            return []
        events: list[dict] = []
        try:
            with open(self.events_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        events.append(json.loads(line))
        except (OSError, json.JSONDecodeError) as exc:
            log.error("Failed to load events: %s", exc)
            raise StorageError(f"Load failed: {exc}") from exc
        return events[-limit:]

    def disk_usage(self) -> dict:
        try:
            total, used, free = shutil.disk_usage(self.root)
            return {"total_gb": round(total / 1e9, 2), "used_gb": round(used / 1e9, 2), "free_gb": round(free / 1e9, 2)}
        except OSError as exc:
            log.warning("disk_usage failed: %s", exc)
            return {}

    def cleanup_old(self, days: int = 30) -> int:
        removed = 0
        cutoff = datetime.now(timezone.utc).timestamp() - days * 86400
        for path in self.root.rglob("*"):
            try:
                if path.is_file() and path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
            except OSError as exc:
                log.warning("Cleanup failed for %s: %s", path, exc)
        log.info("Cleanup removed %d files older than %d days", removed, days)
        return removed
