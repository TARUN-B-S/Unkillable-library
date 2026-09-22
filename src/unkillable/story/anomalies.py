"""Suspicious-Pattern Copilot: baselines plus deviation flags.

Baselines are per ``(camera, hour, label)`` over a trailing window of days.
Two deterministic flags — both require a non-trivial baseline so sparse data
can't false-positive:

* **surge**   — today's count >= max(min_observed, baseline mean * 2)
* **missing** — baseline mean >= 1 and today's count == 0
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from unkillable.utils.logging_config import get_logger

log = get_logger(__name__)

DEFAULT_WINDOW_DAYS = 14
MIN_BASELINE = 1.0  # a label must normally appear this often/day to matter
MIN_OBSERVED = 3  # surges below this are noise


@dataclass
class Anomaly:
    camera: str
    hour: int
    label: str
    kind: str  # "surge" | "missing"
    observed: float
    baseline: float
    message: str

    def to_dict(self) -> dict:
        return {
            "camera": self.camera,
            "hour": self.hour,
            "label": self.label,
            "kind": self.kind,
            "observed": self.observed,
            "baseline": round(self.baseline, 3),
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Anomaly:
        return cls(**d)


def _per_day_key_counts(entries) -> dict[str, dict]:
    """{iso_date: {(camera, hour, label): frame-count}} from entry wall times."""
    per_day: dict[str, dict] = {}
    for e in entries:
        if not e.wall_time:
            continue
        dt = datetime.fromtimestamp(e.wall_time)
        day = per_day.setdefault(dt.date().isoformat(), {})
        for label in e.labels or []:
            key = (e.camera, dt.hour, label)
            day[key] = day.get(key, 0) + 1
    return per_day


def detect_anomalies(entries, target: date, window_days: int = DEFAULT_WINDOW_DAYS) -> list[Anomaly]:
    """Flag surge/missing deviations for ``target`` vs the trailing window."""
    per_day = _per_day_key_counts(entries)
    target_iso = target.isoformat()
    today = per_day.get(target_iso, {})

    baseline_sum: dict = {}
    baseline_n: dict = {}
    for off in range(1, window_days + 1):
        day = per_day.get((target - timedelta(days=off)).isoformat())
        if not day:
            continue
        for key, count in day.items():
            baseline_sum[key] = baseline_sum.get(key, 0.0) + count
            baseline_n[key] = baseline_n.get(key, 0) + 1

    flagged: list[Anomaly] = []
    keys = set(today) | set(baseline_sum)
    for camera, hour, label in sorted(keys, key=lambda k: (k[0], k[2], k[1])):
        observed = float(today.get((camera, hour, label), 0.0))
        mean = baseline_sum.get(
            (camera, hour, label), 0.0
        ) / max(1, baseline_n.get((camera, hour, label), 0))
        if mean < MIN_BASELINE:
            continue  # not enough history for this key — stay quiet
        if observed >= max(MIN_OBSERVED, mean * 2.0):
            flagged.append(
                Anomaly(
                    camera, hour, label, "surge", observed, mean,
                    f"unusually many {label} sightings at {hour:02d}:00 — "
                    f"{observed:.0f} today vs ~{mean:.1f}/day",
                )
            )
        elif observed == 0.0 and mean >= MIN_BASELINE:
            flagged.append(
                Anomaly(
                    camera, hour, label, "missing", observed, mean,
                    f"no {label} sightings at {hour:02d}:00 — usually ~{mean:.1f}/day",
                )
            )
    flagged.sort(key=lambda a: (a.camera, a.hour, a.label))
    return flagged


def event_anomalies(anomalies: list[Anomaly], camera: str, event_hour: int, labels) -> list[str]:
    """Anomaly messages matching an event's (camera, hour, label) keys."""
    by_key = {(a.camera, a.hour, a.label): a for a in anomalies}
    msgs = []
    for label in labels:
        anomaly = by_key.get((camera, event_hour, label))
        if anomaly:
            msgs.append(anomaly.message)
    return msgs


def load(path: Path | str) -> list[Anomaly]:
    path = Path(path)
    if not path.exists():
        return []
    anomalies: list[Anomaly] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                anomalies.append(Anomaly.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError, KeyError):
                log.warning("Skipping corrupt anomaly line in %s", path)
    except OSError as exc:
        log.error("Failed to read anomalies %s: %s", path, exc)
    return anomalies


def write(path: Path | str, anomalies: list[Anomaly]) -> Path:
    """Rewrite the whole anomalies file (idempotent per run)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(a.to_dict(), ensure_ascii=False) + "\n" for a in anomalies)
    tmp.replace(path)
    return path