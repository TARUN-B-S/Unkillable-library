import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from unkillable.detection.color import dominant_color
from unkillable.utils.logging_config import get_logger

log = get_logger(__name__)

SUPPORTED_LABELS = ["person", "car", "truck", "bus", "dog", "cat", "bird", "bicycle", "motorcycle"]

DEFAULT_YOLO_MODEL = os.environ.get("UNKILLABLE_YOLO_MODEL", "yolov8n.pt")
YOLO_FALLBACKS = ["yolov8n.pt", "yolo11n.pt"]
SKIP_MODEL_DOWNLOAD = os.environ.get("UNKILLABLE_SKIP_MODEL_DOWNLOAD", "") not in ("", "0", "false")


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]
    frame_path: str | None = None
    color_hex: str | None = None
    color_name: str | None = None
    size: str | None = None

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "bbox": [round(v, 2) for v in self.bbox],
            "area": round(self.area, 1),
            "color_hex": self.color_hex,
            "color_name": self.color_name,
            "size": self.size,
        }


def classify_size(bbox, frame_area: float) -> str:
    """Relative bbox coverage → coarse size class used for tagging/scoring."""
    x1, y1, x2, y2 = bbox
    ratio = (max(0.0, x2 - x1) * max(0.0, y2 - y1)) / frame_area if frame_area else 0.0
    if ratio >= 0.40:
        return "huge"
    if ratio >= 0.20:
        return "large"
    if ratio >= 0.08:
        return "medium"
    if ratio >= 0.02:
        return "small"
    return "tiny"


class DetectorError(RuntimeError):
    pass


class Detector:
    def __init__(
        self,
        model: str = "cpu",
        track_labels: list[str] | None = None,
        threshold: float = 0.5,
        model_name: str | None = None,
        imgsz: int = 640,
        iou_threshold: float = 0.45,
        min_area_ratio: float = 0.005,
    ) -> None:
        self.model = model
        self.track_labels = track_labels or ["person", "car", "dog"]
        self.threshold = threshold
        self.model_name = model_name or DEFAULT_YOLO_MODEL
        self.imgsz = imgsz
        self.iou_threshold = iou_threshold
        self.min_area_ratio = min_area_ratio
        self._yolo = None
        self._yolo_model = None
        self._init_model()

    def _init_model(self) -> None:
        invalid = set(self.track_labels) - set(SUPPORTED_LABELS)
        if invalid:
            raise DetectorError(f"Unsupported labels: {invalid}. Supported: {SUPPORTED_LABELS}")

        if SKIP_MODEL_DOWNLOAD:
            log.info("Model download disabled (UNKILLABLE_SKIP_MODEL_DOWNLOAD) — heuristic mode")
            return

        try:
            import importlib.util

            if importlib.util.find_spec("ultralytics") is None:
                log.warning("ultralytics not installed, using heuristic detector")
                return
            from ultralytics import YOLO

            candidates = [self.model_name] + [m for m in YOLO_FALLBACKS if m != self.model_name]
            for name in candidates:
                try:
                    self._yolo = YOLO(name)
                    self._yolo_model = name
                    log.info("YOLO model loaded: %s", name)
                    return
                except Exception as exc:
                    log.debug("Falling back from %s: %s", name, exc)
            log.warning("Could not load any YOLO model — using heuristic detector")
        except Exception as exc:
            log.warning("Detector init fallback: %s", exc)
            self._yolo = None

    def detect(self, image_path: Path | str) -> list[Detection]:
        p = Path(image_path)
        if not p.exists():
            raise DetectorError(f"Image not found: {p}")
        if self._yolo is not None:
            try:
                results = self._yolo(
                    str(p),
                    verbose=False,
                    conf=self.threshold,
                    iou=self.iou_threshold,
                    imgsz=self.imgsz,
                )
                detections: list[Detection] = []
                for r in results:
                    for box in r.boxes:
                        label = r.names[int(box.cls)]
                        conf = float(box.conf)
                        if label not in self.track_labels or conf < self.threshold:
                            continue
                        x1, y1, x2, y2 = map(float, box.xyxy[0])
                        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                        frame_area = float(r.orig_shape[0] * r.orig_shape[1])
                        if frame_area and area / frame_area < self.min_area_ratio:
                            continue  # drop tiny / speckle detections
                        size_name = classify_size((x1, y1, x2, y2), frame_area) if frame_area else None
                        detections.append(
                            Detection(
                                label=label,
                                confidence=conf,
                                bbox=(x1, y1, x2, y2),
                                frame_path=str(p),
                                size=size_name,
                            )
                        )
                self._annotate_colors(p, detections)
                log.info(
                    "Detected %d objects in %s", len(detections), p.name
                )
                return detections
            except Exception as exc:
                log.error("YOLO inference failed: %s", exc)
        return self._heuristic_detect(p)

    def _annotate_colors(self, image: Path | str | np.ndarray, detections: list[Detection]) -> None:
        """Fill color_hex/color_name on each detection from its bbox region."""
        img: Path | str | np.ndarray = image
        if not isinstance(image, np.ndarray):
            img = Path(image)
        for det in detections:
            try:
                det.color_name, det.color_hex = dominant_color(img, det.bbox)
            except Exception as exc:
                log.debug("Color extraction failed for %s: %s", det.label, exc)

    def _heuristic_detect(self, path: Path) -> list[Detection]:
        log.debug("Heuristic detect for %s — returning empty (no model)", path)
        return []

    def filter_by_label(self, detections: list[Detection]) -> list[Detection]:
        return [d for d in detections if d.label in self.track_labels and d.confidence >= self.threshold]

    @staticmethod
    def counts(detections: list[Detection] | None) -> dict[str, int]:
        """Count detected objects per label, e.g. {'person': 2, 'car': 1}."""
        tally: dict[str, int] = {}
        for d in detections or []:
            tally[d.label] = tally.get(d.label, 0) + 1
        return tally

    @staticmethod
    def mean_confidence(detections: list[Detection] | None) -> float:
        """Aggregate detection quality for a frame — mean confidence (0 if none)."""
        dets = list(detections or [])
        if not dets:
            return 0.0
        return round(sum(d.confidence for d in dets) / len(dets), 4)