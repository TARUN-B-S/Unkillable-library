from dataclasses import dataclass
from pathlib import Path

from unkillable.utils.logging_config import get_logger

log = get_logger(__name__)

SUPPORTED_LABELS = ["person", "car", "truck", "bus", "dog", "cat", "bird", "bicycle", "motorcycle"]


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]
    frame_path: str | None = None


class DetectorError(RuntimeError):
    pass


class Detector:
    def __init__(self, model: str = "cpu", track_labels: list[str] | None = None, threshold: float = 0.5) -> None:
        self.model = model
        self.track_labels = track_labels or ["person", "car", "dog"]
        self.threshold = threshold
        self._yolo = None
        self._init_model()

    def _init_model(self) -> None:
        invalid = set(self.track_labels) - set(SUPPORTED_LABELS)
        if invalid:
            raise DetectorError(f"Unsupported labels: {invalid}. Supported: {SUPPORTED_LABELS}")
        try:
            import importlib.util

            if importlib.util.find_spec("ultralytics") is not None:
                from ultralytics import YOLO

                self._yolo = YOLO("yolov8n.pt")
                log.info("YOLO model loaded: %s", self.model)
            else:
                log.warning("ultralytics not installed, using heuristic detector")
        except Exception as exc:
            log.warning("Detector init fallback: %s", exc)
            self._yolo = None

    def detect(self, image_path: Path | str) -> list[Detection]:
        p = Path(image_path)
        if not p.exists():
            raise DetectorError(f"Image not found: {p}")
        if self._yolo is not None:
            try:
                results = self._yolo(str(p), verbose=False)
                detections: list[Detection] = []
                for r in results:
                    for box in r.boxes:
                        label = r.names[int(box.cls)]
                        conf = float(box.conf)
                        if label in self.track_labels and conf >= self.threshold:
                            x1, y1, x2, y2 = map(float, box.xyxy[0])
                            detections.append(Detection(label=label, confidence=conf, bbox=(x1, y1, x2, y2), frame_path=str(p)))
                log.info("Detected %d objects in %s", len(detections), p.name)
                return detections
            except Exception as exc:
                log.error("YOLO inference failed: %s", exc)
        return self._heuristic_detect(p)

    def _heuristic_detect(self, path: Path) -> list[Detection]:
        log.debug("Heuristic detect for %s — returning empty (no model)", path)
        return []

    def filter_by_label(self, detections: list[Detection]) -> list[Detection]:
        return [d for d in detections if d.label in self.track_labels and d.confidence >= self.threshold]
