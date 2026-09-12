import pytest


@pytest.fixture(autouse=True)
def _no_yolo_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent test runs from downloading YOLO weights; force heuristic mode."""
    from unkillable.detection import detector as detector_module

    monkeypatch.setattr(detector_module, "SKIP_MODEL_DOWNLOAD", True)