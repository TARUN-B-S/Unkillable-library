"""Object color extraction: dominant color of a detection bbox region.

OpenCV reads images as BGR. Hue is 0-179. Achromatic pixels (black/white/gray)
are decided by saturation/value first; chromatic pixels fall into hue bins.
"""
import colorsys
from pathlib import Path

import cv2
import numpy as np

# OpenCV hue bins (0-179). Red wraps at the ends.
HUE_BINS = [
    (0, 8, "red"),
    (9, 24, "orange"),
    (25, 34, "yellow"),
    (35, 80, "green"),
    (81, 95, "cyan"),
    (96, 129, "blue"),
    (130, 170, "purple"),
    (171, 180, "red"),
]


def _load_image(image: Path | str | np.ndarray) -> np.ndarray:
    if isinstance(image, np.ndarray):
        img = image
        if img.ndim != 3:
            raise ValueError("Image array must be HxWx3 BGR")
        return img
    p = Path(image)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    img = cv2.imread(str(p))
    if img is None:
        raise ValueError(f"Could not read image: {p}")
    return img


def _to_int(v) -> int:
    return int(np.rint(v))


def _clip_bbox(bbox, w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (_to_int(v) for v in bbox)
    x1, x2 = sorted((max(0, x1), min(w, x2)))
    y1, y2 = sorted((max(0, y1), min(h, y2)))
    return x1, y1, x2, y2


def _hex_from_hsv(h, s, v) -> str:
    r, g, b = (int(c * 255) for c in colorsys.hsv_to_rgb(h / 180.0, s / 255.0, v / 255.0))
    return f"#{r:02x}{g:02x}{b:02x}"


def dominant_color(image: Path | str | np.ndarray, bbox) -> tuple[str, str | None]:
    """Return (color_name, hex) for the dominant color inside ``bbox``.

    :param image: BGR ndarray or a path to an image file.
    :param bbox: (x1, y1, x2, y2) in image pixel coordinates.
    :returns: e.g. ("red", "#ff0000"). hex is None when the region has no
        usable pixels (e.g. degenerate bbox).
    """
    img = _load_image(image)
    h, w = img.shape[:2]
    x1, y1, x2, y2 = _clip_bbox(bbox, w, h)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return ("unknown", None)

    crop = img[y1:y2, x1:x2]
    # Center-weight: the middle 80% of the bbox is more likely the object body
    # than the edges (background bleed, vignette, borders).
    ch, cw = crop.shape[:2]
    cx0, cy0 = max(0, cw // 10), max(0, ch // 10)
    cx1, cy1 = cw - cx0, ch - cy0
    crop = crop[cy0:cy1, cx0:cx1]
    if crop.size == 0:
        return ("unknown", None)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, sat, val = np.split(hsv, 3, axis=2)
    sat = sat.ravel().astype(np.int32)
    val = val.ravel().astype(np.int32)

    mean_s = int(sat.mean())
    mean_v = int(val.mean())

    if mean_v < 40:
        return ("black", _hex_from_hsv(0, 0, int(val.mean())))
    if mean_s < 25:
        # Achromatic: white/gray/black — still report a truthful hex.
        if mean_v >= 200:
            return ("white", _hex_from_hsv(0, 0, 255))
        return ("gray", _hex_from_hsv(0, 0, int(val.mean())))

    hue = hue.ravel().astype(np.int32)
    weights = sat * val
    best_bin = max(HUE_BINS, key=lambda b: int(weights[_bin_mask(hue, b[0], b[1])].sum()))
    name = best_bin[2]
    mask = _bin_mask(hue, best_bin[0], best_bin[1])
    if not mask.any():
        return (name, None)
    # Representative shade: the most frequent hue inside the dominant bin
    # (mode), averaged with the pixel's saturation/value — the argmax-weight
    # pixel alone is noise-prone.
    best_hue = int(np.bincount(hue[mask]).argmax())
    idx = (hue[mask] == best_hue)
    s_ = int(sat[mask][idx].mean())
    v_ = int(val[mask][idx].mean())
    return (name, _hex_from_hsv(best_hue, s_, v_))


def _bin_mask(hue: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Boolean mask of pixels whose hue is inside [lo, hi)."""
    if lo == 0:
        return hue < hi  # red wraps at the start: hue in [0, hi)
    if hi >= 180:
        return hue >= lo  # red wraps at the end: hue in [lo, 179]
    return (hue >= lo) & (hue < hi)