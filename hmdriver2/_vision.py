# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class MatchResult:
    score: float
    # top-left in screenshot pixels
    x: int
    y: int
    w: int
    h: int

    @property
    def center(self) -> Tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h // 2


def _require_cv2():
    try:
        import cv2  # type: ignore

        return cv2
    except Exception as e:
        raise RuntimeError(
            "OpenCV is required for vision features. "
            'Install with `pip install -U \"hmdriver2[opencv-python]\"` '
            "(opencv-python-headless)."
        ) from e


def find_image(
    screenshot_path: str,
    template_path: str,
    threshold: float = 0.85,
    grayscale: bool = True,
) -> Optional[MatchResult]:
    """
    Find *template_path* in *screenshot_path* using OpenCV template matching.
    Returns the best match (single) if its score >= threshold, else None.
    """
    cv2 = _require_cv2()

    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(screenshot_path, flag)
    tpl = cv2.imread(template_path, flag)
    if img is None:
        raise FileNotFoundError(f"cannot read screenshot: {screenshot_path}")
    if tpl is None:
        raise FileNotFoundError(f"cannot read template: {template_path}")

    ih, iw = img.shape[:2]
    th, tw = tpl.shape[:2]
    if tw <= 0 or th <= 0 or tw > iw or th > ih:
        return None

    res = cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED)
    _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(res)
    if float(max_val) < float(threshold):
        return None
    x, y = int(max_loc[0]), int(max_loc[1])
    return MatchResult(score=float(max_val), x=x, y=y, w=int(tw), h=int(th))


def find_color(
    screenshot_path: str,
    rgb: Tuple[int, int, int],
    tolerance: int = 10,
    region: Optional[Tuple[int, int, int, int]] = None,
) -> Optional[Tuple[int, int]]:
    """
    Find the first pixel matching rgb within tolerance.

    Args:
        screenshot_path: screenshot file path.
        rgb: target color in RGB.
        tolerance: per-channel absolute tolerance (0-255).
        region: optional (x1, y1, x2, y2) in screenshot pixels.

    Returns:
        (x, y) in screenshot pixels if found, else None.
    """
    cv2 = _require_cv2()
    import numpy as np  # type: ignore

    img = cv2.imread(screenshot_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read screenshot: {screenshot_path}")

    # OpenCV uses BGR
    bgr = np.array([rgb[2], rgb[1], rgb[0]], dtype=np.int16)
    tol = int(max(0, min(255, tolerance)))

    if region is not None:
        x1, y1, x2, y2 = region
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = max(x1 + 1, int(x2))
        y2 = max(y1 + 1, int(y2))
        img2 = img[y1:y2, x1:x2]
        offset = (x1, y1)
    else:
        img2 = img
        offset = (0, 0)

    arr = img2.astype(np.int16)
    lo = bgr - tol
    hi = bgr + tol
    mask = (
        (arr[:, :, 0] >= lo[0])
        & (arr[:, :, 0] <= hi[0])
        & (arr[:, :, 1] >= lo[1])
        & (arr[:, :, 1] <= hi[1])
        & (arr[:, :, 2] >= lo[2])
        & (arr[:, :, 2] <= hi[2])
    )
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    x = int(xs[0]) + offset[0]
    y = int(ys[0]) + offset[1]
    return x, y

