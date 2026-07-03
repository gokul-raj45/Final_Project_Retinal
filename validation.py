"""
validation.py
--------------
Lightweight, model-free validation that checks whether an uploaded image
is a retinal fundus photograph *before* it is sent to the existing
ResNet50 + DenseNet121 hybrid disease-prediction pipeline.

This module uses classical OpenCV image-processing heuristics only.
It does NOT load, modify, retrain, or otherwise touch the disease
prediction models — it is a completely independent pre-filter.

Heuristics used (a real fundus photo satisfies all of these):
1. A circular field-of-view (FOV) — fundus cameras always produce a
   circular illuminated region, detected via Hough-circle transform
   with a threshold-based fallback for images where Hough fails.
2. A warm (orange/red/yellow) color profile inside that circular
   region — caused by retinal tissue and blood vessels.
3. A near-black border surrounding the circular FOV — the unlit area
   outside the camera's optical field.

Random photos (faces, animals, objects, screenshots, landscapes, etc.)
essentially never satisfy all three simultaneously, so this gives a
robust, dependency-free filter without needing a second trained model.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------
# Step 1: Locate the circular field-of-view (FOV)
# ---------------------------------------------------------------------
def _find_fov_circle(gray_img):
    """Try Hough Circle Transform to find the fundus FOV circle."""
    blurred = cv2.medianBlur(gray_img, 5)
    h, w = gray_img.shape[:2]
    min_dim = min(h, w)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dim // 2,
        param1=50,
        param2=30,
        minRadius=int(min_dim * 0.25),
        maxRadius=int(min_dim * 0.60),
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # Prefer the circle closest to the image center (most likely
        # to be the real optical FOV rather than a coincidental shape).
        cx_img, cy_img = w // 2, h // 2
        circles = sorted(
            circles,
            key=lambda c: (c[0] - cx_img) ** 2 + (c[1] - cy_img) ** 2,
        )
        x, y, r = circles[0]
        return int(x), int(y), int(r)

    return None


def _fov_from_threshold(gray_img):
    """
    Fallback FOV detector for cases where Hough fails.
    Fundus images generally show one large bright circular blob on a
    near-black background, so thresholding + largest-contour analysis
    approximates the FOV circle.
    """
    _, thresh = cv2.threshold(gray_img, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    img_area = gray_img.shape[0] * gray_img.shape[1]

    # Reject blobs that are implausibly small or that fill the whole frame
    if area < img_area * 0.15 or area > img_area * 0.95:
        return None

    (x, y), r = cv2.minEnclosingCircle(largest)

    # Circularity check: how well does the contour fill its enclosing circle?
    circle_area = np.pi * (r ** 2)
    fill_ratio = area / circle_area if circle_area > 0 else 0
    if fill_ratio < 0.55:
        return None

    return int(x), int(y), int(r)


# ---------------------------------------------------------------------
# Step 2: Color-profile check inside the FOV
# ---------------------------------------------------------------------
def _color_profile_ok(bgr_img, mask):
    """
    Fundus photographs are dominated by warm orange/red tones.
    Checks the mean R/G/B relationship inside the circular FOV mask.
    """
    b, g, r = cv2.split(bgr_img)
    region = mask > 0

    if region.sum() < 500:
        return False, {}

    mean_r = float(np.mean(r[region]))
    mean_g = float(np.mean(g[region]))
    mean_b = float(np.mean(b[region]))

    stats = {"mean_r": mean_r, "mean_g": mean_g, "mean_b": mean_b}

    warm_dominant = mean_r > mean_g and mean_g >= mean_b * 0.85
    strong_red_bias = (mean_r - mean_b) > 15

    return (warm_dominant and strong_red_bias), stats


# ---------------------------------------------------------------------
# Step 3: Dark-border check outside the FOV
# ---------------------------------------------------------------------
def _border_is_dark(gray_img, mask):
    """
    Real fundus camera output almost always has a black/near-black
    border surrounding the circular FOV. Ordinary photos fill the
    entire frame, so this border will not be dark.
    """
    outside = mask == 0
    if outside.sum() < 500:
        # Circle covers almost the entire image -> nothing to compare,
        # treat neutrally rather than penalizing.
        return True

    mean_outside = float(np.mean(gray_img[outside]))
    return mean_outside < 40


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------
def is_retinal_fundus_image(image_path, debug=False):
    """
    Validate whether the image at `image_path` looks like a retinal
    fundus photograph.

    Returns:
        (is_valid: bool, info: dict)  -- info is useful for logging/debugging
    """
    img = cv2.imread(image_path)
    if img is None:
        return False, {"reason": "unreadable_file"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    circle = _find_fov_circle(gray) or _fov_from_threshold(gray)
    if circle is None:
        return False, {"reason": "no_circular_fov_detected"}

    x, y, r = circle
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (x, y), r, 255, -1)

    color_ok, color_stats = _color_profile_ok(img, mask)
    border_ok = _border_is_dark(gray, mask)

    # Require BOTH the color profile and dark border checks to pass.
    score = int(color_ok) + int(border_ok)
    is_valid = score >= 2

    info = {
        "circle": circle,
        "color_ok": color_ok,
        "border_ok": border_ok,
        "color_stats": color_stats,
        "score": score,
    }

    if debug:
        print("[validation]", info)

    return is_valid, info
