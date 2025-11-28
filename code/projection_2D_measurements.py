#!/usr/bin/env python3
"""
Pixel-space triangle measurements for Tri-Bench.

Usage:
    python measure_triangle.py path/to/image.jpg

What it does:
- Opens the image.
- You click the centres of the coloured stickers in order:
    A = RED, B = YELLOW, C = BLUE.
- Draws an overlay with the triangle and labels and saves it to overlays/.
- Appends a row to measurements.csv with:
    - pixel coordinates of A, B, C
    - side lengths in pixels
    - interior angles in degrees
    - derived quantities matching the VLM prompt:
        side_type, angle_type,
        ab_over_ac, abs_b_minus_c_deg,
        max_over_min_side, angle_range_deg
"""

import os
import csv
import json
import math
import argparse

import cv2
import numpy as np
from PIL import Image, ImageOps

CSV_PATH = "measurements.csv"
OVERLAY_DIR = "overlays"

CSV_FIELDS = [
    "image", "img_w", "img_h",
    "Ax", "Ay", "Bx", "By", "Cx", "Cy",
    "AB_px", "BC_px", "CA_px",
    "A_deg", "B_deg", "C_deg",
    "side_type", "angle_type",
    "ab_over_ac", "abs_b_minus_c_deg",
    "max_over_min_side", "angle_range_deg",
]

# ~1.5% side-length tolerance for "equal"
TOL_EQ = 0.015
# degree tolerance around 90° for "right"
TOL_RIGHT = 1.0


def load_rgb(path):
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    im = im.convert("RGB")
    # convert to OpenCV BGR
    return np.array(im)[:, :, ::-1]


def dist(p, q):
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)
    return float(np.linalg.norm(p - q))


def triangle_angles(a, b, c):
    def safe_acos(x):
        x = max(-1.0, min(1.0, x))
        return math.degrees(math.acos(x))

    A = safe_acos((b * b + c * c - a * a) / (2 * b * c))
    B = safe_acos((a * a + c * c - b * b) / (2 * a * c))
    C = 180.0 - A - B
    return A, B, C


def side_type_from_lengths(AB, BC, CA):
    s = sorted([AB, BC, CA])
    eq01 = abs(s[0] - s[1]) <= TOL_EQ * max(s[0], s[1])
    eq12 = abs(s[1] - s[2]) <= TOL_EQ * max(s[1], s[2])
    if eq01 and eq12:
        return "equilateral"
    if eq01 or eq12:
        return "isosceles"
    return "scalene"


def angle_type_from_angles(A, B, C):
    mx = max(A, B, C)
    if abs(mx - 90.0) <= TOL_RIGHT:
        return "right"
    return "obtuse" if mx > 90.0 else "acute"


def fmt4(x):
    """Format as float with 4 decimals, returned as string."""
    return f"{float(x):.4f}"


def append_row(row_dict):
    write_header = not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row_dict)


def draw_label(img, pt, text, color_bgr):
    # black outline for readability
    cv2.putText(
        img, text, (pt[0] + 6, pt[1] - 6),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA
    )
    cv2.putText(
        img, text, (pt[0] + 6, pt[1] - 6),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 1, cv2.LINE_AA
    )


def click_points(img_bgr):
    clone = img_bgr.copy()
    pts = []
    win = "click A (RED), B (YELLOW), C (BLUE) – Esc=reset, Enter=accept"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    colors = [(0, 0, 255), (0, 255, 255), (255, 0, 0)]  # A,B,C
    labels = ["A", "B", "C"]

    def cb(event, x, y, flags, param):
        nonlocal pts
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 3:
            pts.append((x, y))

    cv2.setMouseCallback(win, cb)

    while True:
        frame = clone.copy()
        for i, p in enumerate(pts):
            col = colors[i]
            cv2.circle(frame, p, 7, col, -1)
            draw_label(frame, p, labels[i], col)
        if len(pts) == 3:
            cv2.line(frame, pts[0], pts[1], (255, 255, 255), 2)
            cv2.line(frame, pts[1], pts[2], (255, 255, 255), 2)
            cv2.line(frame, pts[2], pts[0], (255, 255, 255), 2)
        cv2.imshow(win, frame)
        k = cv2.waitKey(16) & 0xFF
        if k == 27:  # Esc → reset
            pts = []
        if len(pts) == 3 and k in (10, 13):  # Enter → accept
            cv2.destroyWindow(win)
            return pts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    args = parser.parse_args()

    os.makedirs(OVERLAY_DIR, exist_ok=True)

    img_bgr = load_rgb(args.image_path)
    h, w = img_bgr.shape[:2]

    # Click A, B, C
    A, B, C = click_points(img_bgr)

    # Side lengths
    AB = dist(A, B)
    BC = dist(B, C)
    CA = dist(C, A)

    # Angles at A,B,C (opposite BC, CA, AB)
    a, b, c = BC, CA, AB
    Adeg, Bdeg, Cdeg = triangle_angles(a, b, c)

    side_t = side_type_from_lengths(AB, BC, CA)
    angle_t = angle_type_from_angles(Adeg, Bdeg, Cdeg)

    ab_over_ac = AB / CA if CA > 0 else float("nan")
    abs_b_minus_c = abs(Bdeg - Cdeg)
    max_over_min = max(AB, BC, CA) / min(AB, BC, CA) if min(AB, BC, CA) > 0 else float("inf")
    angle_range = max(Adeg, Bdeg, Cdeg) - min(Adeg, Bdeg, Cdeg)

    # Overlay (for sanity-checking clicks)
    overlay = img_bgr.copy()
    for p, lbl, col in zip((A, B, C), ("A", "B", "C"), [(0, 0, 255), (0, 255, 255), (255, 0, 0)]):
        cv2.circle(overlay, p, 7, col, -1)
        draw_label(overlay, p, lbl, col)
    cv2.line(overlay, A, B, (255, 255, 255), 2)
    cv2.line(overlay, B, C, (255, 255, 255), 2)
    cv2.line(overlay, C, A, (255, 255, 255), 2)

    base = os.path.splitext(os.path.basename(args.image_path))[0]
    overlay_path = os.path.join(OVERLAY_DIR, f"{base}_overlay.png")
    cv2.imwrite(overlay_path, overlay)

    # CSV row (all numeric fields formatted to 4 decimals as strings)
    row = {
        "image": os.path.basename(args.image_path),
        "img_w": fmt4(w), "img_h": fmt4(h),
        "Ax": fmt4(A[0]), "Ay": fmt4(A[1]),
        "Bx": fmt4(B[0]), "By": fmt4(B[1]),
        "Cx": fmt4(C[0]), "Cy": fmt4(C[1]),
        "AB_px": fmt4(AB), "BC_px": fmt4(BC), "CA_px": fmt4(CA),
        "A_deg": fmt4(Adeg), "B_deg": fmt4(Bdeg), "C_deg": fmt4(Cdeg),
        "side_type": side_t,
        "angle_type": angle_t,
        "ab_over_ac": fmt4(ab_over_ac),
        "abs_b_minus_c_deg": fmt4(abs_b_minus_c),
        "max_over_min_side": fmt4(max_over_min),
        "angle_range_deg": fmt4(angle_range),
    }
    append_row(row)

    print(
        f"[OK] {row['image']} | "
        f"AB={AB:.1f}px BC={BC:.1f}px CA={CA:.1f}px "
        f"(overlay: {overlay_path})"
    )

    out = {
        "side_type": side_t,
        "angle_type": angle_t,
        "ab_over_ac": fmt4(ab_over_ac),
        "abs_b_minus_c_deg": fmt4(abs_b_minus_c),
        "max_over_min_side": fmt4(max_over_min),
        "angle_range_deg": fmt4(angle_range),
    }
    print(json.dumps(out, separators=(",", ":")))


if __name__ == "__main__":
    main()