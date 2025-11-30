#!/usr/bin/env python3
"""
homography_example.py

Interactively select the square border corners and triangle vertices A, B, C
for a single image, compute a homography that maps the square to a 1x1 board,
and then derive the ground truth quantities for Q1-Q6.

"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

TOL_SIDE = 0.03
TOL_RIGHT_DEG = 2.0


def compute_homography(src_pts, dst_pts):
    """
    Compute homography H (3x3) such that dst ~ H * src,
    using the standard DLT formulation with SVD.

    src_pts, dst_pts: shape (4, 2), float
    """
    assert src_pts.shape == (4, 2)
    assert dst_pts.shape == (4, 2)

    A = []
    for (x, y), (X, Y) in zip(src_pts, dst_pts):
        A.append([-x, -y, -1, 0, 0, 0, X * x, X * y, X])
        A.append([0, 0, 0, -x, -y, -1, Y * x, Y * y, Y])

    A = np.array(A, dtype=float)
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1, :]
    H = h.reshape(3, 3)
    return H


def apply_homography(H, pts):
    """
    Apply homography H to an array of 2D points.

    pts: shape (N, 2)
    Returns: shape (N, 2)
    """
    pts = np.asarray(pts, dtype=float)
    n = pts.shape[0]
    pts_h = np.hstack([pts, np.ones((n, 1))])
    mapped = (H @ pts_h.T).T
    mapped = mapped[:, :2] / mapped[:, 2:3]
    return mapped


def side_lengths(pts_board):
    """
    Given A, B, C in board coordinates, return side lengths AB, BC, CA.
    """
    A, B, C = pts_board
    AB = np.linalg.norm(B - A)
    BC = np.linalg.norm(C - B)
    CA = np.linalg.norm(A - C)
    return AB, BC, CA


def angle_at(p_prev, p_vertex, p_next):
    """
    Angle at p_vertex (in degrees) formed by p_prev -- p_vertex -- p_next.
    """
    v1 = p_prev - p_vertex
    v2 = p_next - p_vertex
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0.0
    cosang = np.dot(v1, v2) / denom
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def classify_side_type(AB, BC, CA, tol=TOL_SIDE):
    """
    Classify triangle as 'equilateral', 'isosceles', or 'scalene'
    using the same tolerance rules as the dataset.

    Let (a, b, c) be the side lengths (unsorted is fine; we look at all pairs).
    - Equilateral: all pairwise relative differences <= tol.
    - Isosceles:   min pairwise relative difference <= tol (but not equilateral).
    - Else:        scalene.
    """
    sides = np.array([AB, BC, CA], dtype=float)
    ixs = [(0, 1), (1, 2), (0, 2)]
    rel_diffs = []
    for i, j in ixs:
        s1, s2 = sides[i], sides[j]
        rel = abs(s1 - s2) / max(s1, s2)
        rel_diffs.append(rel)
    rel_diffs = np.array(rel_diffs)

    if np.all(rel_diffs <= tol):
        return "equilateral"
    elif np.min(rel_diffs) <= tol:
        return "isosceles"
    else:
        return "scalene"


def classify_angle_type(angles_deg, tol_right=TOL_RIGHT_DEG):
    """
    Given [A, B, C] in degrees, classify as 'acute', 'right', or 'obtuse'.

    - Right: if any angle is within tol_right of 90 degrees.
    - Else Obtu se: if max angle > 90.
    - Else Acute.
    """
    angles = np.array(angles_deg, dtype=float)
    if np.any(np.abs(angles - 90.0) <= tol_right):
        return "right"
    elif np.max(angles) > 90.0:
        return "obtuse"
    else:
        return "acute"


def compute_triangle_quantities(board_pts):
    """
    Given A, B, C in board coordinates, compute all quantities for Q1--Q6.

    Returns:
      side_type, angle_type,
      ab_over_ac, abs_b_minus_c_deg,
      max_over_min_side, angle_range_deg
    """

    AB, BC, CA = side_lengths(board_pts)

    A_pt, B_pt, C_pt = board_pts
    angle_A = angle_at(B_pt, A_pt, C_pt)
    angle_B = angle_at(A_pt, B_pt, C_pt)
    angle_C = angle_at(A_pt, C_pt, B_pt)

    side_type = classify_side_type(AB, BC, CA)
    angle_type = classify_angle_type([angle_A, angle_B, angle_C])

    ab_over_ac = AB / CA if CA != 0 else np.nan

    abs_b_minus_c_deg = abs(angle_B - angle_C)

    max_over_min_side = max(AB, BC, CA) / min(AB, BC, CA)

    angles = np.array([angle_A, angle_B, angle_C])
    angle_range_deg = float(np.max(angles) - np.min(angles))

    return (side_type, angle_type,
            ab_over_ac, abs_b_minus_c_deg,
            max_over_min_side, angle_range_deg)


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 homography_example.py path/to/image.jpg")
        sys.exit(1)

    img_path = sys.argv[1]
    img = imread(img_path)

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title(
        "Click 4 square corners (TL, TR, BR, BL), then A, B, C.\n"
        "Close window when done."
    )
    plt.axis("on")
    pts = plt.ginput(7, timeout=0)
    plt.close(fig)

    if len(pts) != 7:
        print(f"Expected 7 points, got {len(pts)}. Aborting.")
        sys.exit(1)

    pts = np.array(pts, dtype=float)
    square_img = pts[:4] 
    tri_img = pts[4:] 

    square_board = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ], dtype=float)

    H = compute_homography(square_img, square_board)

    tri_board = apply_homography(H, tri_img)

    print("\nImage-space coordinates (pixels):")
    print(f"Square corners (TL, TR, BR, BL):\n{square_img}")
    print(f"Triangle A, B, C:\n{tri_img}")

    print("\nBoard-space coordinates (after homography):")
    print(f"A, B, C:\n{tri_board}")

    (side_type, angle_type,
     ab_over_ac, abs_b_minus_c_deg,
     max_over_min_side, angle_range_deg) = compute_triangle_quantities(tri_board)

    print("\nDerived quantities (unrounded):")
    print(f"side_type:           {side_type}")
    print(f"angle_type:          {angle_type}")
    print(f"AB/AC:               {ab_over_ac}")
    print(f"|∠ABC - ∠ACB| (deg): {abs_b_minus_c_deg}")
    print(f"max/min side:        {max_over_min_side}")
    print(f"angle range (deg):   {angle_range_deg}")

    result = {
        "side_type": side_type,
        "angle_type": angle_type,
        "ab_over_ac": round(ab_over_ac, 4),
        "abs_b_minus_c_deg": round(abs_b_minus_c_deg, 4),
        "max_over_min_side": round(max_over_min_side, 4),
        "angle_range_deg": round(angle_range_deg, 4),
    }

    print("\nJSON (4 decimals, matching Q1--Q6 format):")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
