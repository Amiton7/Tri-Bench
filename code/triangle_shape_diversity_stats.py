#!/usr/bin/env python3
"""
Triangle shape diversity and 3D↔2D mismatch statistics.

- 3D: distribution over 100 physical triangles (majority label per triangle_id)
- 2D: distribution over 400 projected images (per row)
- Confusion matrices (rows=2D, cols=3D) for:
    * side_type: scalene / isosceles / equilateral
    * angle_type: acute / obtuse / right
"""

import pandas as pd
from pathlib import Path


def main():
    # repo root: .../github_export
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"

    tri3d = pd.read_csv(data_dir / "tri_bench_triangles_3d.csv")
    tri2d = pd.read_csv(data_dir / "tri_bench_pixel_geometry_2d.csv")

    # ---------- 3D diversity (100 physical triangles) ----------
    grouped_3d = tri3d.groupby("triangle_id").agg(
        side_type=("side_type", lambda s: s.mode().iat[0]),
        angle_type=("angle_type", lambda s: s.mode().iat[0]),
    )

    print("=== 3D diversity (per triangle_id, 100 triangles) ===")
    print("\nSide type:")
    print(grouped_3d["side_type"].value_counts().sort_index())
    print("\nAngle type:")
    print(grouped_3d["angle_type"].value_counts().sort_index())
    print()

    # ---------- 2D diversity (400 projected images) ----------
    print("=== 2D diversity (per image, 400 images) ===")
    print("\nSide type:")
    print(tri2d["side_type"].value_counts().sort_index())
    print("\nAngle type:")
    print(tri2d["angle_type"].value_counts().sort_index())
    print()

    # ---------- 3D - 2D confusion (per image) ----------
    merged = tri3d.merge(
        tri2d,
        on=["img_original", "triangle_id", "camera_view", "object_in_square"],
        suffixes=("_3d", "_2d"),
    )

    # Side-type confusion: rows = 2D, columns = 3D
    side_conf = pd.crosstab(merged["side_type_2d"], merged["side_type_3d"])
    side_conf = side_conf.reindex(
        index=["scalene", "isosceles", "equilateral"],
        columns=["scalene", "isosceles", "equilateral"],
    )

    # Angle-type confusion: rows = 2D, columns = 3D
    angle_conf = pd.crosstab(merged["angle_type_2d"], merged["angle_type_3d"])
    angle_conf = angle_conf.reindex(
        index=["acute", "obtuse", "right"],
        columns=["acute", "obtuse", "right"],
    )

    print("=== Side-type confusion (rows: 2D, cols: 3D) ===")
    print(side_conf.fillna(0).astype(int))
    print()

    print("=== Angle-type confusion (rows: 2D, cols: 3D) ===")
    print(angle_conf.fillna(0).astype(int))
    print()

    # Mismatch counts
    side_conf_filled = side_conf.fillna(0).astype(int)
    angle_conf_filled = angle_conf.fillna(0).astype(int)

    side_total = side_conf_filled.to_numpy().sum()
    side_diag = side_conf_filled.values.diagonal().sum()
    angle_total = angle_conf_filled.to_numpy().sum()
    angle_diag = angle_conf_filled.values.diagonal().sum()

    print(f"Side-type mismatches: {side_total - side_diag} / {side_total}")
    print(f"Angle-type mismatches: {angle_total - angle_diag} / {angle_total}")


if __name__ == "__main__":
    main()
