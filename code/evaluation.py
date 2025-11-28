"""
Tri-Bench: VLM accuracy computation, finding patterns in VLM spatial reasoning capacities.

Inputs (from data/):
- tri_bench_triangles_3d.csv
- tri_bench_pixel_geometry_2d.csv
- tri_bench_vlm_predictions.csv
- tri_bench_occlusion_annotations.csv   (optional, for occlusion related analysis)

Outputs (to data/):
- tri_bench_vlm_accuracy_by_image.csv
- tri_bench_vlm_table_1_3D_vs_2D.csv (optional)
- tri_bench_vlm_table_2_shape_bias.csv (optional)
- tri_bench_vlm_table_3_question_wise.csv (optional)
- tri_bench_vlm_table_3_plot.jpg (optional)
"""

from pathlib import Path
import numpy as np
import pandas as pd

Q_KEYS = [
    "side_type", 
    "angle_type",   
    "ab_over_ac",
    "abs_b_minus_c_deg", 
    "max_over_min_side",  
    "angle_range_deg",    
]

Q_LABELS = {
    "side_type": "Q1",
    "angle_type": "Q2",
    "ab_over_ac": "Q3",
    "abs_b_minus_c_deg": "Q4",
    "max_over_min_side": "Q5",
    "angle_range_deg": "Q6",
}

VLM_ORDER = [
    "gemini_2.5_pro",
    "gemini_2.5_flash",
    "openai_gpt_5",
    "qwen_2.5_32b",
]

SIDE_CLASSES = ["scalene", "isosceles", "equilateral"]
ANGLE_CLASSES = ["acute", "obtuse", "right"]


def load_and_merge(root: Path) -> pd.DataFrame:
    data_dir = root / "data"

    tri_3d = pd.read_csv(data_dir / "tri_bench_triangles_3d.csv")
    px_2d = pd.read_csv(data_dir / "tri_bench_pixel_geometry_2d.csv")
    vlm = pd.read_csv(data_dir / "tri_bench_vlm_predictions.csv")

    tri_3d_gt = tri_3d[["img_original"] + Q_KEYS].copy()
    tri_3d_gt = tri_3d_gt.rename(columns={k: f"{k}_3d" for k in Q_KEYS})

    px_2d_gt = px_2d[["img_original"] + Q_KEYS].copy()
    px_2d_gt = px_2d_gt.rename(columns={k: f"{k}_2d" for k in Q_KEYS})

    df = vlm.merge(tri_3d_gt, on="img_original", how="left")
    df = df.merge(px_2d_gt, on="img_original", how="left")

    print("merged df shape:", df.shape)
    return df


def discover_prediction_columns(df: pd.DataFrame):
    """
    Finding VLM prediction columns of the form:
      side_type_gemini_2.5_pro  or  gemini_2.5_pro_side_type
    """
    pred_cols = {q: {} for q in Q_KEYS}

    ignore_suffixes = ("_3d", "_2d")
    meta_cols = {"img_original"}

    for col in df.columns:
        if col in meta_cols:
            continue
        if col.endswith(ignore_suffixes):
            continue

        for q in Q_KEYS:
            prefix = q + "_"
            suffix = "_" + q
            if col.startswith(prefix):
                model = col[len(prefix):]
                pred_cols[q][model] = col
                break
            elif col.endswith(suffix):
                model = col[: -len(suffix)]
                pred_cols[q][model] = col
                break

    print("\nprediction columns:")
    for q, m in pred_cols.items():
        print(" ", q, "→", sorted(m.keys()))
    return pred_cols


def categorical_kappa(y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    yt = y_true.astype(str).str.strip().str.lower()
    yp = y_pred.astype(str).str.strip().str.lower()
    return (yt == yp).astype(float)


def ratio_kappa(y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    t = pd.to_numeric(y_true, errors="coerce")
    p = pd.to_numeric(y_pred, errors="coerce")
    k = pd.Series(np.nan, index=t.index, dtype=float)
    mask = t.abs() > 1e-8
    err = (p[mask] - t[mask]).abs() / t[mask].abs()
    k[mask] = 1.0 - err
    return k


def angle_kappa(y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    t = pd.to_numeric(y_true, errors="coerce")
    p = pd.to_numeric(y_pred, errors="coerce")
    err = (p - t).abs() / 180.0
    return 1.0 - err


def compute_per_image_accuracies(df: pd.DataFrame) -> pd.DataFrame:
    preds = discover_prediction_columns(df)

    acc_df = pd.DataFrame({"img_original": df["img_original"]})

    # keep 3D/2D triangle classes for Table 2
    for col in ["side_type_3d", "angle_type_3d", "side_type_2d", "angle_type_2d"]:
        if col in df.columns:
            acc_df[col] = df[col]

    for q in Q_KEYS:
        q_label = Q_LABELS[q]
        for model, pred_col in preds[q].items():
            gt3 = df[f"{q}_3d"]
            gt2 = df[f"{q}_2d"]

            if q in ("side_type", "angle_type"):
                k3 = categorical_kappa(gt3, df[pred_col])
                k2 = categorical_kappa(gt2, df[pred_col])
            elif q in ("ab_over_ac", "max_over_min_side"):
                k3 = ratio_kappa(gt3, df[pred_col])
                k2 = ratio_kappa(gt2, df[pred_col])
            else:
                k3 = angle_kappa(gt3, df[pred_col])
                k2 = angle_kappa(gt2, df[pred_col])

            acc_df[f"{model}_{q_label}_acc_3d"] = k3
            acc_df[f"{model}_{q_label}_acc_2d"] = k2

    return acc_df


def overall_accuracy_table(acc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Table 1: overall 3D / 2D accuracy per VLM (percent).
    """
    rows = []
    for model in VLM_ORDER:
        cols3 = [c for c in acc_df.columns if c.startswith(model + "_") and c.endswith("_acc_3d")]
        cols2 = [c for c in acc_df.columns if c.startswith(model + "_") and c.endswith("_acc_2d")]
        if not cols3:
            continue
        v3 = acc_df[cols3].to_numpy().ravel()
        v2 = acc_df[cols2].to_numpy().ravel()
        rows.append(
            {
                "vlm": model,
                "acc_3d_percent": 100.0 * float(np.nanmean(v3)),
                "acc_2d_percent": 100.0 * float(np.nanmean(v2)),
            }
        )

    table = pd.DataFrame(rows, columns=["vlm", "acc_3d_percent", "acc_2d_percent"])
    if not table.empty:
        avg = {
            "vlm": "AVERAGE",
            "acc_3d_percent": table["acc_3d_percent"].mean(),
            "acc_2d_percent": table["acc_2d_percent"].mean(),
        }
        table = pd.concat([table, pd.DataFrame([avg])], ignore_index=True)
    return table


def class_accuracy_table(acc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Table 2: accuracy vs. 3D triangle class (Q1/Q2), per VLM (percent).
    """
    side_true = acc_df["side_type_3d"].astype(str).str.strip().str.lower()
    angle_true = acc_df["angle_type_3d"].astype(str).str.strip().str.lower()

    rows = []
    for model in VLM_ORDER:
        c_q1 = f"{model}_Q1_acc_3d"
        c_q2 = f"{model}_Q2_acc_3d"
        if c_q1 not in acc_df.columns or c_q2 not in acc_df.columns:
            continue

        row = {"vlm": model}

        for cls in SIDE_CLASSES:
            m = side_true == cls
            row[f"Q1_{cls}_percent"] = 100.0 * acc_df.loc[m, c_q1].mean()

        for cls in ANGLE_CLASSES:
            m = angle_true == cls
            row[f"Q2_{cls}_percent"] = 100.0 * acc_df.loc[m, c_q2].mean()

        rows.append(row)

    cols = (
        ["vlm"]
        + [f"Q1_{c}_percent" for c in SIDE_CLASSES]
        + [f"Q2_{c}_percent" for c in ANGLE_CLASSES]
    )
    table = pd.DataFrame(rows)[cols]

    if not table.empty:
        avg = {"vlm": "AVERAGE"}
        for c in cols[1:]:
            avg[c] = table[c].mean()
        table = pd.concat([table, pd.DataFrame([avg])], ignore_index=True)
    return table


def image_type_accuracy_table(root: Path, acc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Table 3: for each question and image type (P0, T0, P1, T1),
    average 3D accuracy across all VLMs (percent).
    """
    data_dir = root / "data"
    occ_path = data_dir / "tri_bench_occlusion_annotations.csv"
    if not occ_path.exists():
        raise FileNotFoundError(str(occ_path))

    occ = pd.read_csv(occ_path)[["img_original", "camera_view", "object_in_square"]]
    df = acc_df.merge(occ, on="img_original", how="left")

    def label_image_type(row):
        view = str(row["camera_view"]).strip().lower()
        obj = str(row["object_in_square"]).strip().lower()
        has_obj = obj not in ("", "none", "no object", "no_object")
        if view == "planar":
            return "P1" if has_obj else "P0"
        if view == "tilted":
            return "T1" if has_obj else "T0"
        return "unknown"

    df["image_type"] = df.apply(label_image_type, axis=1)

    qs = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]
    suffix = "_acc_3d"

    for q in qs:
        cols = [c for c in df.columns if c.endswith(f"_{q}{suffix}")]
        if cols:
            df[q] = df[cols].mean(axis=1)

    summary = df.groupby("image_type")[qs].mean().reset_index()

    rows = []
    for _, row in summary.iterrows():
        img_type = row["image_type"]
        if img_type == "unknown":
            continue
        for q in qs:
            rows.append(
                {
                    "image_type": img_type,
                    "question": q,
                    "accuracy_percent": 100.0 * row[q],
                }
            )

    table = pd.DataFrame(rows)
    order = ["P0", "T0", "P1", "T1"]
    if not table.empty:
        table["image_type"] = pd.Categorical(table["image_type"], order)
        table = table.sort_values(["image_type", "question"]).reset_index(drop=True)
    return table


def plot_image_type_accuracy(root: Path, cond_df: pd.DataFrame) -> None:
    """
    Grouped bar plot for Table-3 results.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("could not import matplotlib, skipping plot:", e)
        return

    qs = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]
    types = ["P0", "T0", "P1", "T1"]

    pivot = cond_df.pivot(index="question", columns="image_type", values="accuracy_percent")
    pivot = pivot.reindex(index=qs, columns=types)

    x = np.arange(len(qs))
    width = 0.2

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, t in enumerate(types):
        if t not in pivot.columns:
            continue
        y = pivot[t].values
        ax.bar(x + (i - 1.5) * width, y, width=width, label=t)

    ax.set_xticks(x)
    ax.set_xticklabels(qs)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend(title="image type")
    ax.set_title("Tri-Bench: accuracy by question and image type")

    fig.tight_layout()
    out_path = root / "data" / "tri_bench_vlm_table_3_plot.jpg"
    fig.savefig(out_path, dpi=300)
    print("wrote plot:", out_path)


def main():
    root = Path(".")

    df = load_and_merge(root)
    acc_df = compute_per_image_accuracies(df)

    acc_path = root / "data" / "tri_bench_vlm_accuracy_by_image.csv"
    acc_df.to_csv(acc_path, index=False)
    print("wrote:", acc_path)

    # # Table 1: overall 3D / 2D accuracy per VLM
    # t1 = overall_accuracy_table(acc_df)
    # t1_path = root / "data" / "tri_bench_vlm_table_1_3D_vs_2D.csv"
    # t1.to_csv(t1_path, index=False)
    # print("wrote:", t1_path)

    # # Table 2: class-wise accuracy for Q1/Q2
    # t2 = class_accuracy_table(acc_df)
    # t2_path = root / "data" / "tri_bench_vlm_table_2_shape_bias.csv"
    # t2.to_csv(t2_path, index=False)
    # print("wrote:", t2_path)

    # # Table 3 + plot: accuracy by question × image type (P0,T0,P1,T1)
    # t3 = image_type_accuracy_table(root, acc_df)
    # t3_path = root / "data" / "tri_bench_vlm_table_3_question_wise.csv"
    # t3.to_csv(t3_path, index=False)
    # print("wrote:", t3_path)
    # if not t3.empty:
    #     plot_image_type_accuracy(root, t3)


if __name__ == "__main__":
    main()