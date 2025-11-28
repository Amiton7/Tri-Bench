from pathlib import Path
import json
import os
import re

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

VLM_NAMES = [
    "gemini_2.5_pro",
    "gemini_2.5_flash",
    "openai_gpt_5",
    "qwen_2.5_32b",
]


def extract_json_from_text(txt: str):
    if not isinstance(txt, str):
        return None
    txt = txt.strip()
    if not txt:
        return None

    if txt.startswith("```"):
        parts = txt.split("```")
        inner = "```".join(parts[1:-1]) if len(parts) >= 3 else parts[-1]
        txt = inner.strip()
        if txt.lower().startswith("json"):
            txt = txt[4:].strip()

    start = txt.find("{")
    end = txt.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    candidate = txt[start : end + 1]

    try:
        return json.loads(candidate)
    except Exception:
        try:
            s2 = re.sub(r",\s*}", "}", candidate)
            s2 = re.sub(r",\s*\]", "]", s2)
            return json.loads(s2)
        except Exception:
            return None


def main():
    root = Path(".")
    data_dir = root / "data"
    in_path = data_dir / "tri_bench_vlm_raw_responses.csv"

    if not in_path.exists():
        raise FileNotFoundError(f"Missing raw VLM file: {in_path}")

    df = pd.read_csv(in_path)
    print("raw shape:", df.shape)
    print("raw columns:", list(df.columns))

    if "img_original" in df.columns:
        img_vals = df["img_original"].astype(str)
    elif "image_path" in df.columns:
        img_vals = df["image_path"].astype(str)
    elif "image" in df.columns:
        img_vals = df["image"].astype(str)
    else:
        raise ValueError(
            "Could not find image column. Expected one of: img_original, image_path, image"
        )

    vlm_cols = {}
    for name in VLM_NAMES:
        col = f"{name}_response"
        if col in df.columns:
            vlm_cols[name] = col

    if not vlm_cols:
        raise ValueError("No <model>_response columns found in raw CSV.")

    print("using VLM columns:")
    for name, col in vlm_cols.items():
        print(f"  {name:15s} <- {col}")

    records = []
    for idx, row in df.iterrows():
        rec = {"img_original": img_vals.iloc[idx]}
        for model, col in vlm_cols.items():
            raw_text = row.get(col, "")
            j = extract_json_from_text(raw_text)
            for key in Q_KEYS:
                out_col = f"{key}_{model}"
                if isinstance(j, dict) and key in j:
                    rec[out_col] = j[key]
                else:
                    rec[out_col] = rec.get(out_col, np.nan)
        records.append(rec)

    out_df = pd.DataFrame.from_records(records)
    out_path = data_dir / "tri_bench_vlm_predictions.csv"
    out_df.to_csv(out_path, index=False)
    print("wrote:", out_path)
    print(out_df.head())


if __name__ == "__main__":
    main()