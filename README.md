# Tri-Bench

This repository contains the data and code for **Tri-Bench: Stress-Testing VLM Reliability on Spatial Reasoning under Camera Tilt and Object Interference** (AAAI 2026 TrustAgent workshop).

**Paper (arXiv for now):** https://arxiv.org/pdf/2512.08860

---

## Overview

Tri-Bench is a diverse benchmark for evaluating 3D spatial reasoning of Vision-Language Models (VLMs). It consists of **400 real photos** of triangles drawn inside a taped square border.  
Each image has:

- 3D / physical triangle measurements,
- 2D pixel geometry from the image plane, and
- answers to six relative-geometry questions (side type, angle type, side length ratios, interior angle differences).

The goal is to see how well VLMs handle **relative spatial reasoning** when we change:

- the **camera pose** (planar vs. tilted), and  
- the **scene context** (10 different everyday objects placed inside the square).

---

## Layout

- `images/`
  - `triangles_original/` – 400 original photos  
  - `triangles_marked/` – same photos with A (red), B (yellow), C (blue) marked  
  - `tri_bench_vlm_table_3_plot.jpg` – accuracy plot used in the paper  
  - `ten_objects_planar.jpg`, `ten_objects_tilted.jpg` – overview shots of the 10 objects

- `data/`
  - CSVs with:
    - 3D ground-truth triangle measurements and labels  
    - 2D pixel coordinates and derived lengths/angles  
    - occlusion annotations  
    - VLM raw JSON responses and parsed predictions  
    - the numbers behind Tables 1–3 and Figure 2 in the paper  

- `code/`
  - scripts to:
    - compute 2D geometry from pixel coordinates  
    - call the four VLMs on all 400 images with a fixed prompt  
    - parse JSON outputs into flat CSVs  
    - reproduce the main tables/plot  
    - run a small homography example and basic triangle-shape stats  

- `prompts/`
  - the exact prompt used for all VLM runs

---

## Using the benchmark

- If you just want the dataset: use `images/` + the CSVs in `data/`. You might be interested in `tri_bench_triangles_3d.csv`, `tri_bench_pixel_geometry_2d.csv`, or `tri_bench_occlusion_annotations.csv`.  
- If you want to reproduce the results: please see the scripts in `code/` (they’re commented with expected paths and usage).

---

## License

All code, data, and images in this repository are released under **CC BY 4.0**.  
See `LICENSE` for details.
