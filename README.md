# Tri-Bench

Official Repo for **Tri-Bench: Stress-testing VLM spatial reasoning under camera tilt and object interference** (AAAI 2026 TrustAgent workshop).

Paper Link: (to be provided)

## Description
The benchmark provides concrete data for stress-testing how capable are Vision Language Models (VLMs) when it comes to raw spatial reasoning skills. The stress-test we used here consists of six relative spatial reasoning Questions. The dataset is open under the mentioned licence, to be used for various different analyses in the future.

## Layout

- `images/`: 400 original + 400 marked triangle images
- `data/`: 3D ground truth and 2D camera projection measurements, VLM raw responses, parsed csv, and evaluation tables
- `code/`: scripts for:
  - `projection_2D_measurements.py`: manual pixel marking, and measuring geometric entities
  - `run_vlm_inference.py`: run VLMs for the 400 images, with a fixed prompt and JSON format
  - `parse_json.py`: parse raw JSON responses into flat CSV
  - `evaluation.py`: generate analysis, i.e. tables/plots
 
## License

All code, data, and images in this repo are released under CC BY 4.0 license. Please see the `LICENSE`.

