"""
Tri-Bench: VLM inference script.

Runs the fixed JSON-style prompt on all triangle images using:
- Gemini 2.5 Pro
- Gemini 2.5 Flash
- GPT-5
- Qwen2.5-VL-32B (via Fireworks API)

Writes one CSV with one row per image and one text column per model.
"""

import os
import glob
import base64
import time
import random
import concurrent.futures

import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

import google.generativeai as genai
import openai
import requests
from pathlib import Path

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

if OPENAI_API_KEY:
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
else:
    openai_client = None

def load_prompt() -> str:
    prompt_path = Path(__file__).resolve().parent.parent / "prompts" / "tri_bench_prompt.txt"
    return prompt_path.read_text(encoding="utf-8")

PROMPT_TEXT = load_prompt()

MAX_OUTPUT_TOKENS = 512

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_gemini_response(image_path: str, text_prompt: str, model: str) -> str:
    try:
        img = Image.open(image_path)
        gemini_model = genai.GenerativeModel(model)
        generation_config = {
            "max_output_tokens": None,
            "response_mime_type": "application/json",
        }
        response = gemini_model.generate_content(
            contents=[img, text_prompt],
            generation_config=generation_config,
        )
        if response.text:
            return response.text
        return "Error: Empty response from Gemini."
    except Exception as e:
        return f"Error: {e}"


def get_openai_response(image_path: str, text_prompt: str, model: str) -> str:
    if openai_client is None:
        return "Error: OPENAI_API_KEY not configured."

    base64_image = encode_image(image_path)
    max_retries = 5
    base_wait = 10

    for attempt in range(max_retries):
        try:
            resp = openai_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=MAX_OUTPUT_TOKENS,
            )
            return resp.choices[0].message.content
        except openai.RateLimitError:
            wait = base_wait * (2 ** attempt) + random.uniform(0, 2)
            print(f"[openai] rate limit on attempt {attempt + 1}, waiting {wait:.1f}s")
            time.sleep(wait)
        except Exception as e:
            return f"Error: {e}"

    return "Error: Exceeded max retries for OpenAI."


def get_fireworks_response(image_path: str, text_prompt: str, model: str) -> str:
    if FIREWORKS_API_KEY is None:
        return "Error: FIREWORKS_API_KEY not configured."

    try:
        base64_image = encode_image(image_path)
        headers = {
            "Authorization": f"Bearer {FIREWORKS_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 1024,
        }
        resp = requests.post(
            "https://api.fireworks.ai/inference/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {e}"


MODELS_TO_RUN = {
    "gemini_2.5_pro": {
        "api_function": get_gemini_response,
        "model_id": "gemini-2.5-pro-preview-03-25",
        "max_workers": 16,
    },
    "gemini_2.5_flash": {
        "api_function": get_gemini_response,
        "model_id": "gemini-2.5-flash",
        "max_workers": 16,
    },
    "openai_gpt_5": {
        "api_function": get_openai_response,
        "model_id": "gpt-5-chat-latest",
        "max_workers": 4,
    },
    "qwen_2.5_32b": {
        "api_function": get_fireworks_response,
        "model_id": "accounts/fireworks/models/qwen2p5-vl-32b-instruct",
        "max_workers": 8,
    },
}


def run_inference_for_model(model_name: str, config: dict, df: pd.DataFrame) -> pd.DataFrame:
    print(f"\nRunning model: {model_name} ({config['model_id']})")
    column_name = f"{model_name}_response"
    api_fn = config["api_function"]
    model_id = config["model_id"]
    max_workers = config["max_workers"]

    responses = [None] * len(df)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(api_fn, row["image_path"], row["prompt"], model=model_id): idx
            for idx, row in df.iterrows()
        }

        for future in tqdm(
            concurrent.futures.as_completed(future_to_idx),
            total=len(df),
            desc=model_name,
        ):
            idx = future_to_idx[future]
            try:
                responses[idx] = future.result()
            except Exception as e:
                responses[idx] = f"Error: {e}"

    df[column_name] = responses
    print(f"Completed: {model_name}")
    return df


def main():
    image_dir = os.getenv("TRIBENCH_IMAGE_DIR", "triangles_original")
    pattern = os.path.join(image_dir, "*.jpg")
    print(f"Searching for images: {pattern}")
    image_paths = sorted(glob.glob(pattern))

    if not image_paths:
        raise RuntimeError("No images found in image_dir.")

    print(f"Found {len(image_paths)} images.")

    results_df = pd.DataFrame({"image_path": image_paths})
    results_df["prompt"] = PROMPT_TEXT

    for name, conf in MODELS_TO_RUN.items():
        results_df = run_inference_for_model(name, conf, results_df)

    output_path = os.getenv(
        "TRIBENCH_VLM_OUTPUT",
        os.path.join("data", "tri_bench_vlm_raw_responses.csv"),
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved all model responses to: {output_path}")
    print(results_df.head())


if __name__ == "__main__":
    main()
