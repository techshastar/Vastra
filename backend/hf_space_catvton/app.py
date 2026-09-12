"""
Clothed Try-On — REAL AI backend (CatVTON mask-free) for Hugging Face Spaces.
============================================================================

What this file is:
  A small Gradio web app that runs the REAL CatVTON virtual try-on model.
  You upload it to a free Hugging Face Space (with free ZeroGPU hardware)
  and it gives you a public URL like:

      https://YOURNAME-clothed-tryon-catvton.hf.space

  Our website (frontend/index.html) then sends "person photo + garment photo"
  to that URL and gets back a REAL AI-generated try-on image.

How it works (3 steps):
  1. On startup it downloads the OFFICIAL CatVTON code (only the small
     `model/` + `utils.py` files, not the big demo images) from the official
     Hugging Face Space repo: https://huggingface.co/spaces/zhengchong/CatVTON
  2. On the first try-on it downloads the CatVTON-MaskFree weights (~4 GB,
     one-time; Hugging Face resumes the download if it gets interrupted).
  3. Every try-on runs the model on the free GPU and returns the result.

Why "mask-free"?
  The classic version needs extra body-tracking models (DensePose/SCHP +
  detectron2) which are painful to install. The mask-free version just takes
  (person photo + garment photo) and figures out the rest itself. Same great
  quality, 10x simpler. This is the exact recipe the official CatVTON Space
  uses for its "Mask-free & Pix2Pix" tab.

If the GPU/weights are unavailable for any reason, this app falls back to a
clearly-labelled DEMO image (side-by-side + "DEMO MODE" banner) instead of
crashing — so you always get *something* back, and you always know which
engine produced it.

Non-commercial use only (CatVTON licence: CC BY-NC-SA 4.0).
"""

import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# STEP 1 — Fetch the official CatVTON source code (a tiny ~3 MB download
# of the official Hugging Face Space repo: model code + utils only).
# ---------------------------------------------------------------------------
SRC_DIR = "catvton_src"          # folder the code will live in
SRC_REPO = "https://huggingface.co/spaces/zhengchong/CatVTON"  # official code
NEEDED_FILES = [
    os.path.join(SRC_DIR, "model", "pipeline.py"),
    os.path.join(SRC_DIR, "utils.py"),
]

if not all(os.path.exists(p) for p in NEEDED_FILES):
    print(">>> Downloading official CatVTON code (~3 MB)...")
    subprocess.run(
        ["git", "clone", "--depth", "1", SRC_REPO, SRC_DIR], check=True
    )
    print(">>> Official CatVTON code ready.")

# Let Python import `model.*` and `utils` from the downloaded folder.
sys.path.insert(0, SRC_DIR)

# ---------------------------------------------------------------------------
# STEP 2 — Normal imports (these come from requirements.txt on the Space).
# ---------------------------------------------------------------------------
import gradio as gr  # noqa: E402  (import AFTER sys.path fix on purpose)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

# Official CatVTON pieces (mask-free needs ONLY these two — no DensePose,
# no human-parsing models, no detectron2 at all).
from utils import init_weight_dtype, resize_and_crop, resize_and_padding  # noqa: E402
from model.pipeline import CatVTONPix2PixPipeline  # noqa: E402

# ---------------------------------------------------------------------------
# STEP 3 — ZeroGPU support. On a Hugging Face Space with ZeroGPU hardware,
# GPU work must happen inside a function decorated with @spaces.GPU.
# When running anywhere else (your laptop, Colab without `spaces`), the
# decorator becomes a harmless no-op so the same file still runs.
# ---------------------------------------------------------------------------
try:
    import spaces  # pre-installed on HF Spaces

    gpu = spaces.GPU(duration=120)  # same 120 s budget the official Space uses
except ImportError:
    def gpu(fn):  # local run: do nothing special
        return fn


# ---------------------------------------------------------------------------
# Settings (same values the official CatVTON Space uses — proven to work).
# ---------------------------------------------------------------------------
WIDTH, HEIGHT = 768, 1024          # model was trained at this size; keep it
MASKFREE_REPO = "zhengchong/CatVTON-MaskFree"   # official weights
BASE_MODEL = "timbrooks/instruct-pix2pix"       # base model for mask-free
ATTN_VERSION = "mix-48k-1024"      # which attention checkpoint to load
GUIDANCE = 2.5                     # official default (higher = more saturated)
ENGINE = "starting..."             # becomes "catvton" or "demo"
PIPE = None                        # the loaded model (None until first use)


def pick_dtype():
    """bf16 on big datacenter GPUs (A100/H100+), fp16 on smaller ones (T4).

    The free T4 has no bf16 hardware support, so bf16 there explodes memory
    (CUDA out of memory). fp16 runs great on T4.
    """
    try:
        major, _ = torch.cuda.get_device_capability()
        return init_weight_dtype("bf16" if major >= 8 else "fp16")
    except Exception:
        return init_weight_dtype("fp16")


def load_pipeline():
    """Download weights (first time only) and build the model. Cached globally."""
    global PIPE, ENGINE
    if PIPE is not None:
        return PIPE
    print(">>> Downloading CatVTON-MaskFree weights (one-time, ~4 GB)...")
    repo_path = snapshot_download(repo_id=MASKFREE_REPO)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    PIPE = CatVTONPix2PixPipeline(
        base_ckpt=BASE_MODEL,
        attn_ckpt=repo_path,
        attn_ckpt_version=ATTN_VERSION,
        weight_dtype=pick_dtype(),
        use_tf32=True,
        device=device,
    )
    ENGINE = "catvton"
    print(">>> REAL CatVTON engine loaded on:", device)
    return PIPE


# Try to load immediately if a GPU is already visible (normal GPU machines).
# On ZeroGPU the GPU appears per-request, so this simply skips and the model
# loads lazily inside the first try-on call instead. Either way is fine.
try:
    if torch.cuda.is_available():
        load_pipeline()
    else:
        print(">>> No GPU at startup — model will load on first request.")
except Exception as e:  # never crash the Space at boot; fallback exists
    print(">>> Startup model load failed, will retry on first request:", e)


def demo_composite(person_img, garment_img):
    """Clearly-labelled fallback used ONLY when the real AI cannot run.

    Returns a side-by-side image with a red DEMO MODE banner, so nobody ever
    mistakes it for a real try-on result.
    """
    global ENGINE
    ENGINE = "demo"
    h = 768
    person = person_img.convert("RGB").copy()
    garment = garment_img.convert("RGB").copy()
    person.thumbnail((512, h))
    garment.thumbnail((512, h))
    canvas = Image.new("RGB", (person.width + garment.width + 30, h + 70),
                       (24, 24, 28))
    # Vertically center each thumbnail in the area below the banner.
    canvas.paste(person, (0, 70 + (h - person.height) // 2))
    canvas.paste(garment, (person.width + 30, 70 + (h - garment.height) // 2))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, canvas.width, 70], fill=(176, 30, 30))
    draw.text((15, 20),
              "DEMO MODE — real AI unavailable (no GPU / weights failed)",
              fill=(255, 255, 255))
    draw.text((15, 45), f"person {person.size}  +  garment {garment.size}",
              fill=(255, 220, 220))
    return canvas


@gpu
def tryon(person_path, garment_path, seed=42, steps=50):
    """Run a REAL try-on. person+garment file paths in, result image out.

    This function is exposed to our website as the API endpoint `/tryon`.
    """
    if not person_path or not garment_path:
        raise gr.Error("Please upload BOTH a person photo and a garment photo.")
    person = Image.open(person_path).convert("RGB")
    garment = Image.open(garment_path).convert("RGB")
    try:
        pipe = load_pipeline()
        # Exact preprocessing the official Space uses:
        person_r = resize_and_crop(person, (WIDTH, HEIGHT))
        garment_r = resize_and_padding(garment, (WIDTH, HEIGHT))
        generator = None
        if int(seed) != -1:
            generator = torch.Generator(device=pipe.device).manual_seed(int(seed))
        result = pipe(
            image=person_r,
            condition_image=garment_r,
            num_inference_steps=int(steps),
            guidance_scale=GUIDANCE,
            generator=generator,
        )[0]
        global ENGINE
        ENGINE = "catvton"
        return result
    except Exception as e:
        print(">>> REAL engine failed, using labelled demo fallback:", repr(e))
        return demo_composite(person, garment)


# ---------------------------------------------------------------------------
# Gradio web UI (what you see when you open the Space in a browser).
# ---------------------------------------------------------------------------
HEADER = """
# 👕 Clothed Try-On — REAL AI backend (CatVTON mask-free)
**Person photo + garment photo in → you wearing it out.** Non-commercial demo.
- First ever run downloads ~4 GB of model weights (one-time). If it times out,
  wait a minute and press **Submit** again — downloads resume automatically.
- Good photos = good results: full/upper body, plain background, fitted clothes.
- If the safety filter blocks a normal photo, just change the **seed**.
"""

with gr.Blocks(title="Clothed Try-On (CatVTON)") as demo:
    gr.Markdown(HEADER)
    with gr.Row():
        person_in = gr.Image(label="1️⃣ Your photo (full/upper body)",
                             type="filepath")
        garment_in = gr.Image(label="2️⃣ Garment photo (flat product pic works)",
                              type="filepath")
    with gr.Row():
        seed_in = gr.Slider(minimum=-1, maximum=10000, step=1, value=42,
                            label="Seed (-1 = random, change if result looks off)")
        steps_in = gr.Slider(minimum=10, maximum=100, step=5, value=50,
                             label="Steps (more = finer detail but slower)")
    btn = gr.Button("✨ Try it on", variant="primary")
    result_out = gr.Image(label="Result (REAL AI — not an overlay!)")
    # api_name="tryon" is what lets our website call this as .../gradio_api/call/tryon
    btn.click(tryon, [person_in, garment_in, seed_in, steps_in],
              result_out, api_name="tryon")

demo.launch()
