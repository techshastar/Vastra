"""
inference.py
============

One function, two modes:

    run_tryon(person_image, garment_image) -> PIL.Image

MODE A  ---  "real" model (IDM-VTON)
    When you have cloned the IDM-VTON repo and downloaded its weights
    (this is done by the Colab notebook / Hugging Face Space), this file
    loads the model once and runs a real virtual-try-on inference.

MODE B  ---  "demo" mode
    When the real model is NOT available (for example you are running this
    on a normal laptop with no GPU, or you just want to test the UI), we
    fall back to a simple "composite" that pastes the garment shape onto
    the person photo. It is NOT photorealistic, but it lets you test the
    FULL app flow (upload -> choose garment -> generate -> see result)
    with zero setup and zero cost.

There is a single switch at the bottom of the file:  ENGINE = "auto"
You normally never touch it - "auto" picks the real model when it is
installed and the demo otherwise.
"""

import os
import io
from typing import Optional

from PIL import Image, ImageOps, ImageDraw, ImageFilter

# --------------------------------------------------------------------------
# Which engine to use.
#   "auto"   -> use IDM-VTON if the repo is present, otherwise demo mode
#   "idm"    -> force the real model (will error if not installed)
#   "demo"   -> force the demo composite (no GPU / internet needed)
# --------------------------------------------------------------------------
ENGINE = "auto"

# The folder where the IDM-VTON repository is cloned. Update this if you put
# it somewhere else on your Colab / Space.
IDM_VTON_REPO_DIR = os.environ.get("IDM_VTON_REPO_DIR", "/content/IDM-VTON")


# ==========================================================================
#  DEMO MODE  (works everywhere, no model needed)
# ==========================================================================
def _demo_tryon(person: Image.Image, garment: Image.Image) -> Image.Image:
    """
    A very simple fake try-on: paste the garment image onto the person's
    torso area. It is intentionally simple so a beginner can follow it and
    so it runs instantly on CPU.
    """
    # Work on a copy so we never modify the user's upload.
    person = person.convert("RGB")
    garment = garment.convert("RGBA")

    # Fit the person into a sane canvas (max 800px tall) to keep things fast.
    max_h = 800
    if person.height > max_h:
        scale = max_h / person.height
        person = person.resize(
            (int(person.width * scale), int(person.height * scale)),
            Image.LANCZOS,
        )
    W, H = person.size

    # Resize the garment so it is about 45% of the person's width.
    target_w = int(W * 0.45)
    target_h = int(target_w * (garment.height / garment.width))
    garment = garment.resize((target_w, target_h), Image.LANCZOS)

    # Decide where to place it: roughly the middle-top of the torso.
    x = (W - garment.width) // 2
    y = int(H * 0.22)

    # Feather the edges a little so it looks less like a hard sticker.
    mask = garment.split()[3]  # alpha channel
    mask = mask.filter(ImageFilter.GaussianBlur(3))

    # Paste the garment over the person using its alpha as a mask.
    person.paste(garment, (x, y), mask)

    # Attach a small label so it is obvious this is the "demo" output.
    label = Image.new("RGB", (int(W * 0.9), 30), (255, 255, 255))
    label_draw = ImageDraw.Draw(label)
    label_draw.text(
        (8, 6),
        "DEMO MODE (open-source AI model not loaded)",
        fill=(200, 40, 40),
    )
    out = Image.new("RGB", (W, H + label.height), (255, 255, 255))
    out.paste(person, (0, 0))
    out.paste(label, (int(W * 0.05), H))
    return out


# ==========================================================================
#  REAL MODEL  (IDM-VTON) - loaded once, reused many times
# ==========================================================================
_model = None            # keep the loaded model in a global so we only load it once
_model_loaded = False


def _load_idm_model():
    """
    Load the IDM-VTON pipeline. This is the part that needs the repo cloned
    and the weights downloaded (see the Colab notebook). It can take 30+ s
    and a few GB of GPU memory, so we do it ONCE and cache the result.
    """
    global _model, _model_loaded
    if _model_loaded:
        return _model

    # Make sure the repo is actually there.
    if not os.path.isdir(IDM_VTON_REPO_DIR):
        raise FileNotFoundError(
            f"IDM-VTON repo not found at {IDM_VTON_REPO_DIR}. "
            "Run the setup in 'IDM_VTON_colab.ipynb' first."
        )

    # Add the repo to the Python path so we can import its modules.
    import sys
    sys.path.insert(0, IDM_VTON_REPO_DIR)

    # ------------------------------------------------------------------
    # The exact import / load calls depend on the IDM-VTON version.
    # The official repo ships its own example script and a Gradio demo.
    # Below is a representative wiring based on the official repo's
    # configs. If you update IDM-VTON, adjust this block to match the
    # repo's own `test.py` / `app` example.
    # ------------------------------------------------------------------
    from transformers import AutoTokenizer
    from vton.inference import IDMVTONPipeline   # example path -> adjust as needed

    device = _get_device()
    dtype = torch.float16

    base_model = "yisol/IDM-VTON"  # HF weights repo
    pipe = IDMVTONPipeline.from_pretrained(
        base_model, torch_dtype=dtype, variant="fp16"
    )
    pipe.to(device)
    _model = {"pipe": pipe}
    _model_loaded = True
    return _model


def _get_device():
    """Return 'cuda' when a GPU is available, otherwise 'cpu'."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _idm_tryon(person: Image.Image, garment: Image.Image) -> Image.Image:
    """
    Run a real IDM-VTON inference.

    INPUT : two PIL images
    OUTPUT: one PIL image (the person wearing the garment)

    Because the repo's exact API changes between releases, this function
    wraps the call and, crucially, always returns a PIL image so the Gradio
    / FastAPI layer does not have to know about model internals.
    """
    import torch

    model = _load_idm_model()
    pipe = model["pipe"]
    device = _get_device()

    # Prepare inputs.
    # IDM-VTON takes the person image, garment image, and a text prompt.
    prompt = "a photo of a person wearing the garment, high quality"
    out = pipe(
        person=person,
        cloth=garment,
        prompt=prompt,
        num_inference_steps=30,
        guidance_scale=2.5,
    )
    return out["images"][0]


# ==========================================================================
#  PUBLIC ENTRY POINT
# ==========================================================================
def run_tryon(person_image: Image.Image, garment_image: Image.Image) -> Image.Image:
    """
    Generate the try-on image.
    Returns a PIL.Image that the server sends back to the browser.

    This is the ONLY function the server (Gradio / FastAPI) needs to call.
    """
    # Normalize inputs a little so both demo and real models behave.
    person = _normalize(person_image)
    garment = _normalize(garment_image)

    mode = ENGINE
    if mode == "auto":
        # Try the real model; if anything is missing, gracefully fall back.
        try:
            return _idm_tryon(person, garment)
        except Exception as e:  # noqa: BLE001
            print(f"[inference] Real model unavailable, using demo. Reason: {e}")
            return _demo_tryon(person, garment)
    elif mode == "idm":
        return _idm_tryon(person, garment)
    else:
        return _demo_tryon(person, garment)


def _normalize(img: Image.Image) -> Image.Image:
    """Convert any input into a crisp RGB PIL image on a white background."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    return img
