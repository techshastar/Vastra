"""
rest_api.py
===========

OPTIONAL alternative backend that exposes a real REST endpoint:

    POST /tryon
        multipart/form-data  with fields:
            person_image : the person photo
            garment_image : the garment image
        -> returns the try-on image (image/png)

Use this if you prefer a plain REST API over Gradio (for example if the
frontend "BACKEND_TYPE" is set to "fastapi"). It reuses the exact same
`run_tryon` function, so the actual image generation is identical.

To run it:
    uvicorn rest_api:app --host 0.0.0.0 --port 8000

Then expose it to the internet (see README) and point the frontend at it.
"""

import io

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from PIL import Image

from inference import run_tryon

app = FastAPI(title="Virtual Try-On API")

# Allow the frontend (GitHub Pages / Netlify / Vercel) to call us from any origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # fine for a demo; lock down before production
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "Virtual Try-On API is running. POST images to /tryon"}


@app.post("/tryon")
async def tryon(
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
):
    """Receive two images, run the model, and return the try-on image."""

    # 1) Read the uploaded files into memory.
    person_bytes = await person_image.read()
    garment_bytes = await garment_image.read()

    # 2) Convert bytes -> PIL images.
    person = Image.open(io.BytesIO(person_bytes))
    garment = Image.open(io.BytesIO(garment_bytes))

    # 3) Run generation (real model, or demo fallback).
    result = run_tryon(person, garment)

    # 4) Convert the result back to bytes.
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    buf.seek(0)

    # 5) Return it directly as an image.
    return Response(content=buf.getvalue(), media_type="image/png")
