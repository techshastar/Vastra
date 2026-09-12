"""
app.py
======

The MAIN backend. This is the friendly, GPU-friendly version that is used by
BOTH the Colab/Kaggle notebook AND the Hugging Face Space.

It is a **Gradio** app with one function called `tryon`. Gradio automatically:

  * gives you a nice web UI at a public URL (so you can test it in your
    browser), and
  * exposes a real REST API at  `/gradio_api/call/tryon`  that the frontend
    drives. No extra code needed.

Why Gradio instead of raw FastAPI for the main app?
  * `demo.launch(share=True)` gives a FREE public internet URL from Colab /
    Kaggle with one line and no account or tunnel token.
  * Hugging Face Spaces run Gradio apps natively (sdk: gradio) - no Docker,
    no server config for a beginner.

(If you specifically want a raw `POST /tryon` FastAPI endpoint instead, see
`rest_api.py` which wraps the SAME `run_tryon` function.)
"""

import os

import gradio as gr

from inference import run_tryon


def tryon_function(person_image, garment_image):
    """
    The function Gradio calls.
    Gradio automatically hands us the images as PIL.Image objects and takes
    the returned PIL.Image and shows it in the UI / returns it via the API.
    """
    result = run_tryon(person_image, garment_image)
    return result


def build_demo():
    with gr.Blocks(title="Virtual Try-On for Clothes") as demo:
        gr.Markdown(
            "# 👕 Virtual Try-On for Clothes\n"
            "Upload a clear, front-facing, full-body photo, pick a garment, "
            "and see yourself wearing it."
        )
        with gr.Row():
            with gr.Column():
                person = gr.Image(
                    type="pil", label="Your photo (person)", sources=["upload", "webcam"]
                )
                garment = gr.Image(
                    type="pil", label="Garment image", sources=["upload"]
                )
                btn = gr.Button("Try this on", variant="primary")
            with gr.Column():
                out = gr.Image(type="pil", label="Try-on result")
        # api_name="tryon" exposes this at POST /gradio_api/call/tryon
        # so the frontend can call it. queue=True enables the submit/poll flow.
        btn.click(
            tryon_function,
            inputs=[person, garment],
            outputs=out,
            api_name="tryon",
            queue=True,
        )
    return demo


demo = build_demo()

# This block only runs when the file is executed directly (e.g. on a laptop,
# Colab, or Space). `share=True` creates a temporary public URL for free.
# Set ENV var "TRYON_SHARE=0" to disable sharing (for HF Spaces, which
# already provide a public URL, or for local-only use).
if __name__ == "__main__":
    share = os.environ.get("TRYON_SHARE", "1") != "0"
    # The endpoint /gradio_api/call/tryon is exposed automatically because the
    # button's click() above sets api_name="tryon".
    demo.launch(share=share, server_name="0.0.0.0", server_port=7860)
