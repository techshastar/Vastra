---
title: Clothed Try-On CatVTON
emoji: 👕
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: cc-by-nc-sa-4.0
---

# 👕 Clothed Try-On — REAL AI backend (CatVTON mask-free)

This Space runs the **real CatVTON virtual try-on model** on Hugging Face's **free GPU**.
Send it a person photo + a garment photo → it returns a photorealistic image of the
person wearing the garment. This is the backend our website calls.

> ⚠️ **If ZeroGPU shows a 🔒 lock icon** (Hugging Face now asks for PRO $9/month
> to *create* your own ZeroGPU Space): **do NOT pay.** Use the 100%-free
> Kaggle/Colab route instead — run `backend/CatVTON_Kaggle_Colab.ipynb`
> (free GPU, no card needed) and use its `https://....gradio.live` URL as your
> backend. The files in this folder stay useful if you ever get PRO access,
> a paid GPU, or your own graphics card.

> **Upload guide (do this once, ~15 minutes, all free):**
>
> 1. **Create a free account** at https://huggingface.co/join (no credit card) and verify your email.
> 2. Click your profile picture (top-right) → **New Space**.
> 3. Fill in: Owner = you, Space name = e.g. `clothed-tryon-catvton`,
>    License = `cc-by-nc-sa-4.0`, SDK = **Gradio**. Click **Create Space**.
> 4. **Pick the free GPU:** go to the Space's **Settings** tab → **Hardware**
>    → choose **ZeroGPU (free)** → Save. (Without this the Space runs on CPU
>    and the real AI will be far too slow — this step matters!)
> 5. Go to the **Files** tab → **Add file → Upload files** → upload `app.py`
>    and `requirements.txt` from this project's `backend/hf_space_catvton/` folder.
>    Then edit `README.md` on the Space and paste this file's contents.
> 6. Wait for the status badge to go **Building → Running** (5–15 min first time).
> 7. **Test it here on this page:** upload a person photo (full/upper body, plain
>    background) + a garment photo → **Submit**. The FIRST run downloads ~4 GB of
>    model weights — if it times out, wait a minute and press Submit again
>    (downloads resume automatically). Later runs take ~30–60 s each.
> 8. **Copy your Space URL** (e.g. `https://YOURNAME-clothed-tryon-catvton.hf.space`)
>    → paste it into the website's `BACKEND_URL` setting (see the main project README).
>
> **Troubleshooting:** build errors → check the **Logs** tab and send them to your
> developer; `quota exceeded` → free ZeroGPU has monthly limits, wait or use the
> Colab backup; blocked/blurry result → change the **seed** and use a better photo.

**How it works:** `app.py` fetches the official CatVTON code, loads the
`zhengchong/CatVTON-MaskFree` weights, and exposes one API endpoint (`/tryon`).
Mask-free = no body-tracking models needed, just 2 photos in, 1 photo out.
Non-commercial use only (CC BY-NC-SA 4.0).
