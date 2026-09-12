# Frontend — Virtual Try-On for Clothes

A **single-file** website (`index.html`): no build tools, no frameworks, no dependencies required for normal use. Just open it in a browser.

## Files

- `index.html` — everything (HTML + CSS + JavaScript). All styles are **inline/embedded**, and the garment thumbnails are **inline SVG** — so the file works even inside a restricted preview and needs no CDNs for the base UI.

---

## The one thing you have to configure

At the **top of the `<script>`** in `index.html`:

```js
const BACKEND_URL  = "https://YOUR-BACKEND-HERE.gradio.live"; // <-- your backend URL (NO trailing slash)
const BACKEND_TYPE = "gradio";   // "gradio"  if you run backend/app.py (Colab/Kaggle/HF Space)
                                 // "fastapi" if you run backend/rest_api.py (local test)
const DEMO_MODE    = false;      // set true to get a local (fake) result when the backend is unreachable
const GEN_API_NAME = "tryon";    // the Gradio api_name exposed by backend/app.py
```

- **`BACKEND_URL`**: paste the URL that your backend prints.
  - Gradio on Colab/Kaggle: `https://xxxx.gradio.live`
  - Hugging Face Space: `https://huggingface.co/spaces/<user>/<name>`
  - Local FastAPI test: `http://localhost:8000`

- **`BACKEND_TYPE`** must match what you started:
  - `gradio`  → for `backend/app.py` (Colab/Kaggle/HF Space). Uses the official `@gradio/client`.
  - `fastapi` → for `backend/rest_api.py`. Uses a plain `POST /tryon`.

---

## How the frontend talks to the backend

1. **Person photo**: read from the upload/camera into a data URL, stored in `personDataURL`.
2. **Garment**: the user clicks a tile; the SVG is rasterized to a small PNG (so it can be sent as an image).
3. On **"Try this on"**:
   - `fastapi` path → `POST {BACKEND_URL}/tryon` with `multipart/form-data` (`person_image`, `garment_image`) and receives the result PNG.
   - `gradio` path → uses the official JS client: `app.predict("/tryon", [handle_file(person), handle_file(garment)])`. The client handles Gradio's internal queue/streaming for you.
4. The returned image is shown alongside the original with a **draggable before/after slider**, plus a **Download** button.

If the backend can't be reached and `DEMO_MODE` is `true`, the page draws a **local placeholder** result so you can still explore the UI.

---

## Serving the site locally

Opening `index.html` by double-clicking works for both the camera and uploads. If you prefer a tiny local server (sometimes better for camera permissions + fetch):

```bash
cd frontend
python3 -m http.server 8080
# open http://localhost:8080
```

---

## Deploying (free)

Since it's one static file, any static host works:

- **GitHub Pages** — push this folder to a repo, then enable Pages.
- **Netlify** — drag-and-drop the folder at https://app.netlify.com/drop.
- **Vercel** — import the repo (static preset, output directory `frontend/`).

> ⚠️ The on-page preview in some tools blocks internet access, so the *base UI* renders (because it's self-contained) but calling a real backend will fail there. Open the file in a normal browser (or a deployed URL) for the full experience.
