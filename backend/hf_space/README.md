# Deploy the backend on Hugging Face Spaces (recommended, free)

Hugging Face Spaces is the easiest way to get an **always-on public URL** for
your try-on model for free. A Gradio Space runs instantly with no Docker and
no tunnel tricks.

## Steps (a beginner can do this)

1. **Create a Space**
   - Go to https://huggingface.co/new-space
   - Give it a name (e.g. `my-clothes-tryon`).
   - **License:** choose `MIT` or `Apache 2.0`.
   - **SDK:** choose **Gradio**.
   - **Hardware:** for a demo, choose the free **CPU Basic**, or better, the
     free **GPU (T4)** if available. IDM-VTON needs a GPU to be usable.

2. **Upload two files** (drag-and-drop into the Space):
   - `app.py`        -> copy from `backend/app.py` in this repo.
   - `inference.py`  -> copy from `backend/inference.py`
   - `requirements.txt` -> copy from `backend/requirements.txt`

3. **Tell the Space where the repo is**
   The notebook clones IDM-VTON to `/content/IDM-VTON`. On a Space you must
   clone it at runtime too. Add these lines to the **top of `app.py`** (or a
   `setup.sh` that runs before the app):

   ```python
   import os, subprocess
   if not os.path.isdir("/content/IDM-VTON"):
       subprocess.run(["git", "clone", "https://github.com/yisol/IDM-VTON", "/content/IDM-VTON"])
       # download weights etc. (see the Colab notebook for the exact commands)
   ```

   > Simpler short-term option: for the *first test*, leave the model off and
   > set `ENGINE = "demo"` in `inference.py`. The Space will then run on free
   > CPU and answer in a second — great for making sure the pipeline works.

4. **Set the port**
   Gradio Spaces serve on port `7860`. `app.py` already uses `server_port=7860`.
   Make sure the last line is `demo.launch(...)`.

5. **Start the Space**
   Click **Restart Space**. The build will install deps and take a few minutes.
   When it is ready you get a URL:
   `https://huggingface.co/spaces/<your-username>/<space-name>`

   Put that URL (adding `https://`) as the **`BACKEND_URL`** in the frontend.

## Important
- Free GPU Spaces **pause** when idle and **throttle** usage. That is normal.
- Free tier often gives **CPU** by default; ask for GPU and note that free GPU
  has a daily limit.
- Because IDM-VTON is licensed **non-commercial**, keep this demo non-profit
  and educational.
