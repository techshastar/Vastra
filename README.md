# VASTRA — Virtual Trial Room

Free, open-source virtual clothing try-on for boys and girls — Lenskart-style, but for
clothes. Upload a full-body photo, pick an outfit (or your own clothes), choose a fit,
and open-source AI shows you wearing it. No sign-up, no fees.

## How it works (Pro pipeline)

```
Your photo + garment photo
        │
        ▼
1. OpenPose (pose) + SCHP human parsing (onnx, CPU) ──► body map
        │
        ▼
2. Garment-area mask, grown/shrunk by Fit (slim/regular/loose)
        │
        ▼
3. OOTDiffusion-DC repaints ONLY the masked garment area (fp16, free T4 GPU)
        │
        ▼
4. Skin-restore: ORIGINAL face/hair/arms/legs pixels pasted back (bit-exact)
4b. Hem-repair: light leftover strips below the new hem restored to original
        │
        ▼
5. Final composite: outside the garment mask = your original photo, pixel-perfect
```

**Body guarantee:** outside the garment region the output pixels *are* your input photo —
face, body shape, background cannot change even slightly. Inside the region you get a
fresh AI-rendered garment in your chosen fit.

## Quick start (zero setup)

Just double-click `frontend/index.html` — it uses a free shared OOTDiffusion GPU
server, no account, no Kaggle needed. Upload photo → pick outfit → *See it on me*.

## Pro mode (your own server: Fit control + skin-restore + hem-repair)

1. Open `backend/OOTD_Pro_Kaggle_Colab.ipynb` in Kaggle (or Colab): GPU T4 x2, Internet ON.
2. Run all cells, copy the `https://....gradio.live` link.
3. Open `frontend/index.html?backend=PASTE-LINK-HERE`.
4. Now **Fit** (Slim/Regular/Loose) unlocks too.

No backend? `frontend/preview/index.html` runs a fully offline local demo (composite
preview, not AI).

## Project structure

```
clothed-tryon/
├── README.md
├── frontend/
│   ├── index.html          # THE app — single file, works by double-click (+ backend URL)
│   ├── preview/index.html  # offline demo copy (no backend needed)
│   ├── assets/             # 6 built-in garment photos (b1-b3 boys, g1-g3 girls)
│   └── vendor/             # real downloaded libs, embedded into index.html:
│                           # gsap 3.12.5, ScrollTrigger 3.12.5, lenis 1.3.26 (all MIT)
└── backend/
    ├── OOTD_Pro_Kaggle_Colab.ipynb  # ★ Pro backend (OOTDiffusion + fit + skin-restore)
    ├── ootd_mask_fit.py             # mask/fit/skin code (mirror of notebook cell 4, tested)
    ├── CatVTON_Kaggle_Colab.ipynb   # backup backend (CatVTON mask-free, older 4-arg API)
    └── hf_space_catvton/            # alt. route for GPU/PRO owners (Hugging Face Space)
```

Backend APIs: default (no `?backend=`) = official OOTDiffusion Space `/process_dc`
(shared, free, Regular fit). With `?backend=` your Pro server's `/tryon` takes
`(person, garment, category, fit)`. The old CatVTON `/tryon` took
`(person, garment, seed, steps)` — kept only as backup.

## Model & library licences (read before any commercial use)

| Component | Licence | Note |
|---|---|---|
| OOTDiffusion weights/code | CC BY-NC-SA 4.0 | demo/hackathon OK, **no commercial use** |
| CatVTON (backup) | non-commercial research | demo only |
| SCHP parsing / OpenPose weights | research use | via OOTDiffusion checkpoints |
| GSAP, Lenis, SegFormer refs | MIT / Apache-2.0 | free incl. commercial |
| Fraunces + Inter fonts | SIL Open Font License | free, embedded in the HTML |

## Limitations (honest)

- Garment types are upper / lower / dress; very loose robes, sarees, complex layers vary.
- Fit = Slim/Regular/Loose rendering (repaint-area control), not tailor-exact cm sizing —
  true sizing needs body measurements.
- Swapping to a dress renders AI legs (your original legs were covered) — everything else
  stays pixel-exact.
- ~1.5–2.5 min per try-on on a free T4; the Kaggle tab must stay open (URL dies with it).
- Best input: front-facing full-body photo, plain background, JPG/PNG.

## Free-tier budget

30 GPU hrs/week ÷ ~2 min/try-on ≈ **700+ try-ons/week** — plenty for a hackathon.
Demo-day rule: start the session ~20 min early, keep 2–3 pre-generated results + the
offline `preview/index.html` as backup.
