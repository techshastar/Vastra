# Virtual Try-On: Deep Research Report
**Project:** Clothed Try-On (boys & girls clothing try-on web app)
**Date:** September 2026 | **Goal:** decide the best architecture for the *best free-tier version* of this app.

---

## 0. TL;DR — the single most important finding

There are **TWO completely different technologies** called "virtual try-on", and Snapchat/Lenskart use the **opposite one** from what our project needs:

| | **A. AI-Photo Try-On** (what WE should build) | **B. Live AR Try-On** (what Snapchat/Lenskart do) |
|---|---|---|
| How it works | Upload photo → AI **generates a brand-new photo** of you wearing the garment (takes 10–60 sec) | Live camera → **3D garment model** draped on your body in real time |
| Garment input | **Any flat photo** (catalog pic, screenshot, thrift-store photo) | Must be a **3D model built by a 3D artist** (hours of work per item) |
| Realism | Photorealistic, correct folds/shadows | Game-like; great for rigid items (glasses, shoes), weak for soft clothing from photos |
| Who uses it | **Google, Walmart/Zeekit, Doppl**, Kolors, FASHN, all open-source models | **Snapchat, Lenskart, Amazon shoes**, Banuba, Perfect Corp |
| Free/open-source? | ✅ Yes — many top models are free | ❌ No — needs 3D assets + paid SDKs ($49–$500+/month) |

**Bottom line:** Snapchat-style live clothing try-on is *impossible* with flat garment photos — every garment needs a hand-built 3D model. Lenskart works because glasses are rigid 3D objects on a face mesh. For "upload any garment photo" (Myntra/Meesho-style catalog images), **AI-photo try-on is the only viable path** — and it's exactly what Google and Walmart use. We're on the right track; we just need the real engine instead of the demo overlay.

---

## 1. Open-Source AI Models (GitHub + papers + benchmarks)

Curated paper/model list: [minar09/awesome-virtual-try-on](https://github.com/minar09/awesome-virtual-try-on) — the best single index of the field.

### 1.1 Head-to-head comparison (2026 state of the field)

| Model | Org / Year | Params | Practical VRAM | Speed* | VITON-HD FID↓ (paired) | License | GitHub |
|---|---|---|---|---|---|---|---|
| **CatVTON** (mask-free + inpaint) | Zheng-Chong / ICLR 2025 | 899M (only 49.6M trainable!) | ~8–12 GB | ~11 s (A100) | 5.43–5.89 | CC BY-NC-SA (maintainers open to commercial talks) | [Zheng-Chong/CatVTON](https://github.com/Zheng-Chong/CatVTON) |
| **IDM-VTON** | Yisol / 2024 | 6.2B | 16–24 GB | ~17 s (A100) | 5.76 | CC BY-NC-SA 4.0 | [yisol/IDM-VTON](https://github.com/yisol/IDM-VTON) |
| **Leffa** | Meta / Dec 2024 | 1.8B | ~8–12 GB | fast | **4.54 (best)** | check repo (open-source) | [franciszzj/Leffa](https://github.com/franciszzj/Leffa) |
| **OOTDiffusion** | 2024 | 2.2B | ~10–16 GB | ~46 s (L40S) | 9.30 | CC BY-NC-SA | [levihsu/OOTDiffusion](https://github.com/levihsu/OOTDiffusion) |
| **StableVITON** | 2023 | 1.6B | ~8–16 GB | medium | 6.44 | CC BY-NC-SA | [rlawjdghek/StableVITON](https://github.com/rlawjdghek/StableVITON) |
| **Kolors VTON** | Kuaishou / 2024 | large | high | medium | not benched here | **Apache-2.0** ✅ (commercial OK w/ registration) | via [Kwai-Kolors](https://huggingface.co/spaces/Kwai-Kolors/Kolors-Virtual-Try-On) |
| **FASHN v1.5** | fashn-ai / 2025 | MMDiT | high | fast (~7 s) | — | open weights + paid API w/ free tier | [fashn-ai/fashn-vton-1.5](https://huggingface.co/fashn-ai/fashn-vton-1.5) |
| **Re-CatVTON** | paper Nov 2025 | 860M | ~2–4 GB (bench) | 1.3 s (bench) | **4.44 (best)** | paper-stage, code TBD | [arXiv:2511.18775](https://arxiv.org/html/2511.18775) |
| VITON-HD / HR-VTON / GP-VTON | GAN era (2021–23) | small | ~8 GB | fastest | 8.7–10.9 | research-only | baselines only — dated |
| **CatV2TON** | 2025 | DiT | high | slow | SOTA + **VIDEO** | open | [Zheng-Chong/CatV2TON](https://github.com/Zheng-Chong/CatV2TON) |

*Speed depends hugely on GPU; figures are A100-class unless noted. Benchmark numbers from the CatVTON paper and the Re-CatVTON efficiency study (Nov 2025). Lower FID = more realistic.*

Key reads:
- Quality shootout with example images: [FASHN blog — Top 4 open-source VTON models](https://fashn.ai/blog/comparing-the-top-4-open-source-virtual-try-on-viton-models) — verdict: *"CatVTON stands out as the strongest open-source model… IDM-VTON impressed with fabric textures and color accuracy."*
- Full 2026 comparison incl. VRAM/licenses/build-vs-buy: [fashiolabs.com](https://fashiolabs.com/blog/open-source-virtual-try-on-compared) and [fitroom.app](https://fitroom.app/blog/open-source-vton-models-vs-managed-apis).
- Honest free-tool roundup: [wearview.co — 8 best free AI try-on tools 2026](https://www.wearview.co/blog/best-free-ai-virtual-try-on-tools).

### 1.2 What each model is best at (plain English)

- **CatVTON** — the efficiency king. One small UNet, no pose/text inputs needed, mask-free option (just person photo + garment photo, nothing else). Best quality-per-GPU-rupee. Runs on free Hugging Face ZeroGPU. **Our default engine.**
- **IDM-VTON** — the texture king. Dual-UNet + garment encoder preserves prints/text/logos best. Needs a big GPU (16–24 GB) — painful on free tiers, great as a "high quality mode" later (Colab Pro / own GPU).
- **Leffa (Meta)** — newest quality leader on benchmarks, detail-preserving, open-source with HF demo. **Strong backup/alternative engine** — worth testing against CatVTON during build.
- **OOTDiffusion** — only pick if we need multi-garment outfits (top + bottom in one pass). Weaker single-garment fidelity.
- **Kolors** — the only big one with **Apache-2.0** license (commercial-friendly). Quality excellent, but heavy; free HF demo is hugely popular. Worth knowing for a future commercial version.
- **CatV2TON** — image **+ video** try-on. Too heavy now, but it's our **Phase-3 video feature** (like Google Doppl's animated results).
- **Re-CatVTON** — brand-new paper (Nov 2025) beating everything at CatVTON's tiny size. Watch this space; if code drops, it's an instant upgrade.

### 1.3 Free demos we can test RIGHT NOW (no setup)
- CatVTON: `huggingface.co/spaces/zhengchong/CatVTON`, `…/camenduru/CatVTON` (ZeroGPU), `…/xiaozaa/catvton-flux-try-on` (FLUX quality boost)
- IDM-VTON: `huggingface.co/spaces/yisol/IDM-VTON`
- Leffa: `huggingface.co/spaces/franciszzj/Leffa`
- Kolors: `huggingface.co/spaces/Kwai-Kolors/Kolors-Virtual-Try-On` (tens of thousands of likes)
- Outfit Anyone (Alibaba, research-grade, preset models only)

### 1.4 Full-stack open-source APP projects (not just models)
- [adarshb3/Virtual-Try-On-Application-using-Flask-Twilio-and-Gradio](https://github.com/adarshb3/Virtual-Try-On-Application-using-Flask-Twilio-and-Gradio) (⭐ 347) — Flask + WhatsApp + Gradio/IDM-VTON. **Validates our exact architecture** (thin app calling a Gradio model API).
- [nftblackmagic/catvton-flux](https://github.com/nftblackmagic/catvton-flux) — CatVTON + FLUX inpainting; community quality upgrade path.
- [TemryL/ComfyUI-IDM-VTON](https://github.com/TemryL/ComfyUI-IDM-VTON) — ComfyUI node (advanced users).
- Note: most GitHub VTON repos are *model demos*, not polished consumer apps — a clean, mobile-first, boys/girls categorized app like ours fills a real gap.

---

## 2. Market Players — how the big ones actually do it

### 2.1 Snapchat (Lens Studio + Camera Kit) — live AR, needs 3D garments
- Free **Lens Studio** desktop tool with purpose-built try-on: **3D Body Mesh + Body Tracking + Cloth Simulation + Physics Colliders**. Garments are imported as **3D meshes** that deform to the body; tight parts are "pinned", loose parts simulated ([docs](https://developers.snap.com/lens-studio/features/try-on/clothing-try-on), [cloth sim guide](https://developers.snap.com/lens-studio/features/try-on/cloth-simulation-try-on)).
- A brand must **model every garment in 3D** (see their "create a new garment 3D model" guide) — this is why only big brands do it.
- **Camera Kit SDK** embeds the same Lens into a brand's own iOS/Android/web app ([overview](https://ar.snap.com/blog/ar-virtual-try-on-ecommerce-lens-studio)).
- Business result: AR Lenses drive ~3× purchase-intent lift, 20–30 s playtime ([guide](https://benly.ai/learn/snapchat-ads/snapchat-ar-lens-ads-guide)).
- **Lesson for us:** real-time + fun + shareable, but fundamentally incompatible with "upload any garment photo". Revisit only for accessories (caps, glasses) much later.

### 2.2 Lenskart — face mesh + 3D frames + AI stylist (the Indian benchmark 🇮🇳)
- Started by investing **$1M in Ditto (US 3D-face-model startup, 2017)** for patented 180° 3D try-on ([Entrackr](https://entrackr.com/2017/09/lenskart-invests-rs-6-5-crore-in-ditto/)); later built it **in-house**: AI maps thousands of facial points → 3D avatar → overlay any of 10,000+ frames, 360° rotation, zoom, save, **share** ([case study](https://www.brewmyapp.io/blog/case-study-lenskart), [MarkHub analysis](https://www.markhub24.com/post/lenskart-s-3d-virtual-try-on-technology-as-a-market-making-strategy-in-indian-eyewear-retail)).
- Killer features beyond overlay: **face-shape detection → frame recommendations ("Digital Stylist")**, "Match My Clothes", "Match My Occasion", personality categories; same tech in physical stores (3D Try-On machine).
- Business result: lower returns, higher conversion; face/preference data = competitive moat.
- **Lessons for us:** (1) rigid product + face mesh = why glasses AR works and cloth AR doesn't; (2) **recommendations + share button** matter as much as the try-on itself — we should copy save/share/occasion-styling features; (3) works because frames are a closed catalog of 3D assets.

### 2.3 Google — Doppl app + Shopping try-on (OUR closest reference ⭐)
- **Google Shopping try-on:** tap "try it on" on any apparel listing → upload full-length photo → diffusion AI renders it accounting for folds/stretch. Works on mobile + desktop ([Fortune, Jul 2025](https://fortune.com/2025/07/25/google-virtually-try-on-clothes-ai/)).
- **Doppl (Google Labs app, Jun 2025, US-only, iOS+Android):** upload full-body photo once → try **any outfit photo or screenshot** (thrift store, friend, Instagram) → get image **+ AI-generated video** of the outfit in motion → **save looks, build archive, share** ([TechCrunch](https://techcrunch.com/2025/06/26/google-launches-doppl-a-new-app-that-lets-you-visualize-how-an-outfit-might-look-on-you/), [details](https://www.squaredtech.co/google-doppl-ai-outfit-try-on-app)).
- Google openly warns: *"fit, appearance and clothing details may not always be accurate"* — even Google sets expectations. We should too.
- **Lessons for us:** Doppl's UX is literally our target spec — (1) accept **any garment image/screenshot**, (2) **save + share looks**, (3) **video output** as the wow-factor (our CatV2TON Phase 3).

### 2.4 Walmart / Zeekit — AI-photo try-on at massive scale (architecture validation ⭐)
- Walmart **acquired Zeekit (2021)**; "Be Your Own Model": upload YOUR photo → AI renders 270,000+ catalog items on it. Explicitly **AI-image, NOT live AR** ([Photta breakdown](https://www.photta.app/business/brands/walmart-virtual-try-on), [PYMNTS](https://www.pymnts.com/news/retail/2021/walmart-buys-virtual-try-on-firm-zeekit/)).
- Also offers preset models by height/shape/skin tone + outfit sharing + mix-and-match.
- **Lesson for us:** the world's biggest retailer chose OUR architecture (photo-in → AI → photo-out) for clothing. Huge validation. Copy: preset model option, mix-and-match, share-for-second-opinion.

### 2.5 Amazon — category-specific pragmatism
- Shoes/eyewear: real-time AR (custom CVML foot/face tracking + 3D models). T-shirts: NOT on your body — shown on **size-matched avatars (XS–4XL)**. Killed "Try Before You Buy" (physical) in favor of digital ([guide](https://www.check-my-fit.com/blog/amazon-virtual-try-on-clothes-guide)).
- **Lesson:** even Amazon doesn't do live-AR clothing on your body — they use avatars. Right tool per category.

### 2.6 Myntra (India) — early mover, dated tech
- "Style Studio" (2013): webcam photo + HTML5 overlay of 2,000+ products + Facebook sharing ([Business Standard](https://www.business-standard.com/article/companies/myntra-launches-virtual-trial-room-style-studio-112101800179_1.html)). Basically our old demo-mode overlay — the industry has moved to generative AI since.
- **Lesson:** India market knows this concept; a *real* AI version is a genuine upgrade over what's existed here.

### 2.7 AR SDK vendors (if we ever want live camera features)
| Vendor | Focus | Entry price | Free tier |
|---|---|---|---|
| Banuba | makeup/eyewear/jewelry/hair | $49/mo (1k try-ons) | 1,000 try-ons/mo (makeup+eyewear) |
| Perfect Corp YouCam | beauty | $379/mo (100 SKUs) | 14-day trial |
| ModiFace (L'Oréal) | beauty enterprise | quote-only | none |
| GlamAR | beauty/apparel | $250/mo | trial |
| PictoFiT | **apparel** | ~$500/mo | trial |
| Fittingbox | glasses | $59/mo | 14-day trial |
([source: Banuba comparisons, Jul 2026](https://www.banuba.com/blog/best-virtual-try-on-plugins)) — all beauty/eyewear-first; true apparel AR is the most expensive tier. **Not for us now.**

### 2.8 Managed try-on APIs (plan B if we outgrow free GPUs)
Fashio AI (~$0.94/image, free dev tier), Photta ($49/mo), fashn.ai (free tier, lower res), StyTrix, Pincel (20 free credits). Rule of thumb from every build-vs-buy analysis: **prototype on open-source; buy API only when traffic pays for it.**

---

## 3. Recommendation: the BEST version of OUR project

### 3.1 Architecture (final)
```
Phone/Laptop (single index.html — upload/camera, Boys/Girls tabs,
"upload ANY garment pic", save+share looks, before/after slider)
        │  person.jpg + garment.jpg
        ▼
Hugging Face Space (FREE ZeroGPU) running CatVTON mask-free
        │  result.jpg  (~15–40 s, ~1–2 min cold start)
        ▼
Result + Download + Share (WhatsApp/Instagram — India-first 🇮🇳)
```
- **Engine 1 (default): CatVTON mask-free** — simplest inputs, lowest VRAM, runs on free ZeroGPU, quality ≈ IDM-VTON.
- **Engine 2 (test during build): Leffa** — benchmark leader; if it runs cleanly on ZeroGPU, A/B-test and pick the winner.
- **Engine 3 (documented upgrade): IDM-VTON on Colab/Kaggle** — "high-detail mode" for prints/logos.
- **Phase 3 (future): CatV2TON video** — Doppl-style animated result.

### 3.2 Feature list stolen from the best (all free to implement)
From **Doppl**: upload-any-garment-screenshot, save looks archive, share. From **Lenskart**: styling suggestions, occasion tags, share-for-opinion. From **Walmart**: preset model photos (for shy users), mix-and-match. From **Myntra-2013-done-right**: boys/girls categories, mobile-first.

### 3.3 What we are explicitly NOT building (and why)
- ❌ Snapchat-style **live camera clothing** — needs per-garment 3D models; impossible with flat photos; $500/mo SDKs.
- ❌ Our own model training/fine-tuning — needs 40–80 GB GPUs, paired datasets, days of training.
- ❌ Commercial launch on these weights — CatVTON/IDM/OOTD are **CC BY-NC-SA (non-commercial)**. Fine for learning/demo/portfolio. Commercial path later = Kolors (Apache-2.0) or a paid API.

### 3.4 Execution plan (updated)
1. **Backend Space (CatVTON mask-free)** — `app.py` + `requirements.txt`, named `/tryon` endpoint, auto Demo-fallback without GPU.
2. **Frontend rewire** — point at Space URL, honest status pipeline (warming up → queued → generating), photo-tips coach, save/share looks.
3. **Real garments** — curated Boys/Girls sample photos + upload-your-own (+ URL/screenshot).
4. **Leffa A/B test** — same frontend, second Space; keep the winner as default.
5. **Docs** — beginner README + this research file + Colab backup notebook.
6. **Future** — video try-on (CatV2TON), Hindi/Hinglish UI option.

---
*Sources are linked inline throughout. Benchmark figures: CatVTON paper (arXiv:2407.15886), Re-CatVTON (arXiv:2511.18775), FASHN/FashioLabs/Fitroom comparisons, vendor docs. Market info: TechCrunch, Fortune, PYMNTS, Entrackr, Snap & Lenskart official docs. All accessed Sep 2026; free-tier details change fast — re-verify at build time.*
