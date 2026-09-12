"""One-time upgrade script (Step 1): real garment photos + upload-your-own-outfit.

What it does to index.html (and preview/index.html):
  1. Compresses frontend/assets/*.jpg and EMBEDS them as data-URIs, so the
     single index.html still works standalone (double-click demo on stage!).
  2. Replaces the cartoon-SVG garment catalog with real product photos.
  3. Adds an "upload your own outfit" button + preview + selection logic.
  4. Deletes the now-dead SVG helper functions.
Every replacement asserts exactly-1 match, so it fails loudly, never silently.
Run:  python3 inject_garments.py   (from the frontend/ folder)
"""
import base64
import io
import sys
from PIL import Image

ASSETS = ["b1", "b2", "b3", "g1", "g2", "g3"]


def datauri(name):
    path = f"assets/{name}.jpg"
    im = Image.open(path).convert("RGB")
    im.thumbnail((800, 800))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=70)
    print(f"  {path}: {im.size} {len(buf.getvalue()) / 1024:.0f}KB")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


NEW_CATALOG = """/* ================================================================
   GARMENT CATALOG  (Boys + Girls) — real product photos
   (embedded below, so this single file works anywhere)
   ================================================================ */
const GARMENTS = [
  // ---- Boys ----
  { id:"b1", cat:"boys", name:"Blue T-Shirt",   img:"%%GARMENT_B1%%" },
  { id:"b2", cat:"boys", name:"Black Jacket",   img:"%%GARMENT_B2%%" },
  { id:"b3", cat:"boys", name:"Red Hoodie",     img:"%%GARMENT_B3%%" },
  // ---- Girls ----
  { id:"g1", cat:"girls", name:"Pink Dress",    img:"%%GARMENT_G1%%" },
  { id:"g2", cat:"girls", name:"Yellow Top",    img:"%%GARMENT_G2%%" },
  { id:"g3", cat:"girls", name:"Purple Skirt",  img:"%%GARMENT_G3%%" },
];

"""

UPLOAD_HTML = """<div class="grid" id="grid"></div>
    <div class="drop" id="garmentDrop" style="margin-top:16px;padding:18px">
      <p style="margin:0"><strong>📤 Or upload your own outfit</strong></p>
      <p style="margin:4px 0 0;color:var(--muted);font-size:.9rem">A clear photo of your own clothes works best (plain background, JPG / PNG)</p>
      <input type="file" id="garmentInput" accept="image/jpeg,image/png,image/jpg" />
    </div>
    <div class="preview" id="customGarmentBox" style="display:none;max-width:180px">
      <img id="customGarmentImg" alt="your garment" />
    </div>"""

UPLOAD_JS = """renderGrid("boys");

// ---- Upload your own garment (custom outfit) ----
const garmentDrop = $("#garmentDrop"), garmentInput = $("#garmentInput");
garmentDrop.addEventListener("click", () => garmentInput.click());
garmentInput.addEventListener("change", (e) => {
  const f = e.target.files[0];
  if (!f) return;
  if (!["image/jpeg", "image/png", "image/jpg"].includes(f.type)) {
    alert("Please choose a JPG or PNG image.");
    return;
  }
  const reader = new FileReader();
  reader.onload = (ev) => {
    selectedGarment = { id: "custom", cat: "custom", name: "Your upload", img: ev.target.result };
    document.querySelectorAll(".garment").forEach((el) => el.classList.remove("selected"));
    $("#customGarmentImg").src = ev.target.result;
    $("#customGarmentBox").style.display = "block";
    garmentTag.textContent = "✔ Garment: Your upload";
    updateGenerateState();
  };
  reader.readAsDataURL(f);
});"""


def block_replace(src, start_anchor, end_anchor, new_text, name):
    """Replace from the /* comment line containing start_anchor up to (not
    including) the /* comment line containing end_anchor."""
    assert src.count(start_anchor) == 1, f"{name}: start anchor x{src.count(start_anchor)}"
    assert src.count(end_anchor) == 1, f"{name}: end anchor x{src.count(end_anchor)}"
    i0 = src.index(start_anchor)
    i = src.rindex("/*", 0, i0)
    j0 = src.index(end_anchor)
    j = src.rindex("/*", 0, j0)
    assert i < j, f"{name}: bad order"
    return src[:i] + new_text + src[j:]


def transform(src, uris, name):
    def rep(old, new):
        nonlocal src
        n = src.count(old)
        assert n == 1, f"{name}: expected 1x, found {n}x: {old[:70]!r}"
        src = src.replace(old, new)

    # R1 — card CSS: svg -> img thumbnails
    rep(".garment svg{width:100%;height:110px;display:block}",
        ".garment img{width:100%;height:110px;object-fit:cover;display:block;border-radius:8px}")
    # R2 — whole cartoon catalog block -> real-photo catalog
    src = block_replace(src, "GARMENT CATALOG", "SMALL HELPERS", NEW_CATALOG, name)
    # R3 — delete dead svgToPng helper (+ its comment)
    src = block_replace(src, "Convert an inline <svg> string", "Turn a data-URL into a Blob", "", name)
    # R4 — render real <img> in the grid
    rep('el.innerHTML = `${garmentSVG(g.type, g.color)}<div class="name">${g.name}</div><div class="cat">${cat}</div>`;',
        'el.innerHTML = `<img src="${g.img}" alt="${g.name}" loading="lazy"/><div class="name">${g.name}</div><div class="cat">${cat}</div>`;')
    # R5 — backend call uses the image directly (no SVG conversion)
    rep("  // Convert the selected garment SVG -> PNG data URL once.\n"
        "  const garmentDataURL = await new Promise((res) => svgToPng(garmentSVG(selectedGarment.type, selectedGarment.color), 512, res));",
        "  // The selected garment is already an image (preset photo or your upload).\n"
        "  const garmentDataURL = selectedGarment.img;")
    # R6 — demo fallback uses the image too
    rep("  // garment\n"
        "  const garmentDataURL = await new Promise((res) => svgToPng(garmentSVG(selectedGarment.type, selectedGarment.color), 512, res));\n"
        "  const garment = await loadImage(garmentDataURL);",
        "  // garment (preset photo or your own upload)\n"
        "  const garment = await loadImage(selectedGarment.img);")
    # R7 — upload-your-own UI (HTML + JS)
    rep('<div class="grid" id="grid"></div>', UPLOAD_HTML)
    rep('renderGrid("boys");', UPLOAD_JS)
    # hide custom preview when a preset card is picked afterwards
    rep("function selectGarment(g) {\n  selectedGarment = g;",
        "function selectGarment(g) {\n  selectedGarment = g;\n"
        '  $("#customGarmentBox").style.display = "none"; // hide custom preview when a preset is picked')
    # R8 — clearer hint
    rep('<p class="hint">Garment images should be on a plain background.</p>',
        '<p class="hint">Pick a ready-made outfit below, or upload a photo of your own clothes. Plain-background photos give the best AI result.</p>')
    # R9 — header copy
    rep("<p>Upload your photo and try on demo clothes for boys and girls. See yourself wearing a garment in seconds using an open-source AI model.</p>",
        "<p>Upload your photo and try on outfits for boys and girls — ready-made styles, or a photo of your own clothes. Real AI, running on a free GPU.</p>")
    # Tokens -> data URIs
    for key, uri in uris.items():
        tok = f"%%GARMENT_{key}%%"
        assert src.count(tok) == 1, f"{name}: token {tok} x{src.count(tok)}"
        src = src.replace(tok, uri)
    assert "garmentSVG" not in src and "svgToPng" not in src, f"{name}: leftover svg refs"
    return src


def main():
    print("Compressing garment photos...")
    uris = {n.upper(): datauri(n) for n in ASSETS}
    for path in ["index.html", "preview/index.html"]:
        src = open(path).read()
        try:
            out = transform(src, uris, path)
        except AssertionError as e:
            print(f"{path}: FAILED: {e}")
            sys.exit(1)
        open(path, "w").write(out)
        print(f"{path}: OK, {len(out) / 1024:.0f}KB")


if __name__ == "__main__":
    main()
