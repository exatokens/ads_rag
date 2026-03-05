import json
import base64
import requests
from pathlib import Path

IMAGES_DIR = Path(__file__).parent / "images"
METADATA_FILE = IMAGES_DIR / "metadata.json"
ATTRIBUTES_FILE = IMAGES_DIR / "attributes.json"
OLLAMA_URL = "http://localhost:11434"

# ── Timeouts ───────────────────────────────────────────────────────────────────
VISION_TIMEOUT = 300   # 5 min per image (vision model is slow)
EMBED_TIMEOUT  = 30    # 30 sec per embedding (fast model)
WARMUP_TIMEOUT = 600   # 10 min for initial model load on Mac


def warmup_models() -> None:
    """Pre-load both models into RAM before the main loop.

    Without this, the first real image call silently freezes for several
    minutes while Ollama loads the vision model — making it look like the
    script is broken.
    """
    print("── Pre-warming models ────────────────────────────────────────────────")

    print("  [1/2] Loading llama3.2-vision into memory...")
    print("        (first load on Mac can take 3-10 min — this is normal)")
    try:
        resp = requests.post(f"{OLLAMA_URL}/api/generate", json={
            "model": "llama3.2-vision",
            "prompt": "ready",
            "stream": False,
        }, timeout=WARMUP_TIMEOUT)
        resp.raise_for_status()
        print("  [1/2] llama3.2-vision ready ✓")
    except requests.exceptions.Timeout:
        print("  [1/2] WARNING: vision model warmup timed out after 10 min.")
        print("        Try running `ollama run llama3.2-vision` in a separate")
        print("        terminal first to pre-load it, then re-run this script.")
        raise SystemExit(1)
    except Exception as e:
        print(f"  [1/2] ERROR: could not reach Ollama — is it running? ({e})")
        raise SystemExit(1)

    print("  [2/2] Loading mxbai-embed-large into memory...")
    try:
        resp = requests.post(f"{OLLAMA_URL}/api/embeddings", json={
            "model": "mxbai-embed-large",
            "prompt": "ready",
        }, timeout=EMBED_TIMEOUT)
        resp.raise_for_status()
        print("  [2/2] mxbai-embed-large ready ✓")
    except Exception as e:
        print(f"  [2/2] ERROR: could not load embed model ({e})")
        raise SystemExit(1)

    print("─────────────────────────────────────────────────────────────────────\n")


def extract_attributes(image_path: Path, category: str) -> dict:
    """Call llama3.2-vision to extract structured visual attributes from an image.

    Args:
        image_path: Path to the local image file.
        category: Product category hint (e.g. "shoe", "vegetable", "seafood").

    Returns:
        Parsed dict of visual attributes.
    """
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    prompt = (
        f"This is a product image. Category: {category}. "
        "Return a JSON object with these fields: "
        "item_name, brand (or 'unknown'), color_primary, color_secondary (or 'none'), "
        "material (or 'none'), style, occasion. "
        "Only return valid JSON. No explanation."
    )

    resp = requests.post(f"{OLLAMA_URL}/api/generate", json={
        "model": "llama3.2-vision",
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
        "format": "json",
    }, timeout=VISION_TIMEOUT)
    resp.raise_for_status()
    import pdb;pdb.set_trace()
    raw = resp.json()["response"]
    return json.loads(raw) if isinstance(raw, str) else raw


def embed_text(text: str) -> list:
    """Embed text using mxbai-embed-large via Ollama.

    Args:
        text: The text string to embed.

    Returns:
        1024-dim embedding as a list of floats.
    """
    resp = requests.post(f"{OLLAMA_URL}/api/embeddings", json={
        "model": "mxbai-embed-large",
        "prompt": text,
    }, timeout=EMBED_TIMEOUT)
    resp.raise_for_status()
    return resp.json()["embedding"]


def build_corpus(meta: dict, attrs: dict) -> str:
    """Build weighted text corpus from metadata and extracted attributes.

    High-signal fields (brand, product) are repeated for extra weight.

    Args:
        meta: Metadata dict (brand, product, category, color).
        attrs: Extracted visual attributes dict from llama3.2-vision.

    Returns:
        Single space-joined text string ready for embedding.
    """
    skip = {"none", "unknown", "n/a", ""}
    brand = meta["brand"] if meta["brand"].lower() not in skip else ""
    parts = [
        brand, brand,                          # repeated for weight
        meta["product"], meta["product"],      # repeated for weight
        meta["category"],
        meta["color"],
        attrs.get("color_primary", ""),
        attrs.get("color_secondary", ""),
        attrs.get("material", ""),
        attrs.get("style", ""),
        attrs.get("occasion", ""),
    ]
    return " ".join(p for p in parts if p.lower() not in skip)


# ── Main ───────────────────────────────────────────────────────────────────────
with open(METADATA_FILE) as f:
    items = json.load(f)

# Pre-load both models before the loop so there's no silent freeze mid-run
warmup_models()

results = []
total = len([i for i in items if (IMAGES_DIR / f"{i['id']}.jpg").exists()])
print(f"Extracting visual attributes + building embeddings ({total} images)...\n")

for idx, item in enumerate(items, 1):
    image_path = IMAGES_DIR / f"{item['id']}.jpg"
    meta = item["metadata"]

    if not image_path.exists():
        print(f"  SKIP [{idx}/{total}] {item['id']} — file not found")
        continue

    print(f"  [{idx}/{total}] {item['id']} — {meta['brand']} {meta['product']}")
    try:
        print(f"         └─ extracting attributes...")
        attrs = extract_attributes(image_path, meta["category"])

        print(f"         └─ building corpus + embedding...")
        corpus = build_corpus(meta, attrs)
        embedding = embed_text(corpus)

        results.append({
            "id": item["id"],
            "attributes": attrs,
            "corpus": corpus,
            "embedding": embedding,
        })
        print(f"         └─ OK: {corpus}")
    except requests.exceptions.Timeout:
        print(f"         └─ FAIL: timed out — skipping {item['id']}")
    except Exception as e:
        print(f"         └─ FAIL: {e}")

with open(ATTRIBUTES_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved {len(results)} / {total} items → {ATTRIBUTES_FILE}")