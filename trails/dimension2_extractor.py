import json
import base64
from pathlib import Path
from ray_cluster_access.sv_inference_api import sv_openai_completion
from ray_cluster_access.sv_ray_cluster_api import embed_text as sv_embed_text
IMAGES_DIR = Path(__file__).parent / "images"
METADATA_FILE = IMAGES_DIR / "metadata.json"
ATTRIBUTES_FILE = IMAGES_DIR / "attributes.json"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def extract_attributes(image_path: Path, category: str) -> dict:
    """Extract visual attributes using SV inference server (Qwen2.5-VL-72B).

    Args:
        image_path (Path): Local path to the product image.
        category (str): Product category hint (e.g. "shoe", "vegetable").

    Returns:
        dict: Parsed visual attributes with keys: item_name, brand,
              color_primary, color_secondary, material, style, occasion.
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

    response = sv_openai_completion(
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            ],
        }],
        max_tokens=500,
    )

    raw = response.choices[0].message.content.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0].strip()
    return json.loads(raw)


def embed_text(text: str) -> list:
    """Embed text using sentence-transformers via SV Ray cluster.

    Args:
        text: The text string to embed.

    Returns:
        Embedding as a list of floats.
    """
    return sv_embed_text([text], model_name=EMBED_MODEL)[0]


def build_corpus(meta: dict, attrs: dict) -> str:
    """Build weighted text corpus from metadata and extracted attributes.

    High-signal fields (brand, product) are repeated for extra weight.

    Args:
        meta: Metadata dict (brand, product, category, color).
        attrs: Extracted visual attributes dict from Qwen2.5-VL-72B.

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
    except TimeoutError:
        print(f"         └─ FAIL: timed out — skipping {item['id']}")
    except Exception as e:
        print(f"         └─ FAIL: {e}")

with open(ATTRIBUTES_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved {len(results)} / {total} items → {ATTRIBUTES_FILE}")