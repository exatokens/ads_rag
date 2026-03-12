import json
from pathlib import Path
from ray_cluster_access.sv_inference_api import sv_openai_completion, _convert_image_to_base64

from ray_cluster_access.sv_ray_cluster_api import embed_text as sv_embed_text
IMAGES_DIR = Path(__file__).parent / "images"
METADATA_FILE = IMAGES_DIR / "metadata.json"
ATTRIBUTES_FILE = IMAGES_DIR / "attributes.json"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def extract_attributes(image_path: Path, category: str, meta: dict = None) -> dict:
    """Extract visual attributes using Qwen3.5-27B vision model on the SV inference server.

    Args:
        image_path (Path): Local path to the product image.
        category (str): Product category hint (e.g. "shoe", "vegetable").
        meta (dict): Metadata dict with brand, product, color, etc.

    Returns:
        dict: Parsed visual attributes with keys: item_name, brand,
              color_primary, color_secondary, material, style, occasion.
    """
    meta = meta or {}
    prompt = (
        f"Look at this product image carefully.\n"
        f"Known metadata — category: {category}, brand: {meta.get('brand', 'unknown')}, "
        f"product: {meta.get('product', '')}, color: {meta.get('color', '')}.\n\n"
        f"Return ONLY a JSON object with these fields and no other text: "
        f"item_name, brand (or 'unknown'), color_primary, color_secondary (or 'none'), "
        f"material (or 'none'), style, occasion."
    )

    image_b64 = _convert_image_to_base64(str(image_path))

    response = sv_openai_completion(
        messages=[
            {"role": "system", "content": "You are a product cataloger. Output only valid JSON. No explanation."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        max_tokens=300,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    raw = response.choices[0].message.content.strip()
    # Extract JSON from anywhere in the response (handles thinking-model preamble)
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in response: {raw[:200]}")
    return json.loads(raw[start:end])


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
        attrs: Extracted visual attributes dict from Qwen3.5-27B.

    Returns:
        Single space-joined text string ready for embedding.
    """
    skip = {"none", "unknown", "n/a", ""}
    brand = meta["brand"] if meta["brand"].lower() not in skip else ""
    style = attrs.get("style", "")
    occasion = attrs.get("occasion", "")
    parts = [
        brand, brand,                          # repeated for weight
        meta["product"], meta["product"],      # repeated for weight
        meta["category"],
        meta["color"],
        attrs.get("color_primary", ""),
        attrs.get("color_secondary", ""),
        attrs.get("material", ""),
        style, style,                          # repeated for weight
        occasion, occasion,                    # repeated for weight
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
        attrs = extract_attributes(image_path, meta["category"], meta)

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