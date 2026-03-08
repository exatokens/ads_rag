import json
import torch
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# ── Constants ──────────────────────────────────────────────────────────────────
IMAGES_DIR = Path(__file__).parent / "images"
METADATA_FILE = IMAGES_DIR / "metadata.json"
ATTRIBUTES_FILE = IMAGES_DIR / "attributes.json"
CLIP_DIM = 512
UNIFIED_DIM = 384

# ── Load CLIP ──────────────────────────────────────────────────────────────────
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ── Load metadata + unified embeddings ────────────────────────────────────────
with open(METADATA_FILE) as f:
    IMAGES = json.load(f)

unified_lookup = {}
if ATTRIBUTES_FILE.exists():
    with open(ATTRIBUTES_FILE) as f:
        for entry in json.load(f):
            unified_lookup[entry["id"]] = entry["embedding"]
    print(f"Loaded unified embeddings for {len(unified_lookup)} items.")
else:
    print("WARNING: attributes.json not found — run dimension2_extractor.py first.")

# ── Qdrant setup ───────────────────────────────────────────────────────────────
client = QdrantClient(url="http://10.0.10.58:6333")

try:
    client.delete_collection("deals")
    print("Cleared existing 'deals' collection.")
except Exception:
    pass

client.create_collection(
    collection_name="deals",
    vectors_config={
        "clip_embedding": VectorParams(size=CLIP_DIM, distance=Distance.COSINE),
        "unified_embedding": VectorParams(size=UNIFIED_DIM, distance=Distance.COSINE),
    },
)

# ── Index images ───────────────────────────────────────────────────────────────
print("\nIndexing images...")
indexed_count = 0
for idx, item in enumerate(IMAGES):
    meta = item["metadata"]
    image_path = IMAGES_DIR / f"{item['id']}.jpg"
    unified_vec = unified_lookup.get(item["id"])

    if not image_path.exists():
        print(f"  SKIP {item['id']} — file not found")
        continue
    if unified_vec is None:
        print(f"  SKIP {item['id']} — no unified embedding (run dimension2_extractor.py first)")
        continue

    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            clip_vec = model.get_image_features(**inputs)
            if not isinstance(clip_vec, torch.Tensor):
                clip_vec = clip_vec.pooler_output
            clip_vec = clip_vec / clip_vec.norm(dim=-1, keepdim=True)

        client.upsert(
            collection_name="deals",
            points=[PointStruct(
                id=idx,
                vector={
                    "clip_embedding": clip_vec[0].numpy().tolist(),
                    "unified_embedding": unified_vec,
                },
                payload={"item_id": item["id"], **meta},
            )],
        )
        indexed_count += 1
        print(f"  OK  {item['id']} — {meta['brand']} {meta['product']}")
    except Exception as e:
        print(f"  FAIL {item['id']} — {meta['brand']} {meta['product']}: {e}")

print(f"\nTotal indexed: {indexed_count} items")
