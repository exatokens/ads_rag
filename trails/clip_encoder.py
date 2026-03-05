import json
import requests
import torch
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient

# ── Constants ──────────────────────────────────────────────────────────────────
IMAGES_DIR = Path(__file__).parent / "images"
ATTRIBUTES_FILE = IMAGES_DIR / "attributes.json"
OLLAMA_URL = "http://localhost:11434"

# ── Load CLIP ──────────────────────────────────────────────────────────────────
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ── Qdrant client ──────────────────────────────────────────────────────────────
client = QdrantClient(url="http://localhost:6333")

# ── Hybrid mode: check if unified embeddings exist ────────────────────────────
hybrid_mode = ATTRIBUTES_FILE.exists()
mode_label = "Hybrid (CLIP + Unified)" if hybrid_mode else "CLIP only"


def ollama_embed(text: str) -> list:
    """Embed a query string using mxbai-embed-large via Ollama.

    Args:
        text: Query text to embed.

    Returns:
        1024-dim embedding as a list of floats.
    """
    resp = requests.post(f"{OLLAMA_URL}/api/embeddings", json={
        "model": "mxbai-embed-large",
        "prompt": text,
    }, timeout=30)
    resp.raise_for_status()
    return resp.json()["embedding"]


# ── Interactive query loop ─────────────────────────────────────────────────────
print(f"\nSearch ready [{mode_label}]. Type a query (or 'quit' to exit).\n")

while True:
    query = input("Search > ").strip()
    if query.lower() in ("quit", "exit", "q"):
        print("Bye!")
        break
    if not query:
        continue

    total = client.count("deals").count

    # ── CLIP query ─────────────────────────────────────────────────────────────
    text_inputs = processor(text=[query], return_tensors="pt", padding=True)
    with torch.no_grad():
        clip_qvec = model.get_text_features(**text_inputs)
        clip_qvec = clip_qvec / clip_qvec.norm(dim=-1, keepdim=True)

    clip_hits = client.query_points(
        collection_name="deals",
        query=clip_qvec[0].numpy().tolist(),
        using="clip_embedding",
        limit=total,
        with_payload=True,
    )
    clip_scores = {h.id: h.score for h in clip_hits.points}
    payload_map = {h.id: h.payload for h in clip_hits.points}

    # ── Unified query ──────────────────────────────────────────────────────────
    unified_scores = {}
    if hybrid_mode:
        unified_qvec = ollama_embed(query)
        unified_hits = client.query_points(
            collection_name="deals",
            query=unified_qvec,
            using="unified_embedding",
            limit=total,
            with_payload=True,
        )
        unified_scores = {h.id: h.score for h in unified_hits.points}

    # ── Combine + rank ─────────────────────────────────────────────────────────
    combined = []
    for pid in clip_scores:
        c = clip_scores.get(pid, 0.0)
        u = unified_scores.get(pid, 0.0)
        final = 0.25 * c + 0.35 * u
        combined.append((pid, final, c, u))

    combined.sort(key=lambda x: x[1], reverse=True)

    print(f"\nTop results for '{query}':")
    for pid, final, c, u in combined[:5]:
        p = payload_map[pid]
        print(
            f"  final={final:.4f} | clip={c:.4f} | unified={u:.4f}"
            f" | {p['brand']} {p['product']} | {p['color']} | {p['category']}"
        )
    print()
