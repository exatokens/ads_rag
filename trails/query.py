"""Query the Qdrant 'deals' collection using natural language."""
import sys
import torch
from transformers import CLIPModel, CLIPProcessor
from qdrant_client import QdrantClient
from ray_cluster_access.sv_inference_api import sv_openai_completion

TOP_K = 3

_clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
_clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
_clip_model.eval()


def expand_query(query: str) -> str:
    """Use the LLM to rewrite the query with broader synonyms for better recall."""
    response = sv_openai_completion(
        messages=[
            {"role": "system", "content": "You are a search query expander. Output only the expanded query text, no explanation."},
            {"role": "user", "content": (
                f"Expand this product search query with synonyms and related terms "
                f"to improve retrieval. Keep it concise (under 20 words).\n\nQuery: {query}"
            )},
        ],
        max_tokens=60,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    expanded = response.choices[0].message.content.strip()
    print(f"  Expanded: \"{expanded}\"")
    return expanded


def embed_query(text: str) -> list:
    inputs = _clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        features = _clip_model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    return features[0].numpy().tolist()


def search(query: str, top_k: int = TOP_K):
    client = QdrantClient(url="http://10.0.10.58:6333")
    expanded = expand_query(query)
    vec = embed_query(expanded)

    hits = client.query_points(
        collection_name="deals",
        query=vec,
        using="clip_embedding",
        limit=top_k,
        with_payload=True,
    ).points

    print(f"\nQuery: \"{query}\"\n{'─' * 50}")
    for i, hit in enumerate(hits, 1):
        p = hit.payload
        print(f"  {i}. [{hit.score:.3f}]  {p.get('brand', '')} {p.get('product', '')}  "
              f"({p.get('category', '')} / {p.get('color', '')})")
    print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        search(" ".join(sys.argv[1:]))
    else:
        print("Interactive search — type a query and press Enter. Ctrl+C to quit.\n")
        while True:
            try:
                query = input("Search: ").strip()
                if query:
                    search(query)
            except KeyboardInterrupt:
                print("\nBye!")
                break
