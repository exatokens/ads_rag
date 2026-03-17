"""Minimal script to check which models are available on the SV inference server."""
import requests
from ray_cluster_access import config

base_url = config["inference_api"]["base_url"]
models_url = f"{base_url}/v1/models"

print(f"Querying: {models_url}\n")
try:
    resp = requests.get(models_url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    models = data.get("data", [])
    if models:
        print(f"Available models ({len(models)}):")
        for m in models:
            print(f"  {m['id']}")
    else:
        print("No models found in response.")
        print("Raw response:", data)
except Exception as e:
    print(f"FAIL: {e}")
