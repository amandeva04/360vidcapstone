import json
import os
import torch

def save_checkpoint(path: str, model_state: dict, config: dict | None = None):
    """Save weights + (optional) config JSON sidecar."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    torch.save({"model_state": model_state, "config": config or {}}, path)
    # sidecar JSON for quick inspection
    try:
        with open(path + ".json", "w") as f:
            json.dump(config or {}, f, indent=2)
    except Exception as e:
        print(f"[WARN] could not write JSON sidecar: {e}")

def load_checkpoint(path: str, model):
    """Load weights into a model and return stored config (if any)."""
    blob = torch.load(path, map_location="cpu")
    model.load_state_dict(blob["model_state"])
    return blob.get("config", {})
