import os, json, numpy as np

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_npz(path: str, **arrays):
    ensure_dir(os.path.dirname(path)); np.savez_compressed(path, **arrays)

def save_json(path: str, payload):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f: json.dump(payload, f, indent=2)
