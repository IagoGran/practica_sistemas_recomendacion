import json
from typing import List
import os

def load_playlists_from_file(filepath: str) -> List[dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("playlists", [])

def iter_playlists_from_dir(folder: str):
    for fn in os.listdir(folder):
        if fn.endswith(".json"):
            path = os.path.join(folder, fn)
            for pl in load_playlists_from_file(path):
                yield pl