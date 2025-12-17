# src/llm_cache.py
import json
import os
import hashlib

class JSONLCache:
    """

    - key: md5(text)
    - value: {"urgency":..., "money_lure":..., "authority":..., "emotion":...}
    存在 cache/llm_features.jsonl 中
    """
    def __init__(self, path: str):
        self.path = path
        self.mem = {}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    self.mem[obj["key"]] = obj["value"]

    def _key(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def get(self, text: str):
        return self.mem.get(self._key(text))

    def set(self, text: str, value: dict):
        k = self._key(text)
        self.mem[k] = value
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": k, "value": value}, ensure_ascii=False) + "\n")
