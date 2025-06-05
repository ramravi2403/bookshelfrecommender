import json
import os
from typing import Dict, Optional

from ocr_parser.interfaces import ICacheStorage


class JsonCacheStorage(ICacheStorage):
    def __init__(self, cache_file: str):
        self.cache_file = cache_file
        self._cache: Dict[str, Dict] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self._cache = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._cache = {}
        else:
            self._cache = {}

    def load(self) -> Dict[str, Dict]:
        return self._cache.copy()

    def save(self, data: Dict[str, Dict]) -> None:
        self._cache = data
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self._cache, f, indent=2)
        except IOError as e:
            print(f"Error saving cache: {e}")

    def get(self, key: str) -> Optional[Dict]:
        return self._cache.get(key)

    def set(self, key: str, value: Dict) -> None:
        self._cache[key] = value