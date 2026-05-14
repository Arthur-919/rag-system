"""
LRU 缓存 —— QA 结果缓存，减少重复 LLM 调用
"""

from collections import OrderedDict


class LRUCache:
    def __init__(self, maxsize=128):
        self._cache = OrderedDict()
        self.maxsize = maxsize

    def get(self, key):
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key, value):
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)

    def clear(self):
        self._cache.clear()


qa_cache = LRUCache(maxsize=128)
