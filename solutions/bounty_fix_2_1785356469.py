"""
Teste 2 - High-Performance Data Processing & Caching Engine
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from typing import Any, Callable, Coroutine, Dict, Generic, List, Optional, TypeVar
import threading

TKey = TypeVar("TKey")
TValue = TypeVar("TValue")


class ExpiringLRUCache(Generic[TKey, TValue]):
    """
    Thread-safe Least Recently Used (LRU) Cache with Time-To-Live (TTL) support.
    Provides O(1) time complexity for get and put operations.
    """

    def __init__(self, capacity: int = 1024, default_ttl_seconds: Optional[float] = 300.0) -> None:
        if capacity <= 0:
            raise ValueError("Capacity must be greater than zero.")
        self._capacity = capacity
        self._default_ttl = default_ttl_seconds
        self._cache: OrderedDict[TKey, tuple[TValue, Optional[float]]] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: TKey) -> Optional[TValue]:
        with self._lock:
            if key not in self._cache:
                return None

            val, expiry = self._cache[key]

            if expiry is not None and time.monotonic() > expiry:
                del self._cache[key]
                return None

            # Move accessed key to end (mark as recently used)
            self._cache.move_to_end(key)
            return val

    def put(self, key: TKey, value: TValue, ttl_seconds: Optional[float] = None) -> None:
        with self._lock:
            ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
            expiry = (time.monotonic() + ttl) if ttl is not None else None

            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (value, expiry)

            # Evict LRU items if capacity exceeded
            if len(self._cache) > self._capacity:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)


class BatchProcessor(Generic[TKey, TValue]):
    """
    Asynchronous task batch processor with concurrency rate limiting.
    """

    def __init__(self, max_concurrency: int = 100) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def execute_task(
        self, task_id: TKey, coro: Callable[[TKey], Coroutine[Any, Any, TValue]]
    ) -> TValue:
        async with self._semaphore:
            return await coro(task_id)

    async def process_batch(
        self,
        items: List[TKey],
        worker_coro: Callable[[TKey], Coroutine[Any, Any, TValue]],
        return_exceptions: bool = False,
    ) -> List[Any]:
        """
        Executes a batch of tasks concurrently up to the configured limit.
        """
        tasks = [self.execute_task(item, worker_coro) for item in items]
        return await asyncio.gather(*tasks, return_exceptions=return_exceptions)


def run_teste_2_pipeline(data: List[int]) -> Dict[str, Any]:
    """
    Sample entry point running data transformation and caching pipeline.
    """
    cache: ExpiringLRUCache[int, int] = ExpiringLRUCache(capacity=500, default_ttl_seconds=60)
    
    processed = []
    for item in data:
        cached = cache.get(item)
        if cached is not None:
            processed.append(cached)
        else:
            result = item * item
            cache.put(item, result)
            processed.append(result)

    return {
        "input_count": len(data),
        "cached_items_count": len(cache),
        "sample_output": processed[:5],
    }


if __name__ == "__main__":
    test_input = [1, 2, 3, 4, 5, 2, 3, 6]
    output = run_teste_2_pipeline(test_input)
    print("Execution Result:", output)