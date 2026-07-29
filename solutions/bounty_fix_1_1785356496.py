"""
High-Performance In-Memory Item Repository
Provides thread-safe O(1) add and delete operations with batch support and event callbacks.
"""

from typing import TypeVar, Generic, Dict, List, Optional, Callable, Set, Any
import threading

T = TypeVar("T")


class RepositoryError(Exception):
    """Base exception for dynamic repository errors."""
    pass


class ItemNotFoundError(RepositoryError):
    """Raised when attempting to delete a non-existent item."""
    pass


class ItemAlreadyExistsError(RepositoryError):
    """Raised when attempting to add an item with a duplicate key."""
    pass


class FastRepository(Generic[T]):
    """
    Thread-safe, high-performance in-memory dynamic store.
    Provides O(1) lookup, addition, and deletion operations.
    """

    def __init__(self, overwrite_on_add: bool = False) -> None:
        self._store: Dict[str, T] = {}
        self._lock: threading.RLock = threading.RLock()
        self._overwrite_on_add: bool = overwrite_on_add

        self._on_add_listeners: List[Callable[[str, T], None]] = []
        self._on_delete_listeners: List[Callable[[str, Optional[T]], None]] = []

    def add(self, key: str, item: T) -> None:
        """
        Adds an item to the repository.
        
        Time Complexity: O(1)
        """
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Key must be a non-empty string.")

        with self._lock:
            if not self._overwrite_on_add and key in self._store:
                raise ItemAlreadyExistsError(f"Item with key '{key}' already exists.")
            
            self._store[key] = item

        # Trigger listeners outside lock to prevent deadlocks
        for listener in self._on_add_listeners:
            listener(key, item)

    def delete(self, key: str) -> T:
        """
        Deletes an item from the repository by key and returns it.
        
        Time Complexity: O(1)
        """
        with self._lock:
            if key not in self._store:
                raise ItemNotFoundError(f"Cannot delete. Key '{key}' not found.")
            
            removed_item = self._store.pop(key)

        for listener in self._on_delete_listeners:
            listener(key, removed_item)

        return removed_item

    def add_batch(self, items: Dict[str, T]) -> None:
        """
        Atomically adds multiple key-value pairs to the repository.
        
        Time Complexity: O(N) where N is the batch size.
        """
        with self._lock:
            if not self._overwrite_on_add:
                duplicates = set(items.keys()).intersection(self._store.keys())
                if duplicates:
                    raise ItemAlreadyExistsError(
                        f"Batch operation aborted. Keys already exist: {list(duplicates)}"
                    )
            
            self._store.update(items)

        for key, item in items.items():
            for listener in self._on_add_listeners:
                listener(key, item)

    def delete_batch(self, keys: List[str]) -> List[T]:
        """
        Atomically deletes multiple items from the repository.
        
        Time Complexity: O(N) where N is the number of keys to delete.
        """
        removed_items: List[T] = []
        with self._lock:
            missing = [k for k in keys if k not in self._store]
            if missing:
                raise ItemNotFoundError(
                    f"Batch delete aborted. Keys not found: {missing}"
                )

            for key in keys:
                item = self._store.pop(key)
                removed_items.append(item)

        for key in keys:
            for listener in self._on_delete_listeners:
                listener(key, None)

        return removed_items

    def contains(self, key: str) -> bool:
        """Checks if a key exists in the repository."""
        with self._lock:
            return key in self._store

    def get(self, key: str) -> Optional[T]:
        """Retrieves an item without removing it."""
        with self._lock:
            return self._store.get(key)

    def clear(self) -> None:
        """Clears all stored items."""
        with self._lock:
            self._store.clear()

    def count(self) -> int:
        """Returns total item count."""
        with self._lock:
            return len(self._store)

    def register_on_add(self, callback: Callable[[str, T], None]) -> None:
        """Registers a callback for item add events."""
        self._on_add_listeners.append(callback)

    def register_on_delete(self, callback: Callable[[str, Optional[T]], None]) -> None:
        """Registers a callback for item delete events."""
        self._on_delete_listeners.append(callback)



import unittest

class TestFastRepository(unittest.TestCase):
    """Test suite for validating 'add' and 'delete' mechanics."""

    def setUp(self) -> None:
        self.repo: FastRepository[Dict[str, Any]] = FastRepository()

    def test_add_and_get(self) -> None:
        item = {"name": "Test Item", "val": 42}
        self.repo.add("item_1", item)
        self.assertTrue(self.repo.contains("item_1"))
        self.assertEqual(self.repo.get("item_1"), item)

    def test_add_duplicate_raises_error(self) -> None:
        self.repo.add("unique_key", {"data": 1})
        with self.assertRaises(ItemAlreadyExistsError):
            self.repo.add("unique_key", {"data": 2})

    def test_delete_success(self) -> None:
        item = {"data": "delete_me"}
        self.repo.add("key_to_delete", item)
        deleted = self.repo.delete("key_to_delete")
        
        self.assertEqual(deleted, item)
        self.assertFalse(self.repo.contains("key_to_delete"))
        self.assertEqual(self.repo.count(), 0)

    def test_delete_non_existent_raises_error(self) -> None:
        with self.assertRaises(ItemNotFoundError):
            self.repo.delete("non_existent_key")

    def test_batch_add_and_delete(self) -> None:
        batch_data = {
            "k1": {"val": 1},
            "k2": {"val": 2},
            "k3": {"val": 3},
        }
        self.repo.add_batch(batch_data)
        self.assertEqual(self.repo.count(), 3)

        deleted = self.repo.delete_batch(["k1", "k2"])
        self.assertEqual(len(deleted), 2)
        self.assertEqual(self.repo.count(), 1)
        self.assertTrue(self.repo.contains("k3"))


if __name__ == "__main__":
    unittest.main()