from dataclasses import dataclass, field, fields
import logging
from pathlib import Path
import pickle
from enum import Enum

import numpy as np
from llm_graph_optimizer.language_models.cache.types import CacheSeed, LLMCacheKey
from llm_graph_optimizer.measurement.measurement import Measurement
from llm_graph_optimizer.types import LLMOutput

class CacheCategory(Enum):
    """
    Enum representing different categories of cache:
    PROCESS: Cache for process-level results.
    PERSISTENT: Cache for persistent results.
    VIRTUAL_PERSISTENT: Used to "mimik" the behaviour of having no cache. Used for measurement purposes when restarting a study.
    """
    PROCESS = "process"
    PERSISTENT = "persistent"
    VIRTUAL_PERSISTENT = "virtual_persistent"

@dataclass
class CacheKey:
    """
    Represents a unique key for cache entries.
    """
    cache_key: LLMCacheKey
    prompt: str | list[dict[str, str]]
    cache_seed: CacheSeed

    def __hash__(self):
        # Convert the list of dictionaries into a tuple of frozensets (hashable representation)
        if isinstance(self.prompt, list):
            hashable_prompt = tuple(frozenset(d.items()) for d in self.prompt)
        else:
            hashable_prompt = self.prompt  # If it's a string, use it directly

        return hash((self.cache_key, hashable_prompt, self.cache_seed))

@dataclass
class CacheEntry:
    """
    Represents an entry in the cache, containing the LLM result and measurement.
    """
    result: LLMOutput
    measurement: Measurement

@dataclass
class Cache:
    """
    Represents a cache for storing and retrieving entries.
    """
    entries: dict[CacheKey, CacheEntry] = field(default_factory=dict)

    def get(self, cache_key: CacheKey) -> CacheEntry | None:
        """
        Retrieve a cache entry by its key.

        :param cache_key: The key of the cache entry to retrieve.
        :return: The corresponding CacheEntry, or None if not found.
        """
        return self.entries.get(cache_key)
    
    def set(self, cache_key: CacheKey, cache_entry: CacheEntry):
        """
        Add or update a cache entry.

        :param cache_key: The key of the cache entry.
        :param cache_entry: The cache entry to store.
        """
        self.entries[cache_key] = cache_entry

    def __add__(self, other: "Cache") -> "Cache":
        """
        Combine two caches into a new cache.

        :param other: The other cache to combine with.
        :return: A new Cache containing entries from both caches.
        """
        return Cache({**self.entries, **other.entries})

    def save(self, file_path: str):
        """
        Save the cache to a file.

        :param file_path: The file path to save the cache.
        """
        with open(file_path, "wb") as f:
            pickle.dump(self.entries, f)

    @classmethod
    def from_file(cls, file_path: str) -> "Cache":
        """
        Load a cache from a file.

        :param file_path: The file path to load the cache from.
        :return: A Cache instance loaded from the file.
        """
        with open(file_path, "rb") as f:
            return Cache(pickle.load(f))

class CacheContainer:
    """
    Container for managing multiple types of caches for language model calls.
    """

    def __init__(self, save_file_path: Path = None):
        """
        Initialize the CacheContainer.

        :param save_file_path: Optional file path to save the persistent cache. Used during a study after each dataset evaluation.
        """
        self.logger = logging.getLogger(__name__)
        self.process_cache: Cache = Cache()
        self.persistent_cache: Cache = Cache()
        self.virtual_persistent_cache: Cache = Cache()
        self.save_file_path: Path = save_file_path

    @classmethod
    def from_persistent_cache_file(cls, file_path: str, skip_on_file_not_found: bool = False, load_as_virtual_persistent_cache: bool = False) -> "CacheContainer":
        """
        Load a persistent cache from a file and return a new CacheContainer instance.

        :param file_path: The file path to load the persistent cache from.
        :param skip_on_file_not_found: Whether to skip loading if the file is not found.
        :param load_as_virtual_persistent_cache: Whether to load the cache as a virtual persistent cache (treating it as empty on measurements).
        :return: A CacheContainer instance with the loaded cache.
        """
        try:
            with open(file_path, "rb") as f:
                cache = CacheContainer(save_file_path=Path(file_path))
                
                loaded_cache = pickle.load(f)
                loaded_cache_entries = loaded_cache.entries
                for key, value in loaded_cache_entries.items():
                    if value is None:
                        continue
                    if not isinstance(value.measurement, Measurement):
                        continue
                    for field in fields(value.measurement):
                        field_value = getattr(value.measurement, field.name)
                        if isinstance(field_value, (int, float)):
                            setattr(value.measurement, field.name, np.float64(field_value))
                    if not all([isinstance(getattr(value.measurement, field.name), np.float64) for field in fields(value.measurement)]):
                        continue
                    if load_as_virtual_persistent_cache:
                        cache.virtual_persistent_cache.entries = loaded_cache_entries
                    else:
                        cache.persistent_cache.entries = loaded_cache_entries

            return cache
        except FileNotFoundError:
            if skip_on_file_not_found:
                return CacheContainer(save_file_path=Path(file_path))
            else:
                raise
        
    def save_persistent_cache(self, file_path: str = None):
        """
        Save the persistent cache to a file. Concatenates the virtual persistent cache and the persistent cache.

        :param file_path: Optional file path to save the persistent cache. If not provided, the save_file_path is used.
        """
        if file_path is None:
            file_path = self.save_file_path
            if file_path is None:
                logging.warning("No file path to save persistent cache to. Skipping.")
                return
        with open(file_path, "wb") as f:
            pickle.dump(self.virtual_persistent_cache + self.persistent_cache, f)
    
    def get(self, cache_key: CacheKey) -> tuple[CacheEntry | None, CacheCategory | None]:
        """
        Retrieve a cache entry and its category by its key.

        :param cache_key: The key of the cache entry to retrieve.
        :return: A tuple containing the CacheEntry and its CacheCategory, or (None, None) if not found.
        """
        process_entry = self.process_cache.get(cache_key)
        persistent_entry = self.persistent_cache.get(cache_key)
        virtual_persistent_entry = self.virtual_persistent_cache.get(cache_key)
        if process_entry:
            return process_entry, CacheCategory.PROCESS
        elif persistent_entry:
            return persistent_entry, CacheCategory.PERSISTENT
        elif virtual_persistent_entry:
            return virtual_persistent_entry, CacheCategory.VIRTUAL_PERSISTENT
        else:
            return None, None
        
        
    def set(self, cache_key: CacheKey, cache_entry: CacheEntry):
        """
        Add or update a cache entry in both the process and persistent caches.

        :param cache_key: The key of the cache entry.
        :param cache_entry: The cache entry to store.
        """
        self.process_cache.set(cache_key, cache_entry)
        self.persistent_cache.set(cache_key, cache_entry)

    def clear_process_cache(self):
        """
        Clear all entries from the process cache. Used between process calls in database evaluations and optimization studies.
        """
        self.process_cache.entries = {}
    
    def clear_persistent_cache(self):
        """
        Clear all entries from the persistent cache.
        """
        self.persistent_cache.entries = {}
