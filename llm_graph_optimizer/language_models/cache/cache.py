from dataclasses import dataclass, field
import logging
import pickle
from enum import Enum
from llm_graph_optimizer.language_models.cache.types import CacheSeed, LLMCacheKey
from llm_graph_optimizer.measurement.measurement import Measurement
from llm_graph_optimizer.types import LLMOutput

class CacheCategory(Enum):
    PROCESS = "process"
    PERSISTENT = "persistent"

@dataclass
class CacheKey:
    cache_key: LLMCacheKey
    prompt: str
    cache_seed: CacheSeed

    def __hash__(self):
        return hash((self.cache_key, self.prompt, self.cache_seed))

@dataclass
class CacheEntry:
    result: tuple[LLMOutput, Measurement]
    measurement: Measurement

@dataclass
class Cache:
    entries: dict[CacheKey, CacheEntry] = field(default_factory=dict)

    def get(self, cache_key: CacheKey) -> CacheEntry | None:
        return self.entries.get(cache_key)
    
    def set(self, cache_key: CacheKey, cache_entry: CacheEntry):
        self.entries[cache_key] = cache_entry

    def save(self, file_path: str):
        with open(file_path, "wb") as f:
            pickle.dump(self.entries, f)

    @classmethod
    def from_file(cls, file_path: str) -> "Cache":
        with open(file_path, "rb") as f:
            return Cache(pickle.load(f))

class CacheContainer:
    """
    Cache for language models calls
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.process_cache: Cache = Cache()
        self.persistent_cache: Cache = Cache()

    @classmethod
    def from_persistent_cache_file(cls, file_path: str, skip_on_file_not_found: bool = False) -> "CacheContainer":
        """
        Loads the persistent cache from a file and returns a new Cache object.
        """
        try:
            with open(file_path, "rb") as f:
                cache = CacheContainer()
                cache.persistent_cache = pickle.load(f)
                return cache
        except FileNotFoundError:
            if skip_on_file_not_found:
                return CacheContainer()
            else:
                raise
        
    def save_persistent_cache(self, file_path: str):
        """
        Saves the persistent cache to a file.
        """
        with open(file_path, "wb") as f:
            pickle.dump(self.persistent_cache, f)
    
    def get(self, cache_key: CacheKey) -> tuple[CacheEntry | None, CacheCategory | None]:
        """
        Get the result from the cache.
        """
        process_entry = self.process_cache.get(cache_key)
        persistent_entry = self.persistent_cache.get(cache_key)
        if process_entry:
            return process_entry, CacheCategory.PROCESS
        elif persistent_entry:
            return persistent_entry, CacheCategory.PERSISTENT
        else:
            return None, None
        
    def set(self, cache_key: CacheKey, cache_entry: CacheEntry):
        """
        Set the result in the cache.
        """
        self.process_cache.set(cache_key, cache_entry)
        self.persistent_cache.set(cache_key, cache_entry)

    def clear_process_cache(self):
        """
        Clears the process cache.
        """
        self.process_cache = {}
    
    def clear_persistent_cache(self):
        """
        Clears the persistent cache.
        """
        self.persistent_cache = {}
