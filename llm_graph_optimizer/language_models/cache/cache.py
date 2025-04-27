from dataclasses import dataclass, field
import logging
from pathlib import Path
import pickle
from enum import Enum
from llm_graph_optimizer.language_models.cache.types import CacheSeed, LLMCacheKey
from llm_graph_optimizer.measurement.measurement import Measurement
from llm_graph_optimizer.types import LLMOutput

class CacheCategory(Enum):
    PROCESS = "process"
    PERSISTENT = "persistent"
    VIRTUAL_PERSISTENT = "virtual_persistent"
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

    def __add__(self, other: "Cache") -> "Cache":
        return Cache({**self.entries, **other.entries})

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

    def __init__(self, save_file_path: Path = None):
        self.logger = logging.getLogger(__name__)
        self.process_cache: Cache = Cache()
        self.persistent_cache: Cache = Cache()
        self.virtual_persistent_cache: Cache = Cache()
        self.save_file_path: Path = save_file_path

    @classmethod
    def from_persistent_cache_file(cls, file_path: str, skip_on_file_not_found: bool = False, load_as_virtual_persistent_cache: bool = False) -> "CacheContainer":
        """
        Loads the persistent cache from a file and returns a new Cache object.
        """
        try:
            with open(file_path, "rb") as f:
                cache = CacheContainer(save_file_path=Path(file_path))
                if load_as_virtual_persistent_cache:
                    virtual_persistent_cache = pickle.load(f)
                    entries = virtual_persistent_cache.entries
                    cache.virtual_persistent_cache.entries = entries
                else:
                    persistent_cache = pickle.load(f)
                    entries = persistent_cache.entries
                    cache.persistent_cache.entries = entries

            return cache
        except FileNotFoundError:
            if skip_on_file_not_found:
                return CacheContainer(save_file_path=Path(file_path))
            else:
                raise
        
    def save_persistent_cache(self, file_path: str = None):
        """
        Saves the persistent cache to a file.
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
        Get the result from the cache.
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
        Set the result in the cache.
        """
        self.process_cache.set(cache_key, cache_entry)
        self.persistent_cache.set(cache_key, cache_entry)

    def clear_process_cache(self):
        """
        Clears the process cache.
        """
        self.process_cache.entries = {}
    
    def clear_persistent_cache(self):
        """
        Clears the persistent cache.
        """
        self.persistent_cache.entries = {}
