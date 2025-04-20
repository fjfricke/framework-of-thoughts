import logging
import pickle
from enum import Enum
from llm_graph_optimizer.language_models.cache.types import CacheSeed, LLMCacheKey
from llm_graph_optimizer.measurement.measurement import Measurement
from llm_graph_optimizer.types import LLMOutput

class CacheCategory(Enum):
    PROCESS = "process"
    PERSISTENT = "persistent"


class Cache:
    """
    Cache for language models calls
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.process_cache: dict[tuple[LLMCacheKey, str, CacheSeed], tuple[LLMOutput, Measurement]] = {}
        self.persistent_cache: dict[tuple[LLMCacheKey, str, CacheSeed], tuple[LLMOutput, Measurement]] = {}

    @classmethod
    def from_file(cls, file_path: str) -> "Cache":
        """
        Loads the persistent cache from a file and returns a new Cache object.
        """
        with open(file_path, "rb") as f:
            cache = Cache()
            cache.persistent_cache = pickle.load(f)
            return cache
        
    def save(self, file_path: str):
        """
        Saves the persistent cache to a file.
        """
        with open(file_path, "wb") as f:
            pickle.dump(self.persistent_cache, f)
    
    def get(self, cache_key: LLMCacheKey, prompt: str, cache_seed: CacheSeed) -> tuple[tuple[LLMOutput, Measurement] | None, CacheCategory | None]:
        """
        Get the result from the cache.
        """
        if (cache_key, prompt, cache_seed) in self.process_cache:
            return self.process_cache[(cache_key, prompt, cache_seed)], CacheCategory.PROCESS
        elif (cache_key, prompt, cache_seed) in self.persistent_cache:
            return self.persistent_cache[(cache_key, prompt, cache_seed)], CacheCategory.PERSISTENT
        else:
            return None, None
        
    def set(self, cache_key: LLMCacheKey, prompt: str, cache_seed: CacheSeed, result: tuple[LLMOutput, Measurement]):
        """
        Set the result in the cache.
        """
        self.process_cache[(cache_key, prompt, cache_seed)] = result
        self.persistent_cache[(cache_key, prompt, cache_seed)] = result

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
