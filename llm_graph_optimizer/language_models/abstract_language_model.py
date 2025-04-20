from abc import ABC, abstractmethod
import logging
from llm_graph_optimizer.language_models.cache.cache import CacheContainer, CacheCategory, CacheEntry, CacheKey
from llm_graph_optimizer.language_models.cache.types import LLMCacheKey, CacheSeed
from llm_graph_optimizer.measurement.measurement import Measurement, MeasurementsWithCache
from llm_graph_optimizer.types import LLMOutput

from .helpers.language_model_config import Config, LLMResponseType

class AbstractLanguageModel(ABC):
    """
    Abstract base class that defines the interface for all language models.
    """

    @abstractmethod
    def __init__(self, config: Config, llm_response_type: LLMResponseType = LLMResponseType.TEXT, execution_cost: float = 1, cache: CacheContainer = None):
        self._config = config
        self._execution_cost = execution_cost
        self.llm_response_type = llm_response_type
        self.cache = cache
        self.logger = logging.getLogger(__name__)
    @property
    @abstractmethod
    def additional_cache_identifiers(self) -> dict[str, object]:
        """
        Additional identifiers for the persistent cache besides Config and class.
        :return: A dictionary of additional identifiers for the cache. Values need to be json serializable.
        """
        pass
    
    @property
    def cache_identifiers(self) -> LLMCacheKey:
        """
        Identifiers for the persistent cache.
        """
        return LLMCacheKey(
            llm_type=self.__class__,
            config=self._config,
            additional_identifiers=self.additional_cache_identifiers
        )

    @abstractmethod
    async def _raw_query(self, prompt: str) -> tuple[LLMOutput, Measurement]:
        """
        Query the language model with a prompt.
        """
        pass

    async def query(self, prompt: str, use_cache: bool = True, cache_seed: CacheSeed = None) -> tuple[LLMOutput, Measurement]:
        """
        Query the language model with caching.
        """

        # Check if the prompt is in the cache
        if use_cache and self.cache:
            cache_entry, cache_category = self.cache.get(CacheKey(self.cache_identifiers, prompt, cache_seed))
            if cache_entry:
                response, no_cache_measurement = cache_entry.result, cache_entry.measurement
                self.logger.debug(f"Cache hit for {self.cache_identifiers} with prompt {prompt} and cache seed {cache_seed}.")
                if cache_category == CacheCategory.PROCESS:
                    measurements = MeasurementsWithCache(
                        no_cache=no_cache_measurement,
                        with_process_cache=Measurement(),
                        with_persistent_cache=Measurement()
                    )
                elif cache_category == CacheCategory.PERSISTENT:
                    measurements = MeasurementsWithCache(
                        no_cache=no_cache_measurement,
                        with_process_cache=no_cache_measurement,
                        with_persistent_cache=Measurement()
                    )
                else:
                    raise ValueError(f"Invalid cache category: {cache_category}")
                return response, measurements

        # If not in cache, query the language model
        response, measurement = await self._raw_query(prompt)

        # Store the result in the cache
        if use_cache and self.cache:
            self.logger.debug(f"Cache miss for {self.cache_identifiers} with prompt {prompt} and cache seed {cache_seed}. Storing result in cache.")
            self.cache.set(CacheKey(self.cache_identifiers, prompt, cache_seed), CacheEntry(response, measurement))

        return response, MeasurementsWithCache(
            no_cache=measurement,
            with_process_cache=measurement,
            with_persistent_cache=measurement
        )
