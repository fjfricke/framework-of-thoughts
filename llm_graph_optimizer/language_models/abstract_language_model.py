from abc import ABC, abstractmethod
import copy
import logging
from llm_graph_optimizer.language_models.cache.cache import CacheContainer, CacheCategory, CacheEntry, CacheKey
from llm_graph_optimizer.language_models.cache.types import LLMCacheKey, CacheSeed
from llm_graph_optimizer.measurement.measurement import Measurement, MeasurementsWithCache
from llm_graph_optimizer.types import LLMOutput

from .helpers.language_model_config import Config, LLMResponseType

class AbstractLanguageModel(ABC):
    """
    Abstract base class that defines the interface for all language models. Inherit from this to create a new LLM.
    """

    @abstractmethod
    def __init__(self, config: Config, llm_response_type: LLMResponseType = LLMResponseType.TEXT, execution_cost: float = 1, cache: CacheContainer = None):
        """
        Initialize the language model. Should be done outside of the controller factory.

        Args:
            config (Config): Configuration for the language model.
            llm_response_type (LLMResponseType, optional): Type of response expected from the language model. Defaults to LLMResponseType.TEXT.
            execution_cost (float, optional): Cost of executing the language model. Defaults to 1.
            cache (CacheContainer, optional): Cache container. Defaults to None.
        """
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

        Returns:
            dict[str, object]: A dictionary of additional identifiers for the cache. Values need to be JSON serializable.
        """
        pass
    
    @property
    def cache_identifiers(self) -> LLMCacheKey:
        """
        Identifiers for the persistent cache.

        Returns:
            LLMCacheKey: Cache key containing the language model type, configuration, and additional identifiers.
        """
        return LLMCacheKey(
            llm_type=self.__class__,
            config=self._config,
            additional_identifiers=self.additional_cache_identifiers
        )

    @abstractmethod
    async def _raw_query(self, prompt: str) -> tuple[LLMOutput, Measurement]:
        """
        Query the language model with a prompt. Needs to be implemented by the LLM.

        Args:
            prompt (str): The input prompt for the language model.

        Returns:
            tuple[LLMOutput, Measurement]: The output from the language model and associated measurement.
        """
        pass

    @abstractmethod
    async def _raw_chat_query(self, prompt: list[dict[str, str]]) -> tuple[LLMOutput, Measurement]:
        """
        Query the language model with a prompt. Needs to be implemented by the LLM.

        Args:
            prompt (list[dict[str, str]]): The input chat prompts with role and content for the language model.

        Returns:
            tuple[LLMOutput, Measurement]: The output from the language model and associated measurement.
        """
        pass

    async def query(self, prompt: str | list[dict[str, str]], use_cache: bool = True, cache_seed: CacheSeed = None) -> tuple[LLMOutput, MeasurementsWithCache]:
        """
        Query the language model with caching.

        Args:
            prompt (str): The input prompt for the language model.
            use_cache (bool, optional): Whether to use caching for the query. Defaults to True.
            cache_seed (CacheSeed, optional): Extra identifier for cache lookup. Use when the same prompt should generate different answers in the graph. Defaults to None.

        Returns:
            tuple[LLMOutput, Measurement]: The output from the language model and associated measurement, including measurements.
        """

        # Check if the prompt is in the cache
        if use_cache and self.cache:
            cache_entry, cache_category = self.cache.get(CacheKey(self.cache_identifiers, prompt, cache_seed))
            if cache_entry:
                response, no_cache_measurement = cache_entry.result, cache_entry.measurement
                self.logger.debug(f"Cache hit for {self.cache_identifiers} with prompt {prompt} and cache seed {cache_seed}.")
                if cache_category == CacheCategory.PROCESS:
                    measurements = MeasurementsWithCache(
                        no_cache=copy.deepcopy(no_cache_measurement),
                        with_process_cache=Measurement(),
                        with_persistent_cache=Measurement()
                    )
                elif cache_category == CacheCategory.PERSISTENT:
                    measurements = MeasurementsWithCache(
                        no_cache=copy.deepcopy(no_cache_measurement),
                        with_process_cache=copy.deepcopy(no_cache_measurement),
                        with_persistent_cache=Measurement()
                    )
                    self.cache.set(CacheKey(self.cache_identifiers, prompt, cache_seed), CacheEntry(response, no_cache_measurement))
                elif cache_category == CacheCategory.VIRTUAL_PERSISTENT:
                    measurements = MeasurementsWithCache(
                        no_cache=copy.deepcopy(no_cache_measurement),
                        with_process_cache=copy.deepcopy(no_cache_measurement),
                        with_persistent_cache=copy.deepcopy(no_cache_measurement)
                    )
                    self.cache.set(CacheKey(self.cache_identifiers, prompt, cache_seed), CacheEntry(response, no_cache_measurement))
                else:
                    raise ValueError(f"Invalid cache category: {cache_category}")
                return response, measurements

        # If not in cache, query the language model
        if isinstance(prompt, str):
            response, measurement = await self._raw_query(prompt)
        else:
            response, measurement = await self._raw_chat_query(prompt)

        # Store the result in the cache
        if use_cache and self.cache:
            self.logger.debug(f"Cache miss for {self.cache_identifiers} with prompt {prompt} and cache seed {cache_seed}. Storing result in cache.")
            self.cache.set(CacheKey(self.cache_identifiers, prompt, cache_seed), CacheEntry(response, measurement))

        return response, MeasurementsWithCache(
            no_cache=copy.deepcopy(measurement),
            with_process_cache=copy.deepcopy(measurement),
            with_persistent_cache=copy.deepcopy(measurement)
        )
