from abc import ABC, abstractmethod
from llm_graph_optimizer.language_models.cache.types import LLMCacheKey
from llm_graph_optimizer.measurement.measurement import Measurement
from llm_graph_optimizer.types import LLMOutput

from .helpers.language_model_config import Config, LLMResponseType

class AbstractLanguageModel(ABC):
    """
    Abstract base class that defines the interface for all language models.
    """

    @abstractmethod
    def __init__(self, config: Config, llm_response_type: LLMResponseType = LLMResponseType.TEXT, execution_cost: float = 1):
        # Initialize the cache as an empty dictionary
        self._cache = {}
        self._cache_with_logprobs = {}
        self._config = config
        self._execution_cost = execution_cost
        self.llm_response_type = llm_response_type

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

    async def query(self, prompt: str, use_cache: bool = True) -> tuple[LLMOutput, Measurement]:
        """
        Query the language model with caching.
        """
        # Check if the prompt is in the cache
        if use_cache and prompt in self._cache:
            return self._cache[prompt]

        # If not in cache, query the language model
        response, measurement = await self._raw_query(prompt)

        # Store the result in the cache
        if use_cache:
            self._cache[prompt] = response

        return response, measurement
