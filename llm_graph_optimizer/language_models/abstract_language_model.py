from abc import ABC, abstractmethod

from .helpers.query_metadata import QueryMetadata
from .helpers.language_model_config import Config

class AbstractLanguageModel(ABC):
    """
    Abstract base class that defines the interface for all language models.
    """

    @abstractmethod
    def __init__(self, config: Config):
        # Initialize the cache as an empty dictionary
        self._cache = {}
        self._cache_with_logprobs = {}
        self._config = config

    @abstractmethod
    async def _raw_query(self, prompt: str) -> tuple[str, QueryMetadata]:
        """
        Query the language model with a prompt.
        """
        pass

    @abstractmethod
    async def _raw_query_with_logprobs(self, prompt: str) -> tuple[list[str, float], QueryMetadata]:
        """
        Query the language model with log probabilities.
        """
        pass

    async def query(self, prompt: str, use_cache: bool = True) -> tuple[str, QueryMetadata]:
        """
        Query the language model with caching.
        """
        # Check if the prompt is in the cache
        if use_cache and prompt in self._cache:
            return self._cache[prompt]

        # If not in cache, query the language model
        response = await self._raw_query(prompt)

        # Store the result in the cache
        if use_cache:
            self._cache[prompt] = response

        return response
    
    async def query_with_logprobs(self, prompt: str, use_cache: bool = True) -> tuple[list[str, float], QueryMetadata]:
        """
        Query the language model with log probabilities.
        """
        # Check if the prompt is in the cache
        if use_cache and prompt in self._cache_with_logprobs:
            return self._cache_with_logprobs[prompt]

        # If not in cache, query the language model
        response = await self._raw_query_with_logprobs(prompt)

        # Store the result in the cache
        if use_cache:
            self._cache_with_logprobs[prompt] = response

        return response

    def reset_cache(self):
        """
        Reset the cache by clearing all stored queries.
        """
        self._cache.clear()
        self._cache_with_logprobs.clear()
