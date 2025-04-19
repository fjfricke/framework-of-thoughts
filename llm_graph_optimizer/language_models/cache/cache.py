import logging
from llm_graph_optimizer.language_models.abstract_language_model import AbstractLanguageModel
from llm_graph_optimizer.language_models.cache.types import LLMCacheKey
from llm_graph_optimizer.measurement.measurement import Measurement
from llm_graph_optimizer.types import LLMOutput


class Cache:
    """
    Cache for language models calls
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.process_cache: dict[tuple[LLMCacheKey, str], tuple[LLMOutput, Measurement]] = {}

    def get_key(self, llm: AbstractLanguageModel) -> str:
        """
        Get the key for a language model.
        """