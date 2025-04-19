from dataclasses import dataclass
from llm_graph_optimizer.language_models.helpers.language_model_config import Config


@dataclass
class LLMCacheKey:
    llm_type: type
    config: Config
    additional_identifiers: dict[str, object]