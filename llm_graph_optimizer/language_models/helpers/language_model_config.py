from dataclasses import dataclass
from enum import Enum
from typing import Union


@dataclass
class Config:
    """
    Configuration for the OpenAIChat model. Entries need to be json serializable.
    """
    temperature: float = None
    max_tokens: int = None
    stop: Union[str, list[str]] = None


class LLMResponseType(Enum):
    TEXT = "text"
    TOKENS_AND_LOGPROBS = "tokens_and_logprobs"