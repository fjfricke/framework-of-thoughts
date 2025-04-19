from dataclasses import dataclass, asdict
from enum import Enum
from typing import Union
import json


@dataclass
class Config:
    """
    Configuration for the OpenAIChat model. Entries need to be json serializable.
    """
    temperature: float = None
    max_tokens: int = None
    stop: Union[str, list[str]] = None

    def __hash__(self):
        return hash(json.dumps(asdict(self), sort_keys=True))


class LLMResponseType(Enum):
    TEXT = "text"
    TOKENS_AND_LOGPROBS = "tokens_and_logprobs"