from dataclasses import dataclass
from typing import Union


@dataclass
class Config:
    """
    Configuration for the OpenAIChat model.
    """
    temperature: float
    max_tokens: int = None
    stop: Union[str, list[str]] = None