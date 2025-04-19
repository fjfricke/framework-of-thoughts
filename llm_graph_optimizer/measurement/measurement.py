from dataclasses import dataclass, fields
from typing import Optional, Generic, TypeVar

T = TypeVar('T')
class SequentialCost(Generic[T]):
    def __repr__(self):
        return "<SequentialCost>"

@dataclass
class Measurements:
    request_tokens: Optional[int] = None
    response_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    request_cost: Optional[float] = None
    response_cost: Optional[float] = None
    total_cost: Optional[float] = None
    execution_duration: SequentialCost[Optional[float]] = None
    execution_cost: SequentialCost[Optional[float]] = None

    @staticmethod
    def _add_entries(entry_1, entry_2):
        if entry_1 is None:
            return entry_2
        if entry_2 is None:
            return entry_1
        return entry_1 + entry_2

    def __add__(self, other: 'Measurements') -> 'Measurements':
        combined_attributes = {
            field.name: self._add_entries(getattr(self, field.name), getattr(other, field.name))
            for field in fields(self)
        }

        return Measurements(**combined_attributes)
