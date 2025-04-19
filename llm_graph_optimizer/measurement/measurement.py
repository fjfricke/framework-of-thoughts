from dataclasses import dataclass, fields
from typing import Optional, Generic, TypeVar

T = TypeVar('T')
class SequentialCost(Generic[T]):
    def __repr__(self):
        return "<SequentialCost>"

@dataclass
class Measurement:
    request_tokens: Optional[int] = 0
    response_tokens: Optional[int] = 0
    total_tokens: Optional[int] = 0
    request_cost: Optional[float] = 0
    response_cost: Optional[float] = 0
    total_cost: Optional[float] = 0
    execution_duration: SequentialCost[Optional[float]] = 0
    execution_cost: SequentialCost[Optional[float]] = 0

    def __add__(self, other: 'Measurement') -> 'Measurement':
        combined_attributes = {
            field.name: getattr(self, field.name) + getattr(other, field.name)
            for field in fields(self)
        }

        return Measurement(**combined_attributes)


@dataclass
class MeasurementsWithCache:
    no_cache: Measurement = None
    with_process_cache: Measurement = None
    with_persistent_cache: Measurement = None

    @classmethod
    def from_no_cache_measurement(cls, measurement: Measurement) -> 'MeasurementsWithCache':
        return cls(
            no_cache=measurement,
            with_process_cache=measurement,
            with_persistent_cache=measurement
        )
    
    @staticmethod
    def _add_entries(entry_1, entry_2):
        if entry_1 is None:
            entry_1 = Measurement()
        if entry_2 is None:
            entry_2 = Measurement()
        return entry_1 + entry_2

    def __add__(self, other: 'MeasurementsWithCache') -> 'MeasurementsWithCache':
        return MeasurementsWithCache(**{
            field.name: self._add_entries(getattr(self, field.name), getattr(other, field.name))
            for field in fields(self)
        })

    def __str__(self):
        return (
            "MeasurementsWithCache(\n"
            "  no_cache=\n"
            f"    {self.no_cache},\n"
            "  with_process_cache=\n"
            f"    {self.with_process_cache},\n"
            "  with_persistent_cache=\n"
            f"    {self.with_persistent_cache}\n"
            ")"
        )
