import copy
from dataclasses import dataclass, fields, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

SEQUENTIAL_COST_FIELDS = {
    "execution_duration": True,
    "execution_cost": True,
}

@dataclass
class Measurement:
    request_tokens: Optional[int | np.float64] = 0
    response_tokens: Optional[int | np.float64] = 0
    total_tokens: Optional[int | np.float64] = 0
    request_cost: Optional[float | np.float64] = 0
    response_cost: Optional[float | np.float64] = 0
    total_cost: Optional[float | np.float64] = 0
    execution_duration: Optional[float | np.float64] = 0
    execution_cost: Optional[float | np.float64] = 0

    @staticmethod
    def is_sequential_cost(attr_name: str) -> bool:
        return SEQUENTIAL_COST_FIELDS.get(attr_name, False)

    def __post_init__(self):
        for this_field in fields(self):
            field_value = getattr(self, this_field.name)
            if isinstance(field_value, (int, float)):
                setattr(self, this_field.name, np.float64(field_value))

    def __add__(self, other: 'Measurement') -> 'Measurement':
        combined_attributes = {
            field.name: getattr(self, field.name) + getattr(other, field.name)
            for field in fields(self)
        }

        return Measurement(**combined_attributes)
    
    @classmethod
    def map(cls, func: Callable[[np.float64], np.float64], items: list['Measurement']) -> 'Measurement':
        return cls(**{
            field.name: func([np.float64(getattr(item, field.name)) for item in items]) if all([getattr(item, field.name) is not None for item in items]) else None
            for field in fields(cls)
        })
    
    def to_pd(self) -> pd.DataFrame:
        data = {
            field.name: [getattr(self, field.name)] for field in fields(self)
        }

        return pd.DataFrame(data, index=['value'])

@dataclass
class MeasurementsWithCache:
    no_cache: Measurement = field(default_factory=Measurement)
    with_process_cache: Measurement = field(default_factory=Measurement)
    with_persistent_cache: Measurement = field(default_factory=Measurement)

    @classmethod
    def from_no_cache_measurement(cls, measurement: Measurement) -> 'MeasurementsWithCache':
        return cls(
            no_cache=copy.deepcopy(measurement),
            with_process_cache=copy.deepcopy(measurement),
            with_persistent_cache=copy.deepcopy(measurement)
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
    
    @classmethod
    def map(cls, func: Callable[[np.float64], np.float64], items: list['MeasurementsWithCache']) -> 'MeasurementsWithCache':
        return cls(**{
            field.name: Measurement.map(func, [getattr(item, field.name) for item in items])
            for field in fields(cls)
        })
    
    def to_pd(self) -> pd.DataFrame:
        dataframes = []
        for cache_type, measurement in {this_field.name: getattr(self, this_field.name) for this_field in fields(self)}.items():
            if measurement is not None:
                df = measurement.to_pd()
                df.insert(0, 'cache_type', cache_type)
                dataframes.append(df)
        return pd.concat(dataframes, ignore_index=True)

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
