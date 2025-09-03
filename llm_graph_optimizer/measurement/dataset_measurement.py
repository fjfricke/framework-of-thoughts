from dataclasses import dataclass, fields
from pathlib import Path
import pickle
from sys import maxsize
from typing import Callable

import numpy as np
import pandas as pd

from llm_graph_optimizer.measurement.measurement import MeasurementsWithCache
from llm_graph_optimizer.measurement.process_measurement import ProcessMeasurement

Map = Callable[[list[float]], float]

@dataclass
class MappedSequentialAndParallelMeasurementsWithCache:
    sequential: MeasurementsWithCache
    parallel: MeasurementsWithCache

    @classmethod
    def map(cls, func: Callable[[np.float64], np.float64], items: list['MappedSequentialAndParallelMeasurementsWithCache']) -> 'MappedSequentialAndParallelMeasurementsWithCache':
        return cls(
            sequential=MeasurementsWithCache.map(func, [item.sequential for item in items]),
            parallel=MeasurementsWithCache.map(func, [item.parallel for item in items])
        )
    
    def to_pd(self) -> pd.DataFrame:
        dataframes = []
        for cache_type, measurement in {field.name: getattr(self, field.name) for field in fields(self)}.items():
            if measurement is not None:
                df = measurement.to_pd()
                df.insert(0, 'execution_type', cache_type)
                dataframes.append(df)
        return pd.concat(dataframes, ignore_index=True)
    
@dataclass
class GlobalEvaluationMeasurements:
    total_execution_duration: np.float64

    def to_pd_row(self) -> pd.DataFrame:
        return pd.DataFrame({field.name: [getattr(self, field.name)] for field in fields(self)})

@dataclass
class ScoreParameter:
    name: str
    map: Map = np.mean
    confidence_interval_width: float = 0.95
    acceptable_ci_width: float = None

    def __hash__(self):
        return hash(self.name)
    
    def to_pd_row(self) -> pd.DataFrame:
        return pd.DataFrame({field.name: [getattr(self, field.name)] for field in fields(self)})

@dataclass
class Score:
    name: str
    value: np.float64
    confidence_interval_width: np.float64

    def to_pd_row(self) -> pd.DataFrame:
        return pd.DataFrame({field.name: [getattr(self, field.name)] for field in fields(self)})

@dataclass
class DatasetEvaluatorParameters:
    score_parameters: list[ScoreParameter]
    min_runs: int = 1
    max_runs: int = maxsize

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path):
        with open(path, "rb") as f:
            return pickle.load(f)
        
    def to_pd(self) -> pd.DataFrame:
        # Dynamically extract fields from ScoreParameter
        score_param_fields = [field.name for field in fields(ScoreParameter)]

        # Create DataFrame for score_parameters
        score_parameters_dataframes = []
        for score_param in self.score_parameters:
            if hasattr(score_param, 'to_pd_row'):
                df = score_param.to_pd_row()
                score_parameters_dataframes.append(df)
            else:
                # Dynamically handle missing fields
                data = {field: [getattr(score_param, field, None)] for field in score_param_fields}
                score_parameters_dataframes.append(pd.DataFrame(data))
        score_parameters_df = pd.concat(score_parameters_dataframes, ignore_index=True)

        # Add a header for score_parameters
        score_parameters_df.insert(0, 'section', 'score_parameters')

        # Dynamically extract fields for global parameters
        global_param_fields = [field.name for field in fields(DatasetEvaluatorParameters) if field.name != 'score_parameters']
        global_parameters = {
            'parameter': global_param_fields,
            'value': [getattr(self, field, None) for field in global_param_fields]
        }
        global_parameters_df = pd.DataFrame(global_parameters)
        global_parameters_df.insert(0, 'section', 'global_parameters')

        # Combine the two sections with an empty row in between
        empty_row = pd.DataFrame([['', '', '', '', '']], columns=score_parameters_df.columns)
        combined_df = pd.concat([score_parameters_df, empty_row, global_parameters_df], ignore_index=True)

        return combined_df

class DatasetMeasurement:
    def __init__(self):
        self._measurements_for_index: dict[int, MappedSequentialAndParallelMeasurementsWithCache] = {}
        self.global_evaluation_measurements = None
        self.dataset_evaluator_parameters: DatasetEvaluatorParameters = None
        self.scores: list[Score] = []

    def add_measurement(self, index: int, process_measurement: ProcessMeasurement):
        self._measurements_for_index[index] = MappedSequentialAndParallelMeasurementsWithCache(
            process_measurement.total_sequential_cost(), process_measurement.total_parallel_cost()
        )

    @property
    def dataset_measurements(self):
        max_index = max(self._measurements_for_index.keys(), default=-1)
        dataset_measurement_list = [
            self._measurements_for_index.get(i, None) for i in range(max_index + 1)
        ]
        if any(measurement is None for measurement in dataset_measurement_list):
            raise ValueError("Some measurements are None")
        return dataset_measurement_list
    
    def add_global_evaluation_measurement(self, global_evaluation_measurement: GlobalEvaluationMeasurements):
        self.global_evaluation_measurements = global_evaluation_measurement

    def add_dataset_evaluator_parameters(self, dataset_evaluator_parameters: DatasetEvaluatorParameters):
        self.dataset_evaluator_parameters = dataset_evaluator_parameters

    def add_scores(self, scores: list[Score]):
        self.scores = scores

    def calculate_dataset_measurement(self, map: Callable[[list[np.float64]], np.float64]):
        return MappedSequentialAndParallelMeasurementsWithCache.map(map, self.dataset_measurements)
    
    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path):
        with open(path, "rb") as f:
            return pickle.load(f)
    
    def to_excel(self, path: Path, maps_for_measurements: dict[str, Callable[[np.float64], np.float64]]):
        sheets_data = {}
        if self.global_evaluation_measurements:
            df = pd.DataFrame()
            df = pd.concat([df, self.global_evaluation_measurements.to_pd_row()], axis=1)
            sheets_data["global_evaluation_measurements"] = df
        if self.dataset_evaluator_parameters:
            df = pd.DataFrame()
            df = pd.concat([df, self.dataset_evaluator_parameters.to_pd()], axis=1)
            sheets_data["dataset_evaluator_parameters"] = df
        if self.scores:
            df = pd.DataFrame()
            for score in self.scores:
                df = pd.concat([df, score.to_pd_row()], axis=0)
            sheets_data["scores"] = df
        for map_name, map_function in maps_for_measurements.items():
            mapped_sequential_and_parallel_measurements = self.calculate_dataset_measurement(map_function)
            df = mapped_sequential_and_parallel_measurements.to_pd()
            sheets_data[map_name] = df
        with pd.ExcelWriter(path) as writer:
            for sheet_name, data in sheets_data.items():
                data.to_excel(writer, sheet_name=sheet_name, index=False)