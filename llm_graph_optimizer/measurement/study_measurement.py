from pathlib import Path
import numpy as np
import pandas as pd
from llm_graph_optimizer.measurement.dataset_measurement import DatasetMeasurement, MappedSequentialAndParallelMeasurementsWithCache


class StudyMeasurement:
    def __init__(self):
        self.dataset_measurements: list[DatasetMeasurement] = []
        self.best_run_number: int = None

    def add_dataset_measurement(self, dataset_measurement: DatasetMeasurement):
        self.dataset_measurements.append(dataset_measurement)

    def set_best_run(self, best_run: int):
        self.best_run_number = best_run

    @property
    def best_run(self) -> DatasetMeasurement:
        if self.best_run_number is None:
            raise ValueError("Best run number is not set")
        return self.dataset_measurements[self.best_run_number]

    @property
    def mean_total_measurements(self):
        mean_dataset_measurements = [dataset_measurement.calculate_dataset_measurement(np.mean) for dataset_measurement in self.dataset_measurements]
        return MappedSequentialAndParallelMeasurementsWithCache.map(np.mean, mean_dataset_measurements)

    @property
    def sum_total_measurements(self):
        sum_dataset_measurements = [dataset_measurement.calculate_dataset_measurement(np.sum) for dataset_measurement in self.dataset_measurements]
        return MappedSequentialAndParallelMeasurementsWithCache.map(np.sum, sum_dataset_measurements)
    
    def to_excel(self, path: Path):
        sheets_data = {}
        df = pd.DataFrame()
        df = pd.concat([df, self.mean_total_measurements.to_pd()], axis=1)
        sheets_data["mean_total_measurements"] = df
        df = pd.DataFrame()
        df = pd.concat([df, self.sum_total_measurements.to_pd()], axis=1)
        sheets_data["sum_total_measurements"] = df
        with pd.ExcelWriter(path) as writer:
            for sheet_name, data in sheets_data.items():
                data.to_excel(writer, sheet_name=sheet_name, index=False)
        
