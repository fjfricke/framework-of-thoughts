from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import logging
from llm_graph_optimizer.measurement.dataset_measurement import DatasetMeasurement, MappedSequentialAndParallelMeasurementsWithCache


class StudyMeasurement:
    def __init__(self, save_file_path: Path = None):
        self.dataset_measurements: list[DatasetMeasurement] = []
        self.best_run_number: int = None
        self.save_file_path: Path = save_file_path

    def add_dataset_measurement(self, dataset_measurement: DatasetMeasurement):
        self.dataset_measurements.append(dataset_measurement)

    def set_best_run(self, best_run: int):
        self.best_run_number = best_run

    def save(self, file_path: str = None):
        if file_path is None:
            file_path = self.save_file_path
            if file_path is None:
                logging.warning("No file path to save persistent cache to. Skipping.")
                return
        with open(file_path, "wb") as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: Path, skip_on_file_not_found: bool = False):
        try:
            with open(path, "rb") as f:
                new_study_measurement = pickle.load(f)
                new_study_measurement.save_file_path = path
                return new_study_measurement
        except FileNotFoundError:
            if skip_on_file_not_found:
                return cls(save_file_path=path)
            else:
                raise
        
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
        
