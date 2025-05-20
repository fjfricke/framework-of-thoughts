from dataclasses import fields
from typing import TYPE_CHECKING, Dict
import uuid

from llm_graph_optimizer.graph_of_operations.base_graph import BaseGraph

from .measurement import Measurement, MeasurementsWithCache

if TYPE_CHECKING:
    from llm_graph_optimizer.operations.abstract_operation import AbstractOperation


class ProcessMeasurement:

    def __init__(self, graph_of_operations: BaseGraph):
        self.snapshot_graph = graph_of_operations.snapshot
        self.measurement_for_operation: Dict[uuid.UUID, MeasurementsWithCache] = {}

    def add_measurement(self, operation: "AbstractOperation", measurement_or_measurements_with_cache: Measurement | MeasurementsWithCache):
        if isinstance(measurement_or_measurements_with_cache, Measurement):
            measurements_with_cache = MeasurementsWithCache.from_no_cache_measurement(measurement_or_measurements_with_cache)
        else:
            measurements_with_cache = measurement_or_measurements_with_cache
        self.measurement_for_operation[operation.uuid] = measurements_with_cache

    def total_sequential_cost(self) -> MeasurementsWithCache:
        total_measurements = MeasurementsWithCache()
        for measurements in self.measurement_for_operation.values():
            total_measurements += measurements
        return total_measurements
    
    def total_parallel_cost(self) -> MeasurementsWithCache:
        total_sequential_measurements = self.total_sequential_cost()
        total_parallel_measurements = MeasurementsWithCache()

        # Iterate over the fields of MeasurementsWithCache
        for measurement_field in fields(total_sequential_measurements):
            measurement = getattr(total_sequential_measurements, measurement_field.name)
            parallel_measurement = Measurement()

            for attr in fields(measurement):
                attr_value = getattr(measurement, attr.name)
                if measurement.is_sequential_cost(attr.name):
                    # Calculate the longest path in the graph
                    longest_path_cost = self.snapshot_graph.longest_path(
                        weight=lambda from_node: getattr(getattr(self.measurement_for_operation[from_node], measurement_field.name), attr.name, 0) or 0,
                    )
                    setattr(parallel_measurement, attr.name, longest_path_cost)
                else:
                    setattr(parallel_measurement, attr.name, attr_value)

            # Update the corresponding measurement in total_parallel_measurements
            setattr(total_parallel_measurements, measurement_field.name, parallel_measurement)

        return total_parallel_measurements

    def __str__(self):
        return (
            "ProcessMeasurement(\n"
            "  total_sequential_cost=\n"
            f"    {self.total_sequential_cost()},\n"
            "  total_parallel_cost=\n"
            f"    {self.total_parallel_cost()}\n"
            ")"
        )
