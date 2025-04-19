from dataclasses import fields
from typing import Dict, get_origin

from llm_graph_optimizer.graph_of_operations.base_graph import BaseGraph

from .measurement import Measurements, SequentialCost
from llm_graph_optimizer.operations.abstract_operation import AbstractOperation


class ProcessMeasurement:

    def __init__(self, graph_of_operations: BaseGraph):
        self.graph_of_operations = graph_of_operations
        self.measurement_for_operation: Dict[AbstractOperation, Measurements] = {}

    def add_measurement(self, operation: AbstractOperation, measurement: Measurements):
        self.measurement_for_operation[operation] = measurement

    def total_sequential_cost(self) -> Measurements:
        total_measurement = Measurements()
        for _, measurement in self.measurement_for_operation.items():
            total_measurement += measurement
        return total_measurement
    
    def total_parallel_cost(self) -> Measurements:
        total_sequential_cost = self.total_sequential_cost()
        total_parallel_cost = Measurements()

        for attr in fields(total_sequential_cost):
            if get_origin(attr.type) == SequentialCost:
                # Calculate the longest path in the graph
                longest_path_cost = self.graph_of_operations.longest_path(
                    weight=lambda from_node: getattr(self.measurement_for_operation[from_node], attr.name, 0) or 0,
                )
                setattr(total_parallel_cost, attr.name, longest_path_cost)
            else:
                setattr(total_parallel_cost, attr.name, getattr(total_sequential_cost, attr.name))

        return total_parallel_cost
    

    def __str__(self):
        return (
            f"ProcessMeasurement(\n"
            f"  total_sequential_cost={self.total_sequential_cost()},\n"
            f"  total_parallel_cost={self.total_parallel_cost()}\n"
            f")"
        )
    
    def __repr__(self):
        return self.__str__()