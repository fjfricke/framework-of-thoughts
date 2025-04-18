from abc import ABC, abstractmethod
import copy
import logging
from typing import Callable, get_origin
from typeguard import TypeCheckError, check_type

from llm_graph_optimizer.graph_of_operations.graph_of_operations import GraphOfOperations, GraphPartitions
from llm_graph_optimizer.graph_of_operations.types import Dynamic, ManyToOne, ReasoningStateExecutionType, ReasoningStateType, StateNotSet
from .helpers.exceptions import OperationFailed
from .helpers.node_state import NodeState

#TODO: Add caching on class level

class AbstractOperation(ABC):
    """
    Abstract base class for all graph operations.
    """

    def __init__(self, input_types: ReasoningStateType, output_types: ReasoningStateType, params: dict = None, name: str = None):
        self.params = params
        self.cache = {}
        self.node_state = NodeState.WAITING
        self.input_types = input_types
        self.output_types = output_types
        self.output_reasoning_states = {}
        self.name = name or self.__class__.__name__

    @abstractmethod
    async def _execute(self, partitions: GraphPartitions, input_reasoning_states: ReasoningStateExecutionType) -> ReasoningStateExecutionType:
        pass

    async def execute(self, graph: GraphOfOperations) -> None:
        
        input_reasoning_states = graph.get_input_reasoning_states(self)
        
        # Validate input_reasoning_states
        if not isinstance(input_reasoning_states, dict):
            raise TypeError(f"Inputs must be a dictionary, got {type(input_reasoning_states)}")
        
        for key, expected_type in self.input_types.items():
            if key not in input_reasoning_states:
                raise KeyError(f"Missing input key: {key}")
            
            if input_reasoning_states[key] is StateNotSet:
                raise ValueError(f"Input reasoning state for key {key} is not set")
            
            try:
                check_type(input_reasoning_states[key], expected_type)
            except TypeCheckError as e:
                if get_origin(expected_type) is ManyToOne and isinstance(input_reasoning_states[key], list):
                    pass
                else:
                    raise TypeError(f"Input '{key}' must be of type {expected_type}, got {type(input_reasoning_states[key])}") from e

            # If the expected_type is a parameterized generic, validate the elements
            # if hasattr(expected_type, "__args__") and isinstance(input_reasoning_states[key], expected_type.__origin__):
            #     element_type = expected_type.__args__[0]
            #     if not all(isinstance(item, element_type) for item in input_reasoning_states[key]):
            #         raise TypeError(f"Elements of input '{key}' must be of type {element_type}")

        partitions = graph.partitions(self)
        result = await self._execute(partitions, input_reasoning_states)

        # Validate result
        if not isinstance(result, dict):
            raise TypeError(f"Outputs must be a dictionary, got {type(result)}")
        
        if self.output_types is Dynamic:
            logging.warning(f"Output types are dynamic for operation {self.name}, skipping validation")
        else:
            for key, expected_type in self.output_types.items():
                if key not in result:
                    raise KeyError(f"Missing output key: {key}")
                
                # Check the base type
                if not isinstance(result[key], expected_type.__origin__ if hasattr(expected_type, "__origin__") else expected_type):
                    raise TypeError(f"Output '{key}' must be of type {expected_type}, got {type(result[key])}")

                # If the expected_type is a parameterized generic, validate the elements
                if hasattr(expected_type, "__args__") and isinstance(result[key], expected_type.__origin__):
                    element_type = expected_type.__args__[0]
                    if not all(isinstance(item, element_type) for item in result[key]):
                        raise TypeError(f"Elements of output '{key}' must be of type {element_type}")

        self.output_reasoning_states = result
        logging.debug(f"Output reasoning states: {self.output_reasoning_states} for operation {self.name}")
        graph.update_edge_values(self, result)

AbstractOperationFactory = Callable[[], AbstractOperation]