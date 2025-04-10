from abc import ABC, abstractmethod
import copy
import logging

from llm_graph_optimizer.graph_of_operations.graph_of_operations import GraphOfOperations, GraphPartitions
from .helpers.exceptions import OperationFailed
from .helpers.node_state import NodeState

#TODO: Add caching on class level

class AbstractOperation(ABC):
    """
    Abstract base class for all graph operations.
    """

    def __init__(self, input_types: dict[str | int, type], output_types: dict[str | int, type], params: dict = None, name: str = None):
        self.params = params
        self.cache = {}
        self.node_state = NodeState.WAITING
        self.input_types = input_types
        self.output_types = output_types
        self.output_reasoning_states = {}
        self.name = name or self.__class__.__name__

    def copy(self) -> "AbstractOperation":
        new_op = copy.copy(self)  # Create a shallow copy
        new_op.node_state = NodeState.WAITING
        new_op.output_reasoning_states = {}
        return new_op

    @abstractmethod
    async def _execute(self, partitions: GraphPartitions, input_reasoning_states: dict[str | int, any]) -> dict[str | int, any]:
        pass

    async def execute(self, graph: GraphOfOperations) -> None:
        
        input_reasoning_states = graph.get_input_reasoning_states(self)
        
        # Validate input_reasoning_states
        if not isinstance(input_reasoning_states, dict):
            raise TypeError(f"Inputs must be a dictionary, got {type(input_reasoning_states)}")
        
        for key, expected_type in self.input_types.items():
            if key not in input_reasoning_states:
                raise KeyError(f"Missing input key: {key}")
            
            # Check the base type
            if not isinstance(input_reasoning_states[key], expected_type.__origin__ if hasattr(expected_type, "__origin__") else expected_type):
                raise TypeError(f"Input '{key}' must be of type {expected_type}, got {type(input_reasoning_states[key])}")

            # If the expected_type is a parameterized generic, validate the elements
            if hasattr(expected_type, "__args__") and isinstance(input_reasoning_states[key], expected_type.__origin__):
                element_type = expected_type.__args__[0]
                if not all(isinstance(item, element_type) for item in input_reasoning_states[key]):
                    raise TypeError(f"Elements of input '{key}' must be of type {element_type}")

        result = await self._execute(graph.partitions, input_reasoning_states)

        # Validate result
        if not isinstance(result, dict):
            raise TypeError(f"Outputs must be a dictionary, got {type(result)}")
        
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
