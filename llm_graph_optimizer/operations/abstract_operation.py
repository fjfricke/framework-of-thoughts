from abc import ABC, abstractmethod

from llm_graph_optimizer.graph_of_operations.graph_of_operations import GraphOfOperations, GraphPartitions
from .helpers.exceptions import OperationFailed
from .helpers.node_state import NodeState

#TODO: Add caching on class level

class AbstractOperation(ABC):
    """
    Abstract base class for all graph operations.
    """

    def __init__(self, input_types: dict[str, type], output_types: dict[str, type], params: dict = None):
        self.params = params
        self.cache = {}
        self.node_state = NodeState.WAITING
        self.input_types = input_types
        self.output_types = output_types
        self.output_reasoning_states = {}

    @abstractmethod
    async def _execute(self, partitions: GraphPartitions, input_reasoning_states: dict[str, any]) -> dict[str, any]:
        pass

    async def execute(self, graph: GraphOfOperations) -> None:

        if not self.node_state == NodeState.PROCESSABLE:
            raise RuntimeError(f"Node {self} is not in the PROCESSABLE state")
        
        input_reasoning_states = graph.get_input_reasoning_states(self)
        
        # Validate input_reasoning_states
        if not isinstance(input_reasoning_states, dict):
            raise TypeError(f"Inputs must be a dictionary, got {type(input_reasoning_states)}")
        
        for key, expected_type in self.input_types.items():
            if key not in input_reasoning_states:
                raise KeyError(f"Missing input key: {key}")
            if not isinstance(input_reasoning_states[key], expected_type):
                raise TypeError(f"Input '{key}' must be of type {expected_type}, got {type(input_reasoning_states[key])}")

        self.node_state = NodeState.PROCESSING
        try:
            result = await self._execute(graph.partitions, input_reasoning_states)

            # Validate result
            if not isinstance(result, dict):
                raise TypeError(f"Outputs must be a dictionary, got {type(result)}")
            
            for key, expected_type in self.output_types.items():
                if key not in result:
                    raise KeyError(f"Missing output key: {key}")
                if not isinstance(result[key], expected_type):
                    raise TypeError(f"Output '{key}' must be of type {expected_type}, got {type(result[key])}")

            self.node_state = NodeState.DONE
            self.output_reasoning_states = result
            graph.update_edge_values(self, result)
        except OperationFailed:
            self.node_state = NodeState.FAILED
