from typing import Callable, get_origin

from llm_graph_optimizer.graph_of_operations.graph_partitions import GraphPartitions
from llm_graph_optimizer.graph_of_operations.types import ManyToOne, ReasoningState, ReasoningStateType
from llm_graph_optimizer.operations.abstract_operation import AbstractOperation
from llm_graph_optimizer.measurement.measurement import Measurement

class FilterOperation(AbstractOperation):
    """
    FilterOperation is a class that filters a list of reasoning states.
    Input_types is a dictionary of an integer to a state. Nodes should connect to 0,1,.. input_keys
    """

    def __init__(self, output_types: ReasoningStateType, input_types: ReasoningStateType, filter_function: Callable[..., ReasoningState], params: dict = None, name: str = None):
        # INSERT_YOUR_CODE
        if not all(get_origin(value) is ManyToOne for value in input_types.values()):
            raise TypeError("All input types must have keys of origin type ManyToOne")
        self.filter_function = filter_function
        super().__init__(input_types, output_types, params, name)

    async def _execute(self, partitions: GraphPartitions, input_reasoning_states: ReasoningState) -> tuple[ReasoningState, Measurement | None]:
        filtered_reasoning_states = self.filter_function(**input_reasoning_states)
        return filtered_reasoning_states, None