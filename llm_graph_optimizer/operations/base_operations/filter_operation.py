from typing import Callable

from llm_graph_optimizer.graph_of_operations.graph_partitions import GraphPartitions
from llm_graph_optimizer.graph_of_operations.types import ReasoningState, ReasoningStateType
from llm_graph_optimizer.operations.abstract_operation import AbstractOperation
from llm_graph_optimizer.measurement.measurement import Measurements

class FilterOperation(AbstractOperation):
    """
    FilterOperation is a class that filters a list of reasoning states.
    Input_types is a dictionary of an integer to a state. Nodes should connect to 0,1,.. input_keys
    """

    def __init__(self, output_types: ReasoningStateType, length: int, filter_function: Callable[[list[ReasoningState]], ReasoningState], params: dict = None, name: str = None):
        input_types = {i: ReasoningState for i in range(length)}
        self.filter_function = filter_function
        super().__init__(input_types, output_types, params, name)

    async def _execute(self, partitions: GraphPartitions, input_reasoning_states: dict[int, ReasoningState]) -> tuple[ReasoningState, Measurements | None]:
        input_reasoning_states_list = list(input_reasoning_states.values())
        return self.filter_function(input_reasoning_states_list), None  