from llm_graph_optimizer.graph_of_operations.graph_of_operations import GraphPartitions
from llm_graph_optimizer.graph_of_operations.types import ReasoningStateType, ReasoningState
from .abstract_operation import AbstractOperation

class End(AbstractOperation):
    """
    End operation.
    """

    def __init__(self, input_types: ReasoningStateType = None):
        super().__init__(input_types, input_types)

    async def _execute(self, partitions: GraphPartitions, input_reasoning_states: ReasoningState) -> ReasoningState:
        return input_reasoning_states
