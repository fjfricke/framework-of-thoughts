from typing import Callable

from llm_graph_optimizer.graph_of_operations.graph_partitions import GraphPartitions
from llm_graph_optimizer.graph_of_operations.types import ReasoningStateType, ReasoningState
from llm_graph_optimizer.operations.abstract_operation import AbstractOperation
from llm_graph_optimizer.measurement.measurement import Measurement

class BranchOperation(AbstractOperation):
    """
    Branch operation. Conditionally executes a function based on a decision function.
    """
    
    def __init__(self, input_types: ReasoningStateType, output_types: ReasoningStateType, decision_function: Callable[["BranchOperation", GraphPartitions, ReasoningState], tuple[ReasoningState, Measurement | None]], params: dict = None, name: str = None):
        input_types = input_types
        output_types = output_types
        super().__init__(input_types, output_types, params, name)
        self.decision_function = decision_function

    async def _execute(self, partitions: GraphPartitions, input_reasoning_states: ReasoningState) -> tuple[ReasoningState, Measurement | None]:
        return self.decision_function(self, partitions, input_reasoning_states)