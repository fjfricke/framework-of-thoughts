from llm_graph_optimizer.graph_of_operations.graph_of_operations import GraphPartitions
from llm_graph_optimizer.graph_of_operations.types import ReasoningStateType, ReasoningState
from .abstract_operation import AbstractOperation
from llm_graph_optimizer.measurement.measurement import Measurement
class Start(AbstractOperation):
    """
    Start operation.
    """

    def __init__(self, input_types: ReasoningStateType = None, output_types: ReasoningState = None, static_outputs: ReasoningState = {}):
        if not static_outputs:
            super().__init__(input_types, input_types)
        else:
            super().__init__(input_types, output_types)
        self.input_reasoning_states = None
        self.static_outputs = static_outputs
    def set_input_reasoning_states(self, input_reasoning_states: ReasoningState):
        self.input_reasoning_states = input_reasoning_states

    async def _execute(self, partitions: GraphPartitions, input_reasoning_states: ReasoningState) -> tuple[ReasoningState, Measurement | None]:
        return {**self.static_outputs, **input_reasoning_states}, None
