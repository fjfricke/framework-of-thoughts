from llm_graph_optimizer.graph_of_operations.graph_partitions import GraphPartitions
from llm_graph_optimizer.graph_of_operations.types import ReasoningStateType, ReasoningState
from llm_graph_optimizer.measurement.measurement import Measurements
from llm_graph_optimizer.operations.abstract_operation import AbstractOperation


class TestOperation(AbstractOperation):
    def __init__(self, input_types: ReasoningStateType, output_types: ReasoningStateType, params: dict = None, name: str = None):
        super().__init__(input_types, output_types, params, name)

    async def _execute(self, partitions: GraphPartitions, input_reasoning_states: ReasoningState) -> tuple[ReasoningState, Measurements | None]:
        output_reasoning_states = input_reasoning_states
        for key, _ in self.output_types.items():
            output_reasoning_states[key] = None
        return output_reasoning_states, None
