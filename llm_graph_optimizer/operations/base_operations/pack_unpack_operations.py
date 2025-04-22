from llm_graph_optimizer.graph_of_operations.graph_partitions import GraphPartitions
from llm_graph_optimizer.graph_of_operations.types import ManyToOne, ReasoningState, ReasoningStateType
from llm_graph_optimizer.operations.abstract_operation import AbstractOperation
from llm_graph_optimizer.measurement.measurement import Measurement

class PackOperation(AbstractOperation):
    def __init__(self, output_types: ReasoningStateType, params: dict = None, name: str = None):
        input_types = {key: ManyToOne[value] for key, value in output_types.items()}
        super().__init__(input_types, output_types, params, name)

    async def _execute(self, partitions: GraphPartitions, input_reasoning_states: ReasoningState) -> tuple[ReasoningState, Measurement | None]:
        return input_reasoning_states, None
