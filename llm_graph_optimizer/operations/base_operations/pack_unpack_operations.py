from typing import get_args, get_origin
from llm_graph_optimizer.graph_of_operations.graph_partitions import GraphPartitions
from llm_graph_optimizer.graph_of_operations.types import ManyToOne, ReasoningState, ReasoningStateType
from llm_graph_optimizer.operations.abstract_operation import AbstractOperation
from llm_graph_optimizer.measurement.measurement import Measurement

class PackOperation(AbstractOperation):
    def __init__(self, output_types: ReasoningStateType, params: dict = None, name: str = None):
        if not all(get_origin(value) is list for value in output_types.values()):
            raise ValueError("Output types must be lists.")
        input_types = {key: ManyToOne[get_args(value)[0]] for key, value in output_types.items()}
        super().__init__(input_types, output_types, params, name)

    async def _execute(self, partitions: GraphPartitions, input_reasoning_states: ReasoningState) -> tuple[ReasoningState, Measurement | None]:
        return input_reasoning_states, None
