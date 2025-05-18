from typing import get_args, get_origin
from llm_graph_optimizer.graph_of_operations.graph_of_operations import GraphPartitions
from llm_graph_optimizer.graph_of_operations.types import ManyToOne, ReasoningStateType, ReasoningState

from llm_graph_optimizer.measurement.measurement import Measurement
from llm_graph_optimizer.operations.abstract_operation import AbstractOperation
class End(AbstractOperation):
    """
    End operation. Needs to be the last operation in the graph.
    """

    def __init__(self, input_types: ReasoningStateType = None):
        output_types = input_types.copy()

        # replace all ManyToOne[x] with list[x] in the output_types
        for key, value in output_types.items():
            if get_origin(value) == ManyToOne:
                output_types[key] = list[get_args(value)[0]]
        super().__init__(input_types, output_types)

    async def _execute(self, partitions: GraphPartitions, input_reasoning_states: ReasoningState) -> tuple[ReasoningState, Measurement | None]:
        return input_reasoning_states, None
