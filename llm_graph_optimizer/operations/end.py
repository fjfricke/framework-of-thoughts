from llm_graph_optimizer.graph_of_operations.graph_of_operations import GraphPartitions
from .abstract_operation import AbstractOperation

class End(AbstractOperation):
    """
    End operation.
    """

    def __init__(self, input_types: dict[str, type] = None):
        super().__init__(input_types, input_types)

    async def _execute(self, partitions: GraphPartitions, input_reasoning_states: dict[str, any]) -> dict[str, any]:
        return input_reasoning_states
