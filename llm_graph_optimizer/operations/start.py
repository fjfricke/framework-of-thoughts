from llm_graph_optimizer.graph_of_operations.graph_of_operations import GraphPartitions
from .helpers.node_state import NodeState
from .abstract_operation import AbstractOperation

class Start(AbstractOperation):
    """
    Start operation.
    """

    def __init__(self, input_types: list[type] = None):
        super().__init__(None, input_types, input_types)
        self.node_state = NodeState.DONE

    async def _execute(self, partitions: GraphPartitions, input_reasoning_states: list[any]) -> list[any]:
        return input_reasoning_states
