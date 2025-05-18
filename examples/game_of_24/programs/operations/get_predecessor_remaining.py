from examples.game_of_24.programs.operations.evaluate_and_choose_operation import EvaluateAndChooseOperation
from examples.game_of_24.programs.operations.value_operation import ValueOperation
from llm_graph_optimizer.graph_of_operations.graph_partitions import GraphPartitions
from llm_graph_optimizer.graph_of_operations.types import Dynamic, Edge, ReasoningState
from llm_graph_optimizer.measurement.measurement import Measurement
from llm_graph_optimizer.operations.abstract_operation import AbstractOperation
from llm_graph_optimizer.operations.helpers.exceptions import OperationFailed


class GetPredecessorRemaining(AbstractOperation):
    """
    Get the remaining from the predecessor of the current node.
    input_types: {"score": float}
    output_types: {"remaining": list[list[int]]}
    """
    def __init__(self, params: dict = None, name: str = None):
        input_types = {"score": float}
        output_types = Dynamic
        super().__init__(input_types, output_types, params, name)

    async def _execute(self, partitions: GraphPartitions, input_reasoning_states: ReasoningState) -> tuple[ReasoningState, Measurement | None]:

        current_node = self
        while True:
            predecessors = partitions.predecessors.direct_predecessors(current_node, include_dependencies=False)
            if len(predecessors) == 0:
                raise OperationFailed("No predecessors found. Sth. is wrong with the graph.")
            if type(current_node) is ValueOperation and type(predecessors[0]) is EvaluateAndChooseOperation:
                edge_data = partitions.predecessors.get_all_edge_data_between(predecessors[0], current_node)
                if len(edge_data) == 0 or edge_data is None:
                    raise OperationFailed("No edge found between ValueOperation and EvaluateAndChooseOperation")
                from_node_key = list(edge_data.values())[0]["from_node_key"]

                next_node = next(partitions.exclusive_descendants.successors(self))
                partitions.move_edge_start_node(Edge(self, next_node, "remaining", "input"), predecessors[0], from_node_key)
                partitions.exclusive_descendants.add_dependency_edge(self, next_node)

                return {}, None
            current_node = predecessors[0]
