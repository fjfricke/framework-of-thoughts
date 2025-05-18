from examples.game_of_24.programs.operations.evaluate_and_choose_operation import EvaluateAndChooseOperation
from examples.game_of_24.programs.operations.value_operation import ValueOperation
from llm_graph_optimizer.graph_of_operations.graph_partitions import GraphPartitions
from llm_graph_optimizer.graph_of_operations.types import Dynamic, Edge, ReasoningState
from llm_graph_optimizer.measurement.measurement import Measurement
from llm_graph_optimizer.operations.abstract_operation import AbstractOperation
from llm_graph_optimizer.operations.helpers.exceptions import OperationFailed


def get_id_from_key(key: str) -> int:
    return int(key.split("_")[-1])


class GetAllPredecessorRemainingsAndExpressions(AbstractOperation):
    """
    Get the remaining from the predecessor of the current node.
    input_types: {"score": float}
    output_types: {}  # helper
    """
    def __init__(self, params: dict = None, name: str = None):
        input_types = {"score": float | None}
        output_types = Dynamic
        super().__init__(input_types, output_types, params, name)

    async def _execute(self, partitions: GraphPartitions, input_reasoning_states: ReasoningState) -> tuple[ReasoningState, Measurement | None]:

        current_node = self
        predecessors_to_add_edges_from = []
        while True:
            predecessors = partitions.predecessors.direct_predecessors(current_node, include_dependencies=False)
            if len(predecessors) == 0:
                break
            if type(current_node) is ValueOperation and type(predecessors[0]) is EvaluateAndChooseOperation:
                edge_data = partitions.predecessors.get_all_edge_data_between(predecessors[0], current_node)
                if len(edge_data) == 0 or edge_data is None:
                    raise OperationFailed("No edge found between ValueOperation and EvaluateAndChooseOperation")
                from_node_key_remaining = list(edge_data.values())[0]["from_node_key"]
                from_node_key_expression = f"expression_{get_id_from_key(from_node_key_remaining)}"

                predecessors_to_add_edges_from.append((predecessors[0], from_node_key_remaining, from_node_key_expression))
            current_node = predecessors[0]

        next_node = next(partitions.exclusive_descendants.successors(self))
        orders = list(range(len(predecessors_to_add_edges_from)))
        from_nodes, from_node_keys_remainings, from_node_keys_expressions = zip(*predecessors_to_add_edges_from)

        partitions.move_start_node_and_duplicate_edges(Edge(self, next_node, "remaining", "remainings"), from_nodes, from_node_keys_remainings, orders=orders)
        partitions.move_start_node_and_duplicate_edges(Edge(self, next_node, "expression", "expressions"), from_nodes, from_node_keys_expressions, orders=orders)

        partitions.exclusive_descendants.add_dependency_edge(self, next_node)

        if input_reasoning_states["score"] is None:
            raise OperationFailed("Score is not set. No output has been 24.")
        return {}, None
