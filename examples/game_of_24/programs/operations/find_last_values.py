from llm_graph_optimizer.operations.abstract_operation import AbstractOperation
from llm_graph_optimizer.graph_of_operations.types import Dynamic, ReasoningState
from llm_graph_optimizer.measurement.measurement import Measurement
from llm_graph_optimizer.graph_of_operations.graph_partitions import GraphPartitions
from examples.game_of_24.programs.operations.helpers.find_nodes import FindLastValuesType, find_nodes
from llm_graph_optimizer.operations.helpers.exceptions import OperationFailed

class FindLastValuesOperation(AbstractOperation):
    """
    Executes the operation to find the last values in a graph of operations.

    This method identifies the last expression and left nodes based on the operation type
    and updates the graph's edges to point from these nodes to the successor nodes.
    """
    def __init__(self, params: dict = None, name: str = None):
        input_types = {"score": float | None}
        output_types = Dynamic
        super().__init__(input_types, output_types, params, name)

    async def _execute(self, partitions: GraphPartitions, input_reasoning_states: ReasoningState) -> tuple[ReasoningState, Measurement | None]:

        if input_reasoning_states["score"] is None:
            raise OperationFailed("Score is None")

        # Validate the operation type
        if self.params.get("type") not in [FindLastValuesType.ONLY_ONE, FindLastValuesType.ALL]:
            raise ValueError(f"Invalid find last values type: {self.params.get('type')}")

        # Find the last values nodes based on the operation type
        find_last_values_nodes = find_nodes(self, partitions, self.params.get("type"))

        # Get the successor edges and filter them
        successor_edges = partitions.exclusive_descendants.successor_edges(self)
        successor_edges_expressions = [edge for edge in successor_edges if edge.from_node_key == "expression"]
        assert len(successor_edges_expressions) in [0, 1]

        # Update the graph to redirect the expression edge if it exists
        if len(successor_edges_expressions) == 1:
            partitions.move_edge_start_node(
                current_edge=successor_edges_expressions[0],
                new_from_node=find_last_values_nodes.expression_nodes[0],
                new_from_node_key=find_last_values_nodes.expression_nodekeys[0]
            )

        # Filter successor edges by key for "left"
        successor_edges_lefts = [edge for edge in successor_edges if edge.from_node_key == "left"]
        assert len(successor_edges_lefts) in [0, 1]

        # Update the graph to redirect the left edge if it exists
        if len(successor_edges_lefts) == 1:
            partitions.move_edge_start_node(
                current_edge=successor_edges_lefts[0],
                new_from_node=find_last_values_nodes.left_nodes[0],
                new_from_node_key=find_last_values_nodes.left_nodekeys[0]
            )

        return {}, None