from llm_graph_optimizer.graph_of_operations.graph_partitions import GraphPartitions
from llm_graph_optimizer.graph_of_operations.types import Dynamic, Edge, ReasoningState
from llm_graph_optimizer.operations.abstract_operation import AbstractOperation, AbstractOperationFactory
from llm_graph_optimizer.operations.base_operations.filter_operation_with_edge_move import FilterOperationWithEdgeMove


class EvaluateAndChooseOperation(AbstractOperation):
    def __init__(self, value_op: AbstractOperationFactory, score_op: AbstractOperationFactory, is_final_layer: bool = False, num_value_samples: int = 3, params: dict = None, name: str = None):
        input_types = {"expressions": list[str], "remainings": list[list[int]]}
        output_types = Dynamic
        super().__init__(input_types, output_types, params, name)
        self.value_op = value_op
        self.score_op = score_op
        self.if_final_layer = is_final_layer
        self.num_value_samples = num_value_samples
    async def _execute(self, partitions: GraphPartitions, input_reasoning_states: ReasoningState) -> tuple[ReasoningState, None]:
        output_reasoning_states = {}
        score_nodes = []

        for i, (expression, remaining) in enumerate(zip(input_reasoning_states["expressions"], input_reasoning_states["remainings"])):

            score_node = self.score_op()
            partitions.exclusive_descendants.add_node(score_node)
            score_nodes.append(score_node)
            for sample_index in range(self.num_value_samples):
                value_node = self.value_op(cache_seed=sample_index)
                partitions.exclusive_descendants.add_node(value_node)
                partitions.exclusive_descendants.add_edge(Edge(self, value_node, f"remaining_{i}", "input"))
                partitions.exclusive_descendants.add_edge(Edge(value_node, score_node, "value", "values"), order=sample_index)
                output_reasoning_states[f"remaining_{i}"] = remaining
                output_reasoning_states[f"expression_{i}"] = expression
                
        successor_edges = partitions.descendants.successor_edges(self)
        successor_edges = [edge for edge in successor_edges if type(edge.to_node) in [FilterOperationWithEdgeMove]]
        successor_edges = [edge for edge in successor_edges if edge.to_node_key in ["values"]]
        # successor_filter_node = successor_edges[0].to_node
        if len(successor_edges) != 1:
            raise ValueError(f"Only one successor edge is allowed to have a to_node_key of 'values'. Found {len(successor_edges)}.")
        successor_edge = successor_edges[0]
        order = successor_edge.order
        orders = list(range(order, order + len(score_nodes)))
        indices = list(range(len(score_nodes)))
        partitions.move_start_node_and_duplicate_edges(current_edge=successor_edge, new_from_nodes=score_nodes, new_from_node_keys=["score"] * len(score_nodes), orders=orders)
        if self.if_final_layer:
            successor_edges = partitions.descendants.successor_edges(self)
            successor_edges = [edge for edge in successor_edges if type(edge.to_node) in [FilterOperationWithEdgeMove]]
            successor_edges = [edge for edge in successor_edges if edge.to_node_key in ["remainings"]]
            if len(successor_edges) != 1:
                raise ValueError(f"Only one successor edge is allowed to have a to_node_key of 'values'. Found {len(successor_edges)}.")
            successor_edge = successor_edges[0]
            partitions.move_start_node_and_duplicate_edges(current_edge=successor_edge, new_from_nodes=[self] * len(score_nodes), new_from_node_keys=[f"remaining_{i}" for i in indices], orders=orders)

        return output_reasoning_states, None