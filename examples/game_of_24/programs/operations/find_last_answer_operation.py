from llm_graph_optimizer.operations.abstract_operation import AbstractOperation
from llm_graph_optimizer.graph_of_operations.types import Dynamic, ReasoningState
from llm_graph_optimizer.measurement.measurement import Measurement
from llm_graph_optimizer.graph_of_operations.graph_partitions import GraphPartitions


class FindLastAnswerOperation(AbstractOperation):
    def __init__(self, params: dict = None, name: str = None):
        input_types = {"score": float}
        output_types = Dynamic
        super().__init__(input_types, output_types, params, name)

    async def _execute(self, partitions: GraphPartitions, input_reasoning_states: ReasoningState) -> tuple[ReasoningState, Measurement | None]:
        previous_score_node = list(partitions.predecessors.direct_predecessors(self, include_dependencies=False))[0]
        previous_llm_evaluate_node = list(partitions.predecessors.direct_predecessors(previous_score_node, include_dependencies=False))[0]
        answer_edge = [edge for edge in partitions.predecessors.predecessor_edges(previous_llm_evaluate_node) if edge.to_node_key == "answer"][0]

        successor_edge_score = [edge for edge in partitions.descendants.successor_edges(self) if edge.to_node_key == "score"][0]
        successor_edge_answer = [edge for edge in partitions.descendants.successor_edges(self) if edge.to_node_key == "answer"][0]

        partitions.move_edge_start_node(current_edge=successor_edge_answer, new_from_node=answer_edge.from_node, new_from_node_key="answer")
        partitions.move_edge_start_node(current_edge=successor_edge_score, new_from_node=previous_score_node, new_from_node_key="score")

        return {}, None
