from llm_graph_optimizer.operations.abstract_operation import AbstractOperation
from llm_graph_optimizer.graph_of_operations.types import Dynamic, ReasoningState
from llm_graph_optimizer.measurement.measurement import Measurement
from llm_graph_optimizer.graph_of_operations.graph_partitions import GraphPartitions


class FindLastAnswerOperation(AbstractOperation):
    def __init__(self, params: dict = None, name: str = None):
        """
        This operation is designed to find the last answer and score nodes
        within a graph of operations and attach the corresponding edges to the successor nodes.
        """
        input_types = {"score": float}
        output_types = Dynamic
        super().__init__(input_types, output_types, params, name)

    async def _execute(self, partitions: GraphPartitions, input_reasoning_states: ReasoningState) -> tuple[ReasoningState, Measurement | None]:

        # Identify the previous score node by traversing the graph's ancestors
        previous_score_node = list(partitions.ancestors.direct_predecessors(self, include_dependencies=False))[0]
        
        # Identify the previous LLM evaluation node connected to the score node
        previous_llm_evaluate_node = list(partitions.ancestors.direct_predecessors(previous_score_node, include_dependencies=False))[0]
        
        # Find the edge that leads to the answer node from the LLM evaluation node
        answer_edge = [edge for edge in partitions.ancestors.predecessor_edges(previous_llm_evaluate_node) if edge.to_node_key == "answer"][0]

        # Locate the successor edges for score and answer from the current operation
        successor_edge_score = [edge for edge in partitions.descendants.successor_edges(self) if edge.to_node_key == "score"][0]
        successor_edge_answer = [edge for edge in partitions.descendants.successor_edges(self) if edge.to_node_key == "answer"][0]

        # Update the graph to redirect the answer edge to the correct node
        partitions.move_edge_start_node(current_edge=successor_edge_answer, new_from_node=answer_edge.from_node, new_from_node_key="answer")
        
        # Update the graph to redirect the score edge to the correct node
        partitions.move_edge_start_node(current_edge=successor_edge_score, new_from_node=previous_score_node, new_from_node_key="score")

        return {}, None
