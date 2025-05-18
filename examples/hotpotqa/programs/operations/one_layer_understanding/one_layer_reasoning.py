import numpy as np
from examples.hotpotqa.programs.operations.one_layer_understanding.branch import ShouldBranchClassifier
from examples.hotpotqa.programs.operations.one_layer_understanding.one_layer_understanding import OneLayerUnderstanding
from llm_graph_optimizer.graph_of_operations.graph_partitions import GraphPartitions
from llm_graph_optimizer.graph_of_operations.types import Dynamic, Edge, ManyToOne, ReasoningState, StateNotSet
from llm_graph_optimizer.measurement.measurement import Measurement
from llm_graph_optimizer.operations.abstract_operation import AbstractOperation, AbstractOperationFactory
from llm_graph_optimizer.operations.base_operations.end import End
from llm_graph_optimizer.operations.base_operations.pack_unpack_operations import PackOperation


class OneLayerReasoning(AbstractOperation):
    def __init__(self, open_book_op: AbstractOperationFactory, closed_book_op: AbstractOperationFactory, child_aggregate_op: AbstractOperationFactory, filter_op: AbstractOperationFactory, decompose_op: AbstractOperationFactory, understanding_op: AbstractOperationFactory, min_branch_certainty_threshold: float, params: dict = None, name: str = None):
        input_types = {"question": str, "dependency_answers": ManyToOne[str], "question_decomposition_score": float, "should_decompose": bool, "should_decompose_score": float, "max_depth": int}
        output_types = Dynamic
        super().__init__(input_types, output_types, params, name)
        self.open_book_op = open_book_op
        self.closed_book_op = closed_book_op
        self.child_aggregate_op = child_aggregate_op
        self.filter_op = filter_op
        self.decompose_op = decompose_op
        self.understanding_op = understanding_op
        self.min_branch_certainty_threshold = min_branch_certainty_threshold

    async def _execute(self, partitions: GraphPartitions, input_reasoning_states: ReasoningState) -> tuple[ReasoningState, Measurement]:
        question = input_reasoning_states["question"]
        dependency_answers = input_reasoning_states["dependency_answers"]
        max_depth = input_reasoning_states["max_depth"]
        should_decompose = input_reasoning_states["should_decompose"]
        should_decompose_score = input_reasoning_states["should_decompose_score"]

        should_decompose = max_depth > 0 and (should_decompose and np.exp(should_decompose_score) > self.min_branch_certainty_threshold or not should_decompose and np.exp(should_decompose_score) < self.min_branch_certainty_threshold)

        filter_node = self.filter_op()
        partitions.exclusive_descendants.add_node(filter_node)

        open_book_node = self.open_book_op()
        partitions.exclusive_descendants.add_node(open_book_node)
        partitions.exclusive_descendants.add_edge(Edge(self, open_book_node, "question", "question"))
        partitions.exclusive_descendants.add_edge(Edge(self, open_book_node, "dependency_answers", "dependency_answers"))

        partitions.exclusive_descendants.add_edge(Edge(open_book_node, filter_node, "answer", "answers"), order=0)
        partitions.exclusive_descendants.add_edge(Edge(open_book_node, filter_node, "decomposition_score", "decomposition_scores"), order=0)

        closed_book_node = self.closed_book_op()
        partitions.exclusive_descendants.add_node(closed_book_node)
        partitions.exclusive_descendants.add_edge(Edge(self, closed_book_node, "question", "question"))
        partitions.exclusive_descendants.add_edge(Edge(self, closed_book_node, "dependency_answers", "dependency_answers"))

        partitions.exclusive_descendants.add_edge(Edge(closed_book_node, filter_node, "answer", "answers"), order=1)
        partitions.exclusive_descendants.add_edge(Edge(closed_book_node, filter_node, "decomposition_score", "decomposition_scores"), order=1)

        if should_decompose:
            decompose_node = self.decompose_op()
            partitions.exclusive_descendants.add_node(decompose_node)
            partitions.exclusive_descendants.add_edge(Edge(self, decompose_node, "question", "question"))
            partitions.exclusive_descendants.add_edge(Edge(self, decompose_node, "dependency_answers", "dependency_answers"))

            understanding_node = self.understanding_op()
            partitions.exclusive_descendants.add_node(understanding_node)
            partitions.exclusive_descendants.add_edge(Edge(decompose_node, understanding_node, "subquestions", "subquestions"))
            partitions.exclusive_descendants.add_edge(Edge(decompose_node, understanding_node, "question_decomposition_score", "question_decomposition_score"))
            partitions.exclusive_descendants.add_edge(Edge(self, understanding_node, "max_depth", "max_depth"))

            child_aggregate_node = self.child_aggregate_op()
            partitions.exclusive_descendants.add_node(child_aggregate_node)
            partitions.exclusive_descendants.add_edge(Edge(self, child_aggregate_node, "question", "question"))
            partitions.exclusive_descendants.add_edge(Edge(self, child_aggregate_node, "dependency_answers", "dependency_answers"))

            partitions.exclusive_descendants.add_edge(Edge(decompose_node, child_aggregate_node, "subquestions", "subquestions"))
            partitions.exclusive_descendants.add_edge(Edge(understanding_node, child_aggregate_node, "subquestion_answers", "subquestion_answers"))
            partitions.exclusive_descendants.add_edge(Edge(understanding_node, child_aggregate_node, "child_decomposition_scores", "child_decomposition_scores"))
            partitions.exclusive_descendants.add_edge(Edge(decompose_node, child_aggregate_node, "question_decomposition_score", "question_decomposition_score"))

            partitions.exclusive_descendants.add_edge(Edge(child_aggregate_node, filter_node, "answer", "answers"), order=2)
            partitions.exclusive_descendants.add_edge(Edge(child_aggregate_node, filter_node, "decomposition_score", "decomposition_scores"), order=2)

        # move successor edges to UnderstandingGraphUpdating nodes to filter node iff the key is "answer" or "decomposition_score"
        successor_edges = partitions.descendants.successor_edges(self)
        successor_edges = [edge for edge in successor_edges if type(edge.to_node) in [End, OneLayerUnderstanding, PackOperation, OneLayerReasoning, ShouldBranchClassifier]]
        successor_edges_with_answer = [edge for edge in successor_edges if edge.to_node_key in ["answer", "subquestion_answers", "dependency_answers"]]
        for successor_edge in successor_edges_with_answer:
            partitions.move_edge_start_node(current_edge=successor_edge, new_from_node=filter_node, new_from_node_key="answer")
        successor_edges_with_decomposition_score = [edge for edge in successor_edges if edge.to_node_key in ["decomposition_score", "child_decomposition_scores"]]
        for successor_edge in successor_edges_with_decomposition_score:
            partitions.move_edge_start_node(current_edge=successor_edge, new_from_node=filter_node, new_from_node_key="decomposition_score")
        

        return {"answer": StateNotSet, "decomposition_score": StateNotSet, "question": question, "dependency_answers": dependency_answers, "max_depth": max_depth}, None