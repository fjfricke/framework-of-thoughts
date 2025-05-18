import logging

from examples.hotpotqa.programs.operations.reasoning.child_aggregate import ChildAggregateReasoning
from examples.hotpotqa.programs.operations.utils import find_dependencies
from llm_graph_optimizer.graph_of_operations.graph_partitions import GraphPartitions
from llm_graph_optimizer.graph_of_operations.types import Dynamic, Edge, ManyToOne, ReasoningState, StateNotSet
from llm_graph_optimizer.measurement.measurement import Measurement
from llm_graph_optimizer.operations.abstract_operation import AbstractOperation, AbstractOperationFactory
from llm_graph_optimizer.operations.base_operations.end import End


class UnderstandingGraphUpdating(AbstractOperation):
    def __init__(self, open_book_op: AbstractOperationFactory, closed_book_op: AbstractOperationFactory, child_aggregate_op: AbstractOperationFactory, understanding_op: AbstractOperationFactory, filter_op: AbstractOperationFactory, params: dict = None, name: str = None):
        input_types = {"hqdt": dict, "question": str, "question_decomposition_score": float, "dependency_answers": ManyToOne[str]}
        output_types = Dynamic
        super().__init__(input_types, output_types, params, name)
        self.open_book_op = open_book_op
        self.closed_book_op = closed_book_op
        self.child_aggregate_op = child_aggregate_op
        self.understanding_op = understanding_op
        self.filter_op = filter_op
    async def _execute(self, partitions: GraphPartitions, input_reasoning_states: ReasoningState) -> tuple[ReasoningState, Measurement]:
        

        # get state data
        hqdt = input_reasoning_states["hqdt"]
        current_question = input_reasoning_states["question"]
        if current_question not in hqdt:
            subquestions = []
        else:
            subquestions = hqdt[current_question][0]

        dependency_answers = list(input_reasoning_states["dependency_answers"])
        output_reasoning_states = {"hqdt": hqdt, "question": current_question, "answer": StateNotSet, "question_decomposition_score": StateNotSet, "dependency_answers": dependency_answers}

        filter_node = self.filter_op()
        partitions.exclusive_descendants.add_node(filter_node)

        # create reasoning nodes
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
        if subquestions:
            output_reasoning_states["decomposition_score"] = hqdt[current_question][1]
            child_aggregate_node = self.child_aggregate_op()
            partitions.exclusive_descendants.add_node(child_aggregate_node)
            partitions.exclusive_descendants.add_edge(Edge(self, child_aggregate_node, "question", "question"))
            partitions.exclusive_descendants.add_edge(Edge(self, child_aggregate_node, "question_decomposition_score", "question_decomposition_score"))
            partitions.exclusive_descendants.add_edge(Edge(self, child_aggregate_node, "dependency_answers", "dependency_answers"))

            partitions.exclusive_descendants.add_edge(Edge(child_aggregate_node, filter_node, "answer", "answers"), order=2)
            partitions.exclusive_descendants.add_edge(Edge(child_aggregate_node, filter_node, "decomposition_score", "decomposition_scores"), order=2)

        # move successor edges to UnderstandingGraphUpdating nodes to filter node iff the key is "answer" or "decomposition_score"
        successor_edges = partitions.descendants.successor_edges(self)
        successor_edges = [edge for edge in successor_edges if type(edge.to_node) in [End, UnderstandingGraphUpdating]]
        successor_edges = [edge for edge in successor_edges if edge.to_node_key in ["answer", "decomposition_score"]]
        for successor_edge in successor_edges:
            partitions.move_edge_start_node(current_edge=successor_edge, new_from_node=filter_node, new_from_node_key=successor_edge.to_node_key)

        successor_edges = partitions.descendants.successor_edges(self)
        successor_edges = [edge for edge in successor_edges if type(edge.to_node) is UnderstandingGraphUpdating]
        successor_edges = [edge for edge in successor_edges if edge.to_node_key == "dependency_answers"]
        for successor_edge in successor_edges:
            partitions.move_edge_start_node(current_edge=successor_edge, new_from_node=filter_node, new_from_node_key="answer")

        # move successor edges going to previous aggregate to start at filter node
        to_key_to_from_key = {
            "subquestion_answers": "answer",
            "child_decomposition_scores": "decomposition_score"
        }
        successor_edges = partitions.descendants.successor_edges(self)
        successor_edges = [edge for edge in successor_edges if type(edge.to_node) is ChildAggregateReasoning]
        successor_edges = [edge for edge in successor_edges if edge.to_node_key in ["subquestion_answers", "child_decomposition_scores"]]
        for successor_edge in successor_edges:
            partitions.move_edge_start_node(current_edge=successor_edge, new_from_node=filter_node, new_from_node_key=to_key_to_from_key[successor_edge.to_node_key])

        # create understanding nodes for subquestions
        if subquestions:
            question_decomposition_score = [hqdt[current_question][1]]
            output_reasoning_states["question_decomposition_score"] = question_decomposition_score[0]
            understanding_nodes = []
            for i, subquestion in enumerate(hqdt[current_question][0]):
                understanding_node = self.understanding_op()
                potential_dependencies = find_dependencies(subquestion)
                partitions.exclusive_descendants.add_node(understanding_node)
                partitions.exclusive_descendants.add_edge(Edge(self, understanding_node, f"subquestion_{i}", "question"))
                partitions.exclusive_descendants.add_edge(Edge(self, understanding_node, "hqdt", "hqdt"))
                partitions.exclusive_descendants.add_edge(Edge(self, understanding_node, "question_decomposition_score", "question_decomposition_score"))
                partitions.exclusive_descendants.add_edge(Edge(understanding_node, child_aggregate_node, "question", "subquestions"), order=i)
                partitions.exclusive_descendants.add_edge(Edge(understanding_node, child_aggregate_node, "answer", "subquestion_answers"), order=i)
                partitions.exclusive_descendants.add_edge(Edge(understanding_node, child_aggregate_node, "decomposition_score", "child_decomposition_scores"), order=i)
                for dependency in potential_dependencies:
                    try:
                        partitions.exclusive_descendants.add_edge(Edge(understanding_nodes[dependency-1], understanding_node, "answer", "dependency_answers"), order=dependency-1)
                    except IndexError:
                        logging.warning(f"Dependency {dependency} is out of range for subquestion {subquestion}")
                        pass
                output_reasoning_states[f"subquestion_{i}"] = subquestion
                understanding_nodes.append(understanding_node)
                
        return output_reasoning_states, None
