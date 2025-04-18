import logging
import re
from examples.hotpotqa.programs.operations.reasoning.child_aggregate import ChildAggregateReasoning
from examples.hotpotqa.programs.utils import find_dependencies
from llm_graph_optimizer.graph_of_operations.graph_of_operations import GraphOfOperations
from llm_graph_optimizer.graph_of_operations.graph_partitions import GraphPartitions
from llm_graph_optimizer.graph_of_operations.types import Dynamic, Edge, ManyToOne, StateNotSet
from llm_graph_optimizer.operations.abstract_operation import AbstractOperation, AbstractOperationFactory
from llm_graph_optimizer.operations.end import End
from llm_graph_optimizer.operations.filter_operation import FilterOperation
from llm_graph_optimizer.operations.pack_unpack_operations import PackOperation
from llm_graph_optimizer.operations.start import Start
from llm_graph_optimizer.operations.test_operation import TestOperation


class UnderstandingGraphUpdating(AbstractOperation):
    def __init__(self, open_book_op: AbstractOperationFactory, closed_book_op: AbstractOperationFactory, child_aggregate_op: AbstractOperationFactory, understanding_op: AbstractOperationFactory, params: dict = None, name: str = None):
        input_types = {"hqdt": dict, "question": str, "question_decomposition_score": float, "dependency_answers": ManyToOne[str]}
        output_types = Dynamic
        super().__init__(input_types, output_types, params, name)
        self.open_book_op = open_book_op
        self.closed_book_op = closed_book_op
        self.child_aggregate_op = child_aggregate_op
        self.understanding_op = understanding_op

    async def _execute(self, partitions: GraphPartitions, input_reasoning_states: dict[str | int, any]) -> dict[str | int, any]:
        

        # get state data
        hqdt = input_reasoning_states["hqdt"]
        current_question = input_reasoning_states["question"]
        if current_question not in hqdt:
            subquestions = []
        else:
            subquestions = hqdt[current_question][0]

        dependency_answers = list(input_reasoning_states["dependency_answers"])
        output_reasoning_states = {"hqdt": hqdt, "question": current_question, "answer": StateNotSet, "question_decomposition_score": StateNotSet, "dependency_answers": dependency_answers}

        def filter_operation(length: int):
            def filter_function(input_list: list[dict[str, any]]) -> dict[str, any]:
                return max(input_list, key=lambda x: x["decomposition_score"])
            return FilterOperation(output_types={"answer": str, "decomposition_score": float}, length=length, filter_function=filter_function)

        

        pack_op = lambda: PackOperation(input_types={"answer": str, "decomposition_score": float}, output_key="packed")

        # create reasoning nodes
        open_book_node = self.open_book_op()
        partitions.exclusive_descendants.add_node(open_book_node)
        partitions.exclusive_descendants.add_edge(Edge(self, open_book_node, "question", "question"))
        partitions.exclusive_descendants.add_edge(Edge(self, open_book_node, "dependency_answers", "dependency_answers"))
        pack_open_book_node = pack_op()
        partitions.exclusive_descendants.add_node(pack_open_book_node)
        partitions.exclusive_descendants.add_edge(Edge(open_book_node, pack_open_book_node, "answer", "answer"))
        partitions.exclusive_descendants.add_edge(Edge(open_book_node, pack_open_book_node, "decomposition_score", "decomposition_score"))
        closed_book_node = self.closed_book_op()
        partitions.exclusive_descendants.add_node(closed_book_node)
        partitions.exclusive_descendants.add_edge(Edge(self, closed_book_node, "question", "question"))
        partitions.exclusive_descendants.add_edge(Edge(self, closed_book_node, "dependency_answers", "dependency_answers"))
        pack_closed_book_node = pack_op()
        partitions.exclusive_descendants.add_node(pack_closed_book_node)
        partitions.exclusive_descendants.add_edge(Edge(closed_book_node, pack_closed_book_node, "answer", "answer"))
        partitions.exclusive_descendants.add_edge(Edge(closed_book_node, pack_closed_book_node, "decomposition_score", "decomposition_score"))
        if subquestions:
            output_reasoning_states["decomposition_score"] = hqdt[current_question][1]
            child_aggregate_node = self.child_aggregate_op()
            partitions.exclusive_descendants.add_node(child_aggregate_node)
            partitions.exclusive_descendants.add_edge(Edge(self, child_aggregate_node, "question", "question"))
            partitions.exclusive_descendants.add_edge(Edge(self, child_aggregate_node, "question_decomposition_score", "question_decomposition_score"))
            partitions.exclusive_descendants.add_edge(Edge(self, child_aggregate_node, "dependency_answers", "dependency_answers"))
            pack_child_aggregate_node = pack_op()
            partitions.exclusive_descendants.add_node(pack_child_aggregate_node)
            partitions.exclusive_descendants.add_edge(Edge(child_aggregate_node, pack_child_aggregate_node, "answer", "answer"))
            partitions.exclusive_descendants.add_edge(Edge(child_aggregate_node, pack_child_aggregate_node, "decomposition_score", "decomposition_score"))
            filter_node = filter_operation(3)
        else:
            filter_node = filter_operation(2)

        partitions.exclusive_descendants.add_node(filter_node)
        partitions.exclusive_descendants.add_edge(Edge(pack_open_book_node, filter_node, "packed", 0))
        partitions.exclusive_descendants.add_edge(Edge(pack_closed_book_node, filter_node, "packed", 1))
        if subquestions:
            partitions.exclusive_descendants.add_edge(Edge(pack_child_aggregate_node, filter_node, "packed", 2))

        # move successor edges to UnderstandingGraphUpdating nodes to filter node iff the key is "answer" or "decomposition_score"
        successor_edges = partitions.descendants.successor_edges(self)
        successor_edges = [edge for edge in successor_edges if type(edge.to_node) in [End, UnderstandingGraphUpdating]]
        successor_edges = [edge for edge in successor_edges if edge.to_node_key in ["answer", "decomposition_score"]]
        for successor_edge in successor_edges:
            partitions.move_edge(current_edge=successor_edge, new_from_node=filter_node, new_from_node_key=successor_edge.to_node_key)

        successor_edges = partitions.descendants.successor_edges(self)
        successor_edges = [edge for edge in successor_edges if type(edge.to_node) is UnderstandingGraphUpdating]
        successor_edges = [edge for edge in successor_edges if edge.to_node_key == "dependency_answers"]
        for successor_edge in successor_edges:
            partitions.move_edge(current_edge=successor_edge, new_from_node=filter_node, new_from_node_key="answer")

        # move successor edges going to previous aggregate to start at filter node
        to_key_to_from_key = {
            "subquestion_answers": "answer",
            "child_decomposition_scores": "decomposition_score"
        }
        successor_edges = partitions.descendants.successor_edges(self)
        successor_edges = [edge for edge in successor_edges if type(edge.to_node) is ChildAggregateReasoning]
        successor_edges = [edge for edge in successor_edges if edge.to_node_key in ["subquestion_answers", "child_decomposition_scores"]]
        for successor_edge in successor_edges:
            partitions.move_edge(current_edge=successor_edge, new_from_node=filter_node, new_from_node_key=to_key_to_from_key[successor_edge.to_node_key])

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
                
        return output_reasoning_states
    
if __name__ == "__main__":
    import asyncio

    async def main():
        
        operation = UnderstandingGraphUpdating(
            open_book_op=lambda: TestOperation({"question": str, "dependency_answers": list[str]}, {"answer": str, "decomposition_score": float}, name="open_book_op"),
            closed_book_op=lambda: TestOperation({"question": str, "dependency_answers": list[str]}, {"answer": str, "decomposition_score": float}, name="closed_book_op"),
            child_aggregate_op=lambda: TestOperation({"question": str, "question_decomposition_score": float, "dependency_answers": list[str], "subquestions": ManyToOne[str], "subquestion_answers": ManyToOne[str], "child_decomposition_scores": ManyToOne[float]}, {"answer": str, "decomposition_score": float}, name="child_aggregate_op")
        )

        graph = GraphOfOperations()
        graph.add_node(operation)
        start = Start(input_types={"question": str})
        graph.add_node(start)
        end = End(input_types={"answer": str})
        graph.add_node(end)
        graph.add_edge(Edge(start, operation, "question", "question"))
        graph.add_edge(Edge(operation, end, "answer", "answer"))
        hqdt = {'What is the combined population of the biggest 2 neighbour country of the largest country in Europe by capita?': (['What is the largest country in Europe by capita?', 'What are the two biggest neighboring countries of #1?', 'What is the combined population of #2?'], -0.0846547199021159), 'What are the two biggest neighboring countries of #1?': (['What countries border #1?', 'Which are the two biggest countries by area or population among #1?'], -0.20921424507292)}
        await operation._execute(partitions=graph.partitions(operation), input_reasoning_states={"hqdt": hqdt, "question": "What is the combined population of the biggest 2 neighbour country of the largest country in Europe by capita?"})
        graph.view_graph(use_pyvis=True, show_keys=True, edge_length_power=2)
    asyncio.run(main())