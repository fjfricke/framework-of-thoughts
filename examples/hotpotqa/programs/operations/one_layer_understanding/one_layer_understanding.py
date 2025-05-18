from examples.hotpotqa.programs.operations.reasoning.child_aggregate import ChildAggregateReasoning
from examples.hotpotqa.programs.operations.utils import find_dependencies
from llm_graph_optimizer.graph_of_operations.graph_partitions import GraphPartitions
from llm_graph_optimizer.graph_of_operations.types import Dynamic, Edge, ReasoningState
from llm_graph_optimizer.measurement.measurement import Measurement
from llm_graph_optimizer.operations.abstract_operation import AbstractOperation, AbstractOperationFactory
from llm_graph_optimizer.operations.base_operations.pack_unpack_operations import PackOperation
from llm_graph_optimizer.operations.helpers.exceptions import OperationFailed


class OneLayerUnderstanding(AbstractOperation):
    def __init__(self, branch_op: AbstractOperationFactory, reasoning_op: AbstractOperation, params: dict = None, name: str = None):
        input_types = {"subquestions": list[str], "question_decomposition_score": float, "max_depth": int}
        output_types = Dynamic
        super().__init__(input_types, output_types, params, name)
        self.branch_op = branch_op
        self.reasoning_op = reasoning_op

    async def _execute(self, partitions: GraphPartitions, input_reasoning_states: ReasoningState) -> tuple[ReasoningState, Measurement]:
        subquestions = input_reasoning_states["subquestions"]
        question_decomposition_score = input_reasoning_states["question_decomposition_score"]
        max_depth = input_reasoning_states["max_depth"] - 1

        reasoning_nodes = []

        output_reasoning_states = {}

        pack_node = PackOperation(
                output_types={"subquestion_answers": list[str], "child_decomposition_scores": list[float]},
        )
        partitions.exclusive_descendants.add_node(pack_node)

        for i, subquestion in enumerate(subquestions):
            potential_dependencies = find_dependencies(subquestion)
            branch_node = self.branch_op()
            partitions.exclusive_descendants.add_node(branch_node)
            partitions.exclusive_descendants.add_edge(Edge(self, branch_node, f"subquestion_{i}", "question"))

            reasoning_node = self.reasoning_op()
            reasoning_nodes.append(reasoning_node)
            partitions.exclusive_descendants.add_node(reasoning_node)
            partitions.exclusive_descendants.add_edge(Edge(self, reasoning_node, f"subquestion_{i}", "question"))
            partitions.exclusive_descendants.add_edge(Edge(self, reasoning_node, "question_decomposition_score", "question_decomposition_score"))
            partitions.exclusive_descendants.add_edge(Edge(self, reasoning_node, "max_depth", "max_depth"))

            partitions.exclusive_descendants.add_edge(Edge(branch_node, reasoning_node, "should_decompose", "should_decompose"))
            partitions.exclusive_descendants.add_edge(Edge(branch_node, reasoning_node, "decomposition_score", "should_decompose_score"))

            partitions.exclusive_descendants.add_edge(Edge(reasoning_node, pack_node, "answer", "subquestion_answers"), order=i)
            partitions.exclusive_descendants.add_edge(Edge(reasoning_node, pack_node, "decomposition_score", "child_decomposition_scores"), order=i)

            for dependency in potential_dependencies:
                try:
                    partitions.exclusive_descendants.add_edge(Edge(reasoning_nodes[dependency-1], reasoning_node, "answer", "dependency_answers"), order=dependency-1)
                    partitions.exclusive_descendants.add_edge(Edge(reasoning_nodes[dependency-1], branch_node, "answer", "dependency_answers"), order=dependency-1)
                except IndexError:
                    raise OperationFailed(f"Dependency {dependency} is out of range for subquestion {subquestion} with dependencies {potential_dependencies}.")
            
            output_reasoning_states[f"subquestion_{i}"] = subquestion
        # move successor edge to reasoning nodes and make copies
        successor_edges = partitions.descendants.successor_edges(self)
        successor_edges = [edge for edge in successor_edges if type(edge.to_node) is ChildAggregateReasoning]
        successor_edges = [edge for edge in successor_edges if edge.from_node_key in ["subquestion_answers", "child_decomposition_scores"]]
        for successor_edge in successor_edges:
            partitions.move_edge_start_node(current_edge=successor_edge, new_from_node=pack_node, new_from_node_key=successor_edge.from_node_key)

        output_reasoning_states['question_decomposition_score'] = question_decomposition_score
        output_reasoning_states['max_depth'] = max_depth

        return output_reasoning_states, None