from enum import Enum
from dataclasses import dataclass
from examples.game_of_24.programs.operations.value_operation import ValueOperation
from llm_graph_optimizer.graph_of_operations.graph_partitions import GraphPartitions
from llm_graph_optimizer.graph_of_operations.types import NodeKey
from llm_graph_optimizer.operations.abstract_operation import AbstractOperation
from llm_graph_optimizer.operations.base_operations.start import Start

class FindLastValuesType(Enum):
    ONLY_ONE = "only_one"
    ALL = "all"

@dataclass
class FindLastValuesNodes:
    left_nodes: list[AbstractOperation]
    expression_nodes: list[AbstractOperation]
    left_nodekeys: list[NodeKey]
    expression_nodekeys: list[NodeKey]

    def reverse(self) -> "FindLastValuesNodes":
        return FindLastValuesNodes(left_nodes=self.left_nodes[::-1], expression_nodes=self.expression_nodes[::-1], left_nodekeys=self.left_nodekeys[::-1], expression_nodekeys=self.expression_nodekeys[::-1])


def find_nodes(current: AbstractOperation, partitions: GraphPartitions, find_last_values_type: FindLastValuesType) -> FindLastValuesNodes:
    """
    Find the nodes and nodekeys of the previous left and expression nodes in the graph.
    If find_last_values_type is FindLastValuesType.ONLY_ONE, only the previous left and expression nodes of the last FindLastValueOperation or ValueOperation are returned.
    """
    left_nodes = []
    expression_nodes = []
    left_nodekeys = []
    expression_nodekeys = []
    while True:
        if isinstance(current, ValueOperation) or current.name.startswith("LastStepValueOperation"):
            predecessor_edges = partitions.predecessors.predecessor_edges(current)
            from_node = predecessor_edges[0].from_node # parallel_evaluation_node
            to_node_key_to_from_node_key = {edge.to_node_key: edge.from_node_key for edge in predecessor_edges}
            left_nodes.append(from_node)
            expression_nodes.append(from_node)
            left_nodekeys.append(to_node_key_to_from_node_key["left"])
            expression_nodekeys.append(to_node_key_to_from_node_key["expression"])

            if find_last_values_type == FindLastValuesType.ONLY_ONE:
                break

        if isinstance(current, Start):
            left_nodes.append(current)
            left_nodekeys.append("input_list")
            break
        
        if current.name == "Propose" or current.name.startswith("LLMEvaluate"):  # Find only dependency edges
            previous_dependency_nodes = partitions.predecessors.direct_predecessors(current, include_dependencies=True) - partitions.predecessors.direct_predecessors(current, include_dependencies=False)
            if len(previous_dependency_nodes) == 0:  # we are at Start
                current = list(partitions.predecessors.direct_predecessors(current, include_dependencies=False))[0]
            else:
                current = list(previous_dependency_nodes)[0] # FindLastValueOperation or ValueOperation
        else:  # Find only non-dependency edges
            current = list(partitions.predecessors.direct_predecessors(current, include_dependencies=False))[0]
    return FindLastValuesNodes(left_nodes=left_nodes, expression_nodes=expression_nodes, left_nodekeys=left_nodekeys, expression_nodekeys=expression_nodekeys)