from typing import TYPE_CHECKING, get_origin
import networkx as nx
from typeguard import TypeCheckError, check_type
import pickle
from .base_graph import BaseGraph
from .types import Edge, NodeKey, ManyToOne
from llm_graph_optimizer.operations.helpers.node_state import NodeState
from .graph_partitions import Descendants, ExclusiveDescendants, GraphPartitions, Predecessors
if TYPE_CHECKING:
    from llm_graph_optimizer.operations.abstract_operation import AbstractOperation


class GraphOfOperations(BaseGraph):
    """
    Graph of operations.
    """

    def __init__(self, graph: nx.MultiDiGraph = None):
        super().__init__(graph)
    
    @property
    def processable_nodes(self) -> list["AbstractOperation"]:
        processable_nodes = [node for node in self._graph.nodes if node.node_state == NodeState.PROCESSABLE]
        return processable_nodes
    
    def set_next_processable(self):
        # set nodes with all predecessors finished to processable
        for node in self._graph.nodes:
            if all(predecessor.node_state in [NodeState.DONE, NodeState.FAILED] for predecessor in self._graph.predecessors(node)):
                if node.node_state == NodeState.WAITING:
                    node.node_state = NodeState.PROCESSABLE
    
    @property
    def all_scheduled(self) -> bool:
        return all(not node.node_state.not_yet_scheduled for node in self._graph.nodes)
    
    @property
    def all_processed(self) -> bool:
        return all(node.node_state.is_finished for node in self._graph.nodes)

    def add_node(self, node: "AbstractOperation"):
        super()._add_node(node)

    def add_edge(self, edge: Edge, order: int=0):
        super()._add_edge(edge, order)

    def remove_node(self, node: "AbstractOperation"):
        super()._remove_node(node)
    
    def remove_edge(self, edge: Edge):
        super()._remove_edge(edge)
    
    def update_edge_values(self, from_node: "AbstractOperation", value: dict[NodeKey, any]):
        super()._update_edge_values(from_node, value)

    def get_input_reasoning_states(self, node: "AbstractOperation") -> dict[NodeKey, any]:
        if node == self.start_node:
            return node.input_reasoning_states
        predecessors = self._graph.predecessors(node)
        input_reasoning_states = {}
        for key, expected_type in node.input_types.items():
            if get_origin(expected_type) == ManyToOne:
                input_reasoning_states[key] = []
        for predecessor in predecessors:
            if not predecessor.node_state.is_finished:
                raise ValueError(f"Predecessor {predecessor} is not finished")
            edge_data = self._graph.get_edge_data(predecessor, node)
            edge_data_values_sorted = sorted(edge_data.values(), key=lambda x: x["order"])
            for edge in edge_data_values_sorted:
                to_node_key = edge["to_node_key"]
                if to_node_key is not None:
                    value = edge.get("value")
                    # Check if the type is OneToManyList or a parameterized version of it
                    if get_origin(node.input_types[to_node_key]) == ManyToOne:
                        # Aggregate values into a list
                        input_reasoning_states[to_node_key].append(value)
                    else:
                        # Assign the value directly for non-OneToManyList types
                        input_reasoning_states[to_node_key] = value
        return input_reasoning_states
    
    def partitions(self, node: "AbstractOperation") -> GraphPartitions:
        all_nodes = set(self._graph.nodes)

        # Compute predecessors and descendants
        predecessors_nodes = nx.ancestors(self._graph, node) | {node}
        descendants_nodes = nx.descendants(self._graph, node) | {node}

        unconnected_nodes = all_nodes - predecessors_nodes - descendants_nodes - {node}
        unconnected_descendant_nodes = set()
        for unconnected_node in unconnected_nodes:
            unconnected_descendant_nodes.update(nx.descendants(self._graph, unconnected_node))
        exclusive_descendant_nodes = descendants_nodes - unconnected_descendant_nodes | {node}

        return GraphPartitions(
            predecessors=Predecessors(self, self._graph.subgraph(predecessors_nodes)),
            descendants=Descendants(self, self._graph.subgraph(descendants_nodes)),
            exclusive_descendants=ExclusiveDescendants(self, self._graph.subgraph(exclusive_descendant_nodes))
            )

    @property
    def start_node(self) -> "AbstractOperation":
        # Retrieve the start node from the graph attributes
        if 'start_node' not in self._graph.graph:
            start_nodes = [node for node in self._graph.nodes if self._graph.in_degree(node) == 0]
            if len(start_nodes) != 1:
                raise ValueError("Graph must have exactly one start node with in-degree 0")
            self._graph.graph['start_node'] = start_nodes[0]  # Store as graph attribute
        return self._graph.graph['start_node']

    @property
    def end_node(self) -> "AbstractOperation":
        # Retrieve the end node from the graph attributes
        if 'end_node' not in self._graph.graph:
            end_nodes = [node for node in self._graph.nodes if self._graph.out_degree(node) == 0]
            if len(end_nodes) != 1:
                raise ValueError("Graph must have exactly one end node with out-degree 0")
            self._graph.graph['end_node'] = end_nodes[0]  # Store as graph attribute
        return self._graph.graph['end_node']



