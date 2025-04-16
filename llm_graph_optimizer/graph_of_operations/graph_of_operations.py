from typing import TYPE_CHECKING
import networkx as nx

from .base_graph import BaseGraph
from .types import NodeKey
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
        return [node for node in self._graph.nodes if node.node_state == NodeState.PROCESSABLE]
    
    def set_next_processable(self):
        # set nodes with all predecessors finished to processable
        for node in self._graph.nodes:
            if all(predecessor.node_state == NodeState.DONE for predecessor in self._graph.predecessors(node)):
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

    def add_edge(self, from_node: "AbstractOperation", to_node: "AbstractOperation", from_node_key: NodeKey, to_node_key: NodeKey):
        super()._add_edge(from_node, to_node, from_node_key=from_node_key, to_node_key=to_node_key)

    def remove_node(self, node: "AbstractOperation"):
        super()._remove_node(node)
    
    def remove_edge(self, from_node: "AbstractOperation", to_node: "AbstractOperation", from_node_key: NodeKey, to_node_key: NodeKey):
        super()._remove_edge(from_node, to_node, from_node_key=from_node_key, to_node_key=to_node_key)
    
    def update_edge_values(self, from_node: "AbstractOperation", value: dict[NodeKey, any]):
        super()._update_edge_values(from_node, value)

    def get_input_reasoning_states(self, node: "AbstractOperation") -> dict[NodeKey, any]:
        if node == self.start_node:
            return node.input_reasoning_states
        predecessors = self._graph.predecessors(node)
        input_reasoning_states = {}
        for predecessor in predecessors:
            if not predecessor.node_state.is_finished:
                raise ValueError(f"Predecessor {predecessor} is not finished")
            edge_data = self._graph.get_edge_data(predecessor, node)
            for edge in edge_data.values():
                to_node_key = edge["to_node_key"]
                if to_node_key is not None:
                    value = edge.get("value")
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
            predecessors=Predecessors(self._graph.subgraph(predecessors_nodes)),
            descendants=Descendants(self._graph.subgraph(descendants_nodes)),
            exclusive_descendants=ExclusiveDescendants(self._graph.subgraph(exclusive_descendant_nodes))
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



