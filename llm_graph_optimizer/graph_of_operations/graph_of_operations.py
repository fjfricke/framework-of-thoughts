from typing import TYPE_CHECKING, get_origin
import networkx as nx
from typeguard import TypeCheckError, check_type
from .base_graph import BaseGraph
from .types import Edge, NodeKey, ManyToOne, OneToMany, StateSetFailed
from llm_graph_optimizer.operations.helpers.node_state import NodeState
from .graph_partitions import Descendants, ExclusiveDescendants, GraphPartitions, Predecessors
if TYPE_CHECKING:
    from llm_graph_optimizer.operations.abstract_operation import AbstractOperation

import logging
logger = logging.getLogger(__name__)


class GraphOfOperations(BaseGraph):
    """
    Graph of operations (execution graph).
    """

    def __init__(self, graph: nx.MultiDiGraph = None):
        """
        Initialize the GraphOfOperations (execution graph).

        :param graph: An optional MultiDiGraph object to initialize the graph. Usually created empty.
        """
        super().__init__(graph)
    
    @property
    def processable_nodes(self) -> list["AbstractOperation"]:
        """
        Get all nodes in the graph that are in a processable state.

        :return: A list of processable nodes.
        """
        processable_nodes = [node for node in self._graph.nodes if node.node_state == NodeState.PROCESSABLE]
        return processable_nodes
    
    def set_next_processable(self):
        """
        Set nodes with all predecessors finished to a processable state.
        """
        # set nodes with all predecessors finished to processable
        for node in self._graph.nodes:
            if all(predecessor.node_state in [NodeState.DONE, NodeState.FAILED] for predecessor in self._graph.predecessors(node)):
                if node.node_state == NodeState.WAITING:
                    node.node_state = NodeState.PROCESSABLE
    
    @property
    def all_scheduled(self) -> bool:
        """
        Check if all nodes in the graph have been scheduled.

        :return: True if all nodes are scheduled, False otherwise.
        """
        return all(not node.node_state.not_yet_scheduled for node in self._graph.nodes)
    
    @property
    def all_processed(self) -> bool:
        """
        Check if all nodes in the graph have been processed.

        :return: True if all nodes are processed, False otherwise.
        """
        return all(node.node_state.is_finished for node in self._graph.nodes)

    def add_node(self, node: "AbstractOperation"):
        """
        Add a node to the graph.

        :param node: The node to add.
        """
        super()._add_node(node)

    def add_edge(self, edge: Edge, order: int=0, idx: int=0):
        """
        Add an edge to the graph.

        :param edge: The edge to add.
        :param order: The order of the edge. This is needed when using ManyToOne types to ensure that returned lists are in the order given.
        :param idx: The index of the edge. This is needed when using OneToMany types to ensure that returned lists are in the order given.
        """
        super()._add_edge(edge, order, idx)

    def add_dependency_edge(self, from_node: "AbstractOperation", to_node: "AbstractOperation"):
        """
        Add a dependency edge between two nodes in the graph without reasoning state flow.

        :param from_node: The source node of the dependency.
        :param to_node: The target node of the dependency.
        """
        super()._add_dependency_edge(from_node, to_node)

    def remove_node(self, node: "AbstractOperation"):
        """
        Remove a node from the graph.

        :param node: The node to remove.
        """
        super()._remove_node(node)
    
    def remove_edge(self, edge: Edge):
        """
        Remove an edge from the graph.

        :param edge: The edge to remove.
        """
        super()._remove_edge(edge)
    
    def update_edge_values(self, from_node: "AbstractOperation", value: dict[NodeKey, any]):
        """
        Update the values of edges originating from a specific node.

        :param from_node: The node whose outgoing edges will be updated.
        :param value: A dictionary of values to update the edges with.
        """
        super()._update_edge_values(from_node, value)

    def get_input_reasoning_states(self, node: "AbstractOperation") -> dict[NodeKey, any]:
        """
        Build the input reasoning states for `node`.

        For ManyToOne inputs:
        - collect values from all predecessors for the same `to_node_key`
        - globally sort them by `order`
        - allow duplicate orders (all values with the same order go into the list, order among them is arbitrary but logged)
        """
        if node == self.start_node:
            return node.input_reasoning_states

        predecessors = self._graph.predecessors(node)
        input_reasoning_states: dict[NodeKey, any] = {}
        buckets: dict[NodeKey, list[tuple[int, any]]] = {}
        single_values: dict[NodeKey, any] = {}

        for predecessor in predecessors:
            if not predecessor.node_state.is_finished:
                raise ValueError(f"Predecessor {predecessor} is not finished")

            edge_data = self._graph.get_edge_data(predecessor, node)
            if not edge_data:
                continue

            for e in edge_data.values():
                to_node_key = e.get("to_node_key")
                if to_node_key is None:
                    continue

                value = e.get("value") if predecessor.node_state != NodeState.FAILED else StateSetFailed
                try:
                    check_type(value, OneToMany)
                    value = value[e["idx"]]
                except TypeCheckError:
                    pass

                expected_type = node.input_types[to_node_key]
                if get_origin(expected_type) == ManyToOne:
                    order = e.get("order", -1)
                    buckets.setdefault(to_node_key, []).append((order, value))
                else:
                    single_values[to_node_key] = value

        for key, expected_type in node.input_types.items():
            if get_origin(expected_type) == ManyToOne:
                pairs = buckets.get(key, [])
                if not pairs:
                    input_reasoning_states[key] = []
                    continue

                pairs.sort(key=lambda p: p[0])
                orders = [o for o, _ in pairs]
                if len(set(orders)) != len(orders):
                    logger.warning(
                        f"Duplicate 'order' values detected for ManyToOne key={key}: {orders}"
                    )

                input_reasoning_states[key] = [v for _, v in pairs]
            else:
                input_reasoning_states[key] = single_values.get(key, None)

        return input_reasoning_states
    
    def partitions(self, node: "AbstractOperation") -> GraphPartitions:
        """
        Partition the graph into predecessors, descendants, and exclusive descendants of a node (all containing the node itself).

        :param node: The node to partition the graph around.
        :return: A GraphPartitions object containing the partitions.
        """
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
        """
        Get the start node of the graph.

        :return: The start node of the graph.
        """
        # Retrieve the start node from the graph attributes
        if 'start_node' not in self._graph.graph:
            start_nodes = [node for node in self._graph.nodes if self._graph.in_degree(node) == 0]
            if len(start_nodes) != 1:
                raise ValueError("Graph must have exactly one start node with in-degree 0")
            self._graph.graph['start_node'] = start_nodes[0]  # Store as graph attribute
        return self._graph.graph['start_node']

    @property
    def end_node(self) -> "AbstractOperation":
        """
        Get the end node of the graph.

        :return: The end node of the graph.
        """
        # Retrieve the end node from the graph attributes
        if 'end_node' not in self._graph.graph:
            end_nodes = [node for node in self._graph.nodes if self._graph.out_degree(node) == 0]
            if len(end_nodes) != 1:
                raise ValueError("Graph must have exactly one end node with out-degree 0")
            self._graph.graph['end_node'] = end_nodes[0]  # Store as graph attribute
        return self._graph.graph['end_node']



