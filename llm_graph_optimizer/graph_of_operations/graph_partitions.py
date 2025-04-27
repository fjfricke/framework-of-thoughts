from typing import TYPE_CHECKING
import warnings
from .base_graph import BaseGraph
from .types import Edge, NodeKey
from networkx import MultiDiGraph

if TYPE_CHECKING:
    from llm_graph_optimizer.operations.abstract_operation import AbstractOperation


class Predecessors(BaseGraph):
    """
    Represents the predecessors of a node in the graph.
    This is a subgraph containing all nodes and edges that precede a given node and the node itself.
    """

    def __init__(self, original_graph: BaseGraph, subgraph: MultiDiGraph):
        """
        Initialize the Predecessors subgraph. Called from GraphOfOperations.partitions.

        :param original_graph: The original graph from which this subgraph is derived.
        :param subgraph: The subgraph containing the predecessors.
        """
        super().__init__(subgraph)
        self.original_graph = original_graph

    def add_node(self, node: "AbstractOperation"):
        """
        Adding nodes is forbidden in Predecessors.

        :param node: The node to add (not allowed).
        :raises PermissionError: Always raised when this method is called.
        """
        raise PermissionError("Adding nodes is forbidden in Predecessors.")

    def add_edge(self, edge: Edge):
        """
        Adding edges is forbidden in Predecessors.

        :param edge: The edge to add (not allowed).
        :raises PermissionError: Always raised when this method is called.
        """
        raise PermissionError("Adding edges is forbidden in Predecessors.")
    
    def remove_node(self, node: "AbstractOperation"):
        """
        Removing nodes is forbidden in Predecessors.

        :param node: The node to remove (not allowed).
        :raises PermissionError: Always raised when this method is called.
        """
        raise PermissionError("Removing nodes is forbidden in Predecessors.")
    
    def remove_edge(self, edge: Edge):
        """
        Removing edges is forbidden in Predecessors.

        :param edge: The edge to remove (not allowed).
        :raises PermissionError: Always raised when this method is called.
        """
        raise PermissionError("Removing edges is forbidden in Predecessors.")
    
    @property
    def start_node(self) -> "AbstractOperation":
        """
        Get the start node of the Predecessors subgraph.

        :return: The start node of the subgraph.
        """
        return super().start_node

    @property
    def end_node(self) -> "AbstractOperation":
        """
        Get the end node of the Predecessors subgraph.

        :return: The end node of the subgraph.
        :raises ValueError: If the subgraph does not have exactly one end node.
        """
        end_nodes = [node for node in self._graph.nodes if self._graph.out_degree(node) == 0]
        if len(end_nodes) != 1:
            raise ValueError("Predecessor Graph must have exactly one end node with out-degree 0")
        return end_nodes[0]
    
    def __contains__(self, node: "AbstractOperation") -> bool:
        """
        Check if a node exists in the Predecessors subgraph.

        :param node: The node to check.
        :return: True if the node exists in the subgraph, False otherwise.
        """
        return node in self._graph.nodes

class ExclusiveDescendants(BaseGraph):
    """
    Represents the exclusive descendants of a node in the graph and the node itself.
    This is a subgraph containing all nodes and edges that are exclusively reachable from a given node.
    """

    def __init__(self, original_graph: BaseGraph, subgraph: MultiDiGraph):
        """
        Initialize the ExclusiveDescendants subgraph. Called from GraphOfOperations.partitions.

        :param original_graph: The original graph from which this subgraph is derived.
        :param subgraph: The subgraph containing the exclusive descendants.
        """
        super().__init__(subgraph)
        self.original_graph = original_graph
        self.new_nodes = set()
    
    def add_node(self, node: "AbstractOperation"):
        """
        Add a node to the ExclusiveDescendants subgraph.

        :param node: The node to add.
        """
        self.original_graph._add_node(node)
        self.new_nodes.add(node)
    
    def add_edge(self, edge: Edge, order: int=0):
        """
        Add an edge to the ExclusiveDescendants subgraph.

        :param edge: The edge to add.
        :param order: The order of the edge.
        :raises ValueError: If either end of the edge is not in the subgraph.
        """
        warnings.warn("Deprecated: Use partitions.add_edge instead.", DeprecationWarning)
        if edge.from_node in (self._graph.nodes | self.new_nodes) and edge.to_node in (self._graph.nodes | self.new_nodes):
            self.original_graph._add_edge(edge, order)
        else:
            raise ValueError(f"Both ends of the edge must be in the exclusive descendants graph. {edge.from_node} or {edge.to_node} is/are not in the original graph.")
        
    def add_dependency_edge(self, from_node: "AbstractOperation", to_node: "AbstractOperation"):
        """
        Add a dependency edge between two nodes in the ExclusiveDescendants subgraph.

        :param from_node: The source node of the dependency.
        :param to_node: The target node of the dependency.
        """
        self.original_graph._add_dependency_edge(from_node, to_node)

    def remove_node(self, node: "AbstractOperation"):
        """
        Removing nodes is forbidden in ExclusiveDescendants. Use the function in the GraphPartitions class instead.

        :param node: The node to remove (not allowed).
        :raises PermissionError: Always raised when this method is called.
        """
        raise PermissionError("Removing nodes is forbidden in ExclusiveDescendants. Use the function in the GraphPartitions class instead.")
    
    def remove_edge(self, edge: Edge):
        """
        Remove an edge from the ExclusiveDescendants subgraph.

        :param edge: The edge to remove.
        :raises ValueError: If the edge is not in the subgraph.
        """
        if edge in self.edges:
            self.original_graph._remove_edge(edge)
        else:
            raise ValueError(f"Edge {edge} is not in the exclusive descendants graph.")
        
    def successor_edges(self, node: "AbstractOperation") -> list[Edge]:
        """
        Get the successor edges of a node in the ExclusiveDescendants subgraph.

        :param node: The node to retrieve successor edges for.
        :return: A list of successor edges.
        """
        successor_edges = self._graph.out_edges(node, data=True)
        return Edge.from_edge_view(successor_edges)
    
    @property
    def start_node(self) -> "AbstractOperation":
        """
        Get the start node of the ExclusiveDescendants subgraph.

        :return: The start node of the subgraph.
        :raises ValueError: If the subgraph does not have exactly one start node.
        """
        start_nodes = [node for node in self._graph.nodes if self._graph.in_degree(node) == 0]
        if len(start_nodes) != 1:
            raise ValueError("DescendantGraph must have exactly one start node with in-degree 0")
        return start_nodes[0]
    
    @property
    def end_node(self) -> "AbstractOperation":
        """
        End node is not defined for ExclusiveDescendants.

        :raises PermissionError: Always raised when this method is called.
        """
        raise PermissionError("End node is not defined for ExclusiveDescendants.")
    
    def __contains__(self, node: "AbstractOperation") -> bool:
        """
        Check if a node exists in the ExclusiveDescendants subgraph.

        :param node: The node to check.
        :return: True if the node exists in the subgraph, False otherwise.
        """
        return node in self._graph.nodes or node in self.new_nodes

class Descendants(BaseGraph):
    """
    Represents the descendants of a node in the graph.
    This is a subgraph containing all nodes and edges that are reachable from a given node and the node itself.
    """

    def __init__(self, original_graph: BaseGraph, subgraph: MultiDiGraph):
        """
        Initialize the Descendants subgraph. Called from GraphOfOperations.partitions.

        :param original_graph: The original graph from which this subgraph is derived.
        :param subgraph: The subgraph containing the descendants.
        """
        super().__init__(subgraph)
        self.original_graph = original_graph
    
    def add_node(self, node: "AbstractOperation"):
        """
        Adding nodes is forbidden in Descendants. Use the function in the  ExclusiveDescendants class instead.

        :param node: The node to add (not allowed).
        :raises PermissionError: Always raised when this method is called.
        """
        raise PermissionError("Adding nodes is forbidden in Descendants.")
    
    def add_edge(self, edge: Edge):
        """
        Adding edges is forbidden in Descendants.

        :param edge: The edge to add (not allowed).
        :raises PermissionError: Always raised when this method is called.
        """
        raise PermissionError("Adding edges is forbidden in Descendants.")
    
    def remove_node(self, node: "AbstractOperation"):
        """
        Removing nodes is forbidden in Descendants.

        :param node: The node to remove (not allowed).
        :raises PermissionError: Always raised when this method is called.
        """
        raise PermissionError("Removing nodes is forbidden in Descendants.")
    
    def remove_edge(self, edge: Edge):
        """
        Removing edges is forbidden in Descendants.

        :param edge: The edge to remove (not allowed).
        :raises PermissionError: Always raised when this method is called.
        """
        raise PermissionError("Removing edges is forbidden in Descendants.")
    
    def _move_edge(self, current_edge: Edge, new_from_node: "AbstractOperation", new_from_node_key: NodeKey, order: int=0):
        """
        Move an edge within the Descendants subgraph. Do not use this function directly. Use the function in the GraphPartitions class instead.

        :param current_edge: The edge to move.
        :param new_from_node: The new source node for the edge.
        :param new_from_node_key: The new source key for the edge.
        :param order: The order of the edge.
        """
        self.original_graph._remove_edge(current_edge)
        self.original_graph._add_edge(Edge(new_from_node, current_edge.to_node, new_from_node_key, current_edge.to_node_key), order)
    
    def successor_edges(self, node: "AbstractOperation") -> list[Edge]:
        """
        Get the successor edges of a node in the Descendants subgraph.

        :param node: The node to retrieve successor edges for.
        :return: A list of successor edges.
        """
        successor_edges = self._graph.out_edges(node, data=True)
        return Edge.from_edge_view(successor_edges)
    
    @property
    def start_node(self) -> "AbstractOperation":
        """
        Get the start node of the Descendants subgraph.

        :return: The start node of the subgraph.
        :raises ValueError: If the subgraph does not have exactly one start node.
        """
        start_nodes = [node for node in self._graph.nodes if self._graph.in_degree(node) == 0]
        if len(start_nodes) != 1:
            raise ValueError("DescendantGraph must have exactly one start node with in-degree 0")
        return start_nodes[0]
    
    @property
    def end_node(self) -> "AbstractOperation":
        """
        Get the end node of the Descendants subgraph.

        :return: The end node of the subgraph.
        """
        return super().end_node
    
    def __contains__(self, node: "AbstractOperation") -> bool:
        """
        Check if a node exists in the Descendants subgraph.

        :param node: The node to check.
        :return: True if the node exists in the subgraph, False otherwise.
        """
        return node in self._graph.nodes


class GraphPartitions:
    """
    Represents the partitions of a graph around a specific node.
    Contains the predecessors, descendants, and exclusive descendants of the node, as well as functions to move or add edges between the partitions.
    """

    predecessors: Predecessors
    descendants: Descendants
    exclusive_descendants: ExclusiveDescendants

    def __init__(self, predecessors: Predecessors, descendants: Descendants, exclusive_descendants: ExclusiveDescendants):
        """
        Initialize the GraphPartitions. Called from GraphOfOperations.partitions.

        :param predecessors: The predecessors partition.
        :param descendants: The descendants partition.
        :param exclusive_descendants: The exclusive descendants partition.
        """
        self.predecessors = predecessors
        self.descendants = descendants
        self.exclusive_descendants = exclusive_descendants
        self.original_graph = predecessors.original_graph
    def move_edge(self, current_edge: Edge, new_from_node: "AbstractOperation", new_from_node_key: NodeKey):
        """
        Move an edge within the graph partitions. Only allowed between the descendants and exclusive descendants partitions.

        :param current_edge: The edge to move.
        :param new_from_node: The new source node for the edge.
        :param new_from_node_key: The new source key for the edge.
        :raises ValueError: If the edge or nodes are not in the appropriate partitions.
        """
        edge_data = self.descendants.get_edge_data(current_edge)
        if edge_data is None:
            raise ValueError(f"Edge {current_edge} does not exist in the graph.")
        if current_edge.from_node not in self.exclusive_descendants:
            raise ValueError(f"In order to move an edge, the previous from_node must be in the exclusive Descendants graph. {current_edge.from_node} is not.")
        if new_from_node not in self.exclusive_descendants:
            raise ValueError(f"In order to move an edge, the new from_node must be in the exclusive Descendants graph. {new_from_node} is not.")
        self.original_graph._remove_edge(current_edge)
        self.original_graph._add_edge(Edge(new_from_node, current_edge.to_node, new_from_node_key, current_edge.to_node_key), order=edge_data.get("order", 0))

    def add_edge(self, edge: Edge, order: int=0):
        """
        Add an edge to the graph partitions. Only allowed between the predecessors and exclusive descendants partitions. For edges inside a partition, use the function in their respective classes.

        :param edge: The edge to add.
        :param order: The order of the edge.
        :raises ValueError: If the edge is not in the appropriate partitions.
        """
        if not (edge.from_node in self.predecessors or edge.from_node in self.exclusive_descendants):
            raise ValueError(f"The from_node must be in the predecessors or exclusive descendants graph. {edge.from_node} is not.")
        if edge.to_node not in self.exclusive_descendants:
            raise ValueError(f"The to_node must be in the exclusive descendants graph. {edge.to_node} is not.")
        self.original_graph._add_edge(edge, order)
        if edge.from_node in self.predecessors:
            self.predecessors.original_graph._update_new_from_predecessor_edge_values(edge.from_node, edge.to_node, edge.from_node_key)

    def remove_node(self, node: "AbstractOperation"):
        """
        Remove a node from exclusive descendants.

        :param node: The node to remove.
        :raises ValueError: If the node is not in the appropriate partitions.
        """
        if node in self.exclusive_descendants and node not in self.descendants:
            self.original_graph._remove_node(node)
        else:
            raise ValueError(f"Node {node} is not in the exclusive Descendants graph or points to non-exclusivedescendants")