from typing import TYPE_CHECKING
import warnings
from .base_graph import BaseGraph
from .types import Edge, NodeKey
from networkx import MultiDiGraph

if TYPE_CHECKING:
    from llm_graph_optimizer.operations.abstract_operation import AbstractOperation


class Predecessors(BaseGraph):

    def __init__(self, original_graph: BaseGraph, subgraph: MultiDiGraph):
        super().__init__(subgraph)
        self.original_graph = original_graph

    def add_node(self, node: "AbstractOperation"):
        raise PermissionError("Adding nodes is forbidden in Predecessors.")

    def add_edge(self, edge: Edge):
        raise PermissionError("Adding edges is forbidden in Predecessors.")
    
    def remove_node(self, node: "AbstractOperation"):
        raise PermissionError("Removing nodes is forbidden in Predecessors.")
    
    def remove_edge(self, edge: Edge):
        raise PermissionError("Removing edges is forbidden in Predecessors.")
    
    @property
    def start_node(self) -> "AbstractOperation":
        return super().start_node

    @property
    def end_node(self) -> "AbstractOperation":
        end_nodes = [node for node in self._graph.nodes if self._graph.out_degree(node) == 0]
        if len(end_nodes) != 1:
            raise ValueError("Predecessor Graph must have exactly one end node with out-degree 0")
        return end_nodes[0]
    
    def __contains__(self, node: "AbstractOperation") -> bool:
        return node in self._graph.nodes

class ExclusiveDescendants(BaseGraph):

    def __init__(self, original_graph: BaseGraph, subgraph: MultiDiGraph):
        super().__init__(subgraph)
        self.original_graph = original_graph
        self.new_nodes = set()
    
    def add_node(self, node: "AbstractOperation"):
        self.original_graph._add_node(node)
        self.new_nodes.add(node)
    
    def add_edge(self, edge: Edge, order: int=0):
        warnings.warn("Deprecated: Use partitions.add_edge instead.", DeprecationWarning)
        if edge.from_node in (self._graph.nodes | self.new_nodes) and edge.to_node in (self._graph.nodes | self.new_nodes):
            self.original_graph._add_edge(edge, order)
        else:
            raise ValueError(f"Both ends of the edge must be in the exclusive descendants graph. {edge.from_node} or {edge.to_node} is/are not in the original graph.")
        
    def add_dependency_edge(self, from_node: "AbstractOperation", to_node: "AbstractOperation"):
        self.original_graph._add_dependency_edge(from_node, to_node)

    def remove_node(self, node: "AbstractOperation"):
        raise PermissionError("Removing nodes is forbidden in ExclusiveDescendants. Use the function in the GraphPartitions class instead.")
    
    def remove_edge(self, edge: Edge):
        if edge in self.edges:
            self.original_graph._remove_edge(edge)
        else:
            raise ValueError(f"Edge {edge} is not in the exclusive descendants graph.")
        
    def successor_edges(self, node: "AbstractOperation") -> list[Edge]:
        successor_edges = self._graph.out_edges(node, data=True)
        return Edge.from_edge_view(successor_edges)
    
    @property
    def start_node(self) -> "AbstractOperation":
        start_nodes = [node for node in self._graph.nodes if self._graph.in_degree(node) == 0]
        if len(start_nodes) != 1:
            raise ValueError("DescendantGraph must have exactly one start node with in-degree 0")
        return start_nodes[0]
    
    @property
    def end_node(self) -> "AbstractOperation":
        raise PermissionError("End node is not defined for ExclusiveDescendants.")
    
    def __contains__(self, node: "AbstractOperation") -> bool:
        return node in self._graph.nodes or node in self.new_nodes

class Descendants(BaseGraph):

    def __init__(self, original_graph: BaseGraph, subgraph: MultiDiGraph):
        super().__init__(subgraph)
        self.original_graph = original_graph
    
    def add_node(self, node: "AbstractOperation"):
        raise PermissionError("Adding nodes is forbidden in Descendants.")
    
    def add_edge(self, edge: Edge):
        raise PermissionError("Adding edges is forbidden in Descendants.")
    
    def remove_node(self, node: "AbstractOperation"):
        raise PermissionError("Removing nodes is forbidden in Descendants.")
    
    def remove_edge(self, edge: Edge):
        raise PermissionError("Removing edges is forbidden in Descendants.")
    
    def _move_edge(self, current_edge: Edge, new_from_node: "AbstractOperation", new_from_node_key: NodeKey, order: int=0):
        self.original_graph._remove_edge(current_edge)
        self.original_graph._add_edge(Edge(new_from_node, current_edge.to_node, new_from_node_key, current_edge.to_node_key), order)
    
    def successor_edges(self, node: "AbstractOperation") -> list[Edge]:
        successor_edges = self._graph.out_edges(node, data=True)
        return Edge.from_edge_view(successor_edges)
    
    @property
    def start_node(self) -> "AbstractOperation":
        start_nodes = [node for node in self._graph.nodes if self._graph.in_degree(node) == 0]
        if len(start_nodes) != 1:
            raise ValueError("DescendantGraph must have exactly one start node with in-degree 0")
        return start_nodes[0]
    
    @property
    def end_node(self) -> "AbstractOperation":
        return super().end_node
    
    def __contains__(self, node: "AbstractOperation") -> bool:
        return node in self._graph.nodes


class GraphPartitions:
    predecessors: Predecessors
    descendants: Descendants
    exclusive_descendants: ExclusiveDescendants

    def __init__(self, predecessors: Predecessors, descendants: Descendants, exclusive_descendants: ExclusiveDescendants):
        self.predecessors = predecessors
        self.descendants = descendants
        self.exclusive_descendants = exclusive_descendants
        self.original_graph = predecessors.original_graph
    def move_edge(self, current_edge: Edge, new_from_node: "AbstractOperation", new_from_node_key: NodeKey):
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
        if not (edge.from_node in self.predecessors or edge.from_node in self.exclusive_descendants):
            raise ValueError(f"The from_node must be in the predecessors or exclusive descendants graph. {edge.from_node} is not.")
        if edge.to_node not in self.exclusive_descendants:
            raise ValueError(f"The to_node must be in the exclusive descendants graph. {edge.to_node} is not.")
        self.original_graph._add_edge(edge, order)
        if edge.from_node in self.predecessors:
            self.predecessors.original_graph._update_new_from_predecessor_edge_values(edge.from_node, edge.to_node, edge.from_node_key)

    def remove_node(self, node: "AbstractOperation"):
        if node in self.exclusive_descendants and node not in self.descendants:
            self.original_graph._remove_node(node)
        else:
            raise ValueError(f"Node {node} is not in the exclusive Descendants graph or points to non-exclusivedescendants")