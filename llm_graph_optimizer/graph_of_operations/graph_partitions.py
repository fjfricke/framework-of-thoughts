from typing import TYPE_CHECKING
from .base_graph import BaseGraph
from .types import NodeKey

if TYPE_CHECKING:
    from llm_graph_optimizer.operations.abstract_operation import AbstractOperation


class Predecessors(BaseGraph):
    @classmethod
    def from_graph_of_operations(cls, graph_of_operations: BaseGraph):
        return cls(graph_of_operations.graph)

    def add_node(self, node: "AbstractOperation"):
        raise PermissionError("Adding nodes is forbidden in Predecessors.")

    def add_edge(self, from_node: "AbstractOperation", to_node: "AbstractOperation", from_node_key: NodeKey, to_node_key: NodeKey):
        raise PermissionError("Adding edges is forbidden in Predecessors.")
    
    def remove_node(self, node: "AbstractOperation"):
        raise PermissionError("Removing nodes is forbidden in Predecessors.")
    
    def remove_edge(self, from_node: "AbstractOperation", to_node: "AbstractOperation", from_node_key: NodeKey, to_node_key: NodeKey):
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

class ExclusiveDescendants(BaseGraph):
    @classmethod
    def from_graph_of_operations(cls, graph_of_operations: BaseGraph):
        return cls(graph_of_operations.graph)
    
    def add_node(self, node: "AbstractOperation"):
        super()._add_node(node)
    
    def add_edge(self, from_node: "AbstractOperation", to_node: "AbstractOperation", from_node_key: NodeKey, to_node_key: NodeKey):
        super()._add_edge(from_node, to_node, from_node_key=from_node_key, to_node_key=to_node_key)

    def remove_node(self, node: "AbstractOperation"):
        raise PermissionError("Removing nodes is forbidden in ExclusiveDescendants. Use the function in the GraphPartitions class instead.")
    
    def remove_edge(self, from_node: "AbstractOperation", to_node: "AbstractOperation", from_node_key: NodeKey, to_node_key: NodeKey):
        super()._remove_edge(from_node, to_node, from_node_key=from_node_key, to_node_key=to_node_key)
    
    @property
    def start_node(self) -> "AbstractOperation":
        start_nodes = [node for node in self._graph.nodes if self._graph.in_degree(node) == 0]
        if len(start_nodes) != 1:
            raise ValueError("DescendantGraph must have exactly one start node with in-degree 0")
        return start_nodes[0]
    
    @property
    def end_node(self) -> "AbstractOperation":
        raise PermissionError("End node is not defined for ExclusiveDescendants.")

class Descendants(BaseGraph):

    @classmethod
    def from_graph_of_operations(cls, graph_of_operations: BaseGraph):
       return cls(graph_of_operations.graph)
    
    def add_node(self, node: "AbstractOperation"):
        raise PermissionError("Adding nodes is forbidden in Descendants.")
    
    def add_edge(self, from_node: "AbstractOperation", to_node: "AbstractOperation", from_node_key: NodeKey, to_node_key: NodeKey):
        raise PermissionError("Adding edges is forbidden in Descendants.")
    
    def remove_node(self, node: "AbstractOperation"):
        raise PermissionError("Removing nodes is forbidden in Descendants.")
    
    def remove_edge(self, from_node: "AbstractOperation", to_node: "AbstractOperation", from_node_key: NodeKey, to_node_key: NodeKey):
        raise PermissionError("Removing edges is forbidden in Descendants.")
    
    @property
    def start_node(self) -> "AbstractOperation":
        start_nodes = [node for node in self._graph.nodes if self._graph.in_degree(node) == 0]
        if len(start_nodes) != 1:
            raise ValueError("DescendantGraph must have exactly one start node with in-degree 0")
        return start_nodes[0]
    
    @property
    def end_node(self) -> "AbstractOperation":
        return super().end_node


class GraphPartitions:
    predecessors: Predecessors
    descendants: Descendants
    exclusive_descendants: ExclusiveDescendants

    def __init__(self, predecessors: Predecessors, descendants: Descendants, exclusive_descendants: ExclusiveDescendants):
        self.predecessors = predecessors
        self.descendants = descendants
        self.exclusive_descendants = exclusive_descendants

    def move_edge(self, previous_from_node: "AbstractOperation", new_from_node: "AbstractOperation", to_node: "AbstractOperation", previous_from_node_key: NodeKey, new_from_node_key: NodeKey, to_node_key: NodeKey):
        edge_data = self.descendants.get_edge_data(previous_from_node, to_node, previous_from_node_key, to_node_key)
        if edge_data is None:
            raise ValueError(f"Edge ({previous_from_node}, {to_node}, {previous_from_node_key}, {to_node_key}) does not exist in the graph.")
        if previous_from_node not in self.exclusive_descendants:
            raise ValueError(f"In order to move an edge, the previous from_node must be in the exclusive Descendants graph. {previous_from_node} is not.")
        if new_from_node not in self.exclusive_descendants:
            raise ValueError(f"In order to move an edge, the new from_node must be in the exclusive Descendants graph. {new_from_node} is not.")
        self.descendants.add_edge(new_from_node, to_node, from_node_key=new_from_node_key, to_node_key=to_node_key)
        self.descendants.remove_edge(previous_from_node, to_node, previous_from_node_key, to_node_key)

    def remove_node(self, node: "AbstractOperation"):
        if node in self.exclusive_descendants and node not in self.descendants:
            self.exclusive_descendants._remove_node(node)
        else:
            raise ValueError(f"Node {node} is not in the exclusive Descendants graph or points to non-exclusivedescendants.")