from abc import ABC, abstractmethod
from numbers import Number
from typing import TYPE_CHECKING, Callable, get_origin
import networkx as nx

from llm_graph_optimizer.graph_of_operations.snapshot_graph import SnapshotGraph


from .types import Dynamic, Edge, NodeKey, ManyToOne


if TYPE_CHECKING:
    from llm_graph_optimizer.operations.abstract_operation import AbstractOperation

class BaseGraph(ABC):
    """
    Basis for Graph of operations.
    """

    def __init__(self, graph: nx.MultiDiGraph = None):
        self._graph = graph if graph else nx.MultiDiGraph()

    @property
    def digraph(self) -> nx.DiGraph:
        return nx.DiGraph(self._graph)
    
    @property
    def snapshot(self) -> SnapshotGraph:
        # Relabel nodes to strings and remove all parameters
        mapping = {node: str(node) for node in self._graph.nodes}
        inverse_mapping = {v: k for k, v in mapping.items()}
        G_copy: nx.MultiDiGraph = nx.relabel_nodes(self._graph, mapping, copy=True)

        # Remove all attributes from nodes
        for node in G_copy.nodes:
            G_copy.nodes[node].clear()

        # add back the node name and state
        for node in G_copy.nodes:
            G_copy.nodes[node]['label'] = inverse_mapping[node].name
            G_copy.nodes[node]['state'] = inverse_mapping[node].node_state

        return SnapshotGraph(G_copy)

    def _add_node(self, node: "AbstractOperation"):
        self._graph.add_node(node)

    def _add_edge(self, edge: Edge, order: int):
        if edge.from_node not in self._graph:
            raise ValueError(f"Node {edge.from_node} not found in graph")
        if edge.to_node not in self._graph:
            raise ValueError(f"Node {edge.to_node} not found in graph")
        #check if from_node_key is a valid key
        if edge.from_node.output_types is not Dynamic and edge.from_node_key not in edge.from_node.output_types.keys():
            raise ValueError(f"Key {edge.from_node_key} not found in {edge.from_node.output_types}")
        #check if to_node_key is a valid key
        if edge.to_node.input_types is not Dynamic and edge.to_node_key not in edge.to_node.input_types.keys():
            raise ValueError(f"Key {edge.to_node_key} not found in {edge.to_node.input_types}")
        # Check if there is already an edge with the same to_node_key from any predecessor
        if any(
            edge_data.get("to_node_key") == edge.to_node_key and get_origin(edge.to_node.input_types[edge.to_node_key]) is not ManyToOne
            for predecessor in self._graph.predecessors(edge.to_node)
            for edge_key, edge_data in self._graph.get_edge_data(predecessor, edge.to_node).items()
            if edge_data
        ):
            raise ValueError(f"One-to-many relationship violated: Multiple edges to {edge.to_node} with to_node_key '{edge.to_node_key}'")
        self._graph.add_edge(
            edge.from_node,
            edge.to_node,
            key=(edge.from_node_key, edge.to_node_key),
            from_node_key=edge.from_node_key,
            to_node_key=edge.to_node_key,
            order=order
        )
    
    def _remove_node(self, node: "AbstractOperation"):
        self._graph.remove_node(node)
    
    def _remove_edge(self, edge: Edge):
        self._graph.remove_edge(edge.from_node, edge.to_node, key=(edge.from_node_key, edge.to_node_key))

    def _update_edge_values(self, from_node: "AbstractOperation", value: dict[NodeKey, any]):
        for edge in self._graph.edges(from_node, data=True):
            edge_data = edge[2]
            if edge_data["from_node_key"] in value:
                edge_data["value"] = value[edge_data["from_node_key"]]

    def graph_table(self):
        import pandas as pd

        edge_table = pd.DataFrame([
            {
                "from_node": edge[0],
                "to_node": edge[1],
                "from_node_key": edge[2].get("from_node_key"),
                "to_node_key": edge[2].get("to_node_key"),
                "value": edge[2].get("value"),
                "from_node_state": edge[0].node_state,
                "to_node_state": edge[1].node_state
            }
            for edge in self._graph.edges(data=True)
        ])
        return edge_table

    @property
    def _start_node(self) -> "AbstractOperation":
        if 'start_node' not in self._graph.graph:
            raise ValueError("Start node not found in graph")
        return self._graph.graph['start_node']

    @property
    def _end_node(self) -> "AbstractOperation":
        if 'end_node' not in self._graph.graph:
            raise ValueError("End node not found in graph")
        return self._graph.graph['end_node']
    
    def __contains__(self, node: "AbstractOperation") -> bool:
        """
        Check if a node exists in the graph.
        :param node: The node to check for existence.
        :return: True if the node exists in the graph, False otherwise.
        """
        return node in self._graph.nodes
    
    def get_edge_data(self, edge: Edge) -> dict:
        # Iterate over all edges between the nodes
        return self._graph.get_edge_data(edge.from_node, edge.to_node, key=(edge.from_node_key, edge.to_node_key))

    
    def successors(self, node: "AbstractOperation") -> list["AbstractOperation"]:
        return self._graph.successors(node)
    
    @property
    def edges(self) -> list[Edge]:
        return [Edge(from_node=edge[0], to_node=edge[1], from_node_key=edge[2]["from_node_key"], to_node_key=edge[2]["to_node_key"]) for edge in self._graph.edges(data=True)]
    
    def longest_path(self, weight: Callable[["AbstractOperation"], Number]) -> Number:
        path_digraph = self.digraph.copy()
        #if cycles in the graph raise an error
        if not nx.is_directed_acyclic_graph(path_digraph):
            raise NotImplementedError("Graph has cycles. Unrolling not implemented yet for calculating the longest path.")
        # the weight of each edge is the weight of the to_node (adding start_node weight to the edges coming from there)
        for edge in path_digraph.edges(data=True):
            edge[2]["weight"] = weight(edge[1])
            if edge[0] == self._start_node:
                edge[2]["weight"] += weight(edge[0])

        return nx.dag_longest_path_length(path_digraph, default_weight=0)
    
    @property
    @abstractmethod
    def start_node(self) -> "AbstractOperation":
        """Abstract property for the start node."""
        pass

    @property
    @abstractmethod
    def end_node(self) -> "AbstractOperation":
        """Abstract property for the end node."""
        pass

    @abstractmethod
    def add_node(self, node: "AbstractOperation"):
        pass

    @abstractmethod
    def add_edge(self, edge: Edge):
        pass
    
    @abstractmethod
    def remove_node(self, node: "AbstractOperation"):
        pass
    
    @abstractmethod
    def remove_edge(self, edge: Edge):
        pass