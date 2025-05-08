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
        """
        Get a directed graph representation of the current graph. Represents the topology of the reasoning graph.

        :return: A directed graph (DiGraph) object.
        """
        return nx.DiGraph(self._graph)
    
    @property
    def snapshot(self) -> SnapshotGraph:
        """
        Create a snapshot of the graph with relabeled nodes and cleared attributes.
        This is used to store the graph state for debugging purposes.

        :return: A SnapshotGraph object representing the current graph state.
        """
        # Relabel nodes to strings and remove all parameters
        mapping = {node: node.uuid for node in self._graph.nodes}
        inverse_mapping = {v: k for k, v in mapping.items()}
        G_copy: nx.MultiDiGraph = nx.relabel_nodes(self._graph, mapping, copy=True)

        # Remove all attributes from nodes
        for node in G_copy.nodes:
            G_copy.nodes[node].clear()

        # add back the node name and state
        for node in G_copy.nodes:
            G_copy.nodes[node]['label'] = inverse_mapping[node].name
            G_copy.nodes[node]['state'] = inverse_mapping[node].node_state

        return SnapshotGraph(G_copy, self.start_node.uuid, self.end_node.uuid)

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

    def _add_dependency_edge(self, from_node: "AbstractOperation", to_node: "AbstractOperation"):
        self._graph.add_edge(
            from_node,
            to_node,
        )
    
    def _remove_node(self, node: "AbstractOperation"):
        self._graph.remove_node(node)
    
    def _remove_edge(self, edge: Edge):
        self._graph.remove_edge(edge.from_node, edge.to_node, key=(edge.from_node_key, edge.to_node_key))

    def _update_edge_values(self, from_node: "AbstractOperation", value: dict[NodeKey, any]):
        for edge in self._graph.edges(from_node, data=True):
            edge_data = edge[2]
            if "from_node_key" in edge_data:  # dependency edges do not hold data
                if edge_data["from_node_key"] in value:
                    edge_data["value"] = value[edge_data["from_node_key"]]
    
    def _update_new_from_predecessor_edge_values(self, from_node: "AbstractOperation", to_node: "AbstractOperation", from_node_key: NodeKey):
        for edge in self._graph.edges(from_node, data=True):
            if edge[1] == to_node:  
                edge_data = edge[2]
                if from_node_key in edge_data["from_node_key"]:
                    if edge_data["from_node_key"] in from_node.output_reasoning_states:
                        edge_data["value"] = from_node.output_reasoning_states[edge_data["from_node_key"]]

    def graph_table(self):
        """
        Generate a table representation of the graph's edges.

        :return: A pandas DataFrame containing edge details.
        """
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
        """
        Get data associated with a specific edge in the graph.

        :param edge: The edge to retrieve data for.
        :return: A dictionary containing edge data.
        """
        # Iterate over all edges between the nodes
        return self._graph.get_edge_data(edge.from_node, edge.to_node, key=(edge.from_node_key, edge.to_node_key))

    
    def successors(self, node: "AbstractOperation") -> list["AbstractOperation"]:
        """
        Get the successors of a given node in the graph.

        :param node: The node to find successors for.
        :return: A list of successor nodes.
        """
        return self._graph.successors(node)
    
    @property
    def edges(self) -> list[Edge]:
        """
        Get all edges in the graph.

        :return: A list of Edge objects representing the graph's edges.
        """
        return [Edge(from_node=edge[0], to_node=edge[1], from_node_key=edge[2]["from_node_key"], to_node_key=edge[2]["to_node_key"]) for edge in self._graph.edges(data=True)]
    
    def longest_path(self, weight: Callable[["AbstractOperation"], Number]) -> Number:
        """
        Calculate the longest path in the graph based on a weight function.

        :param weight: A callable that returns the weight of a node.
        :return: The length of the longest path in the graph.
        """
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
        """
        Abstract property for the start node.

        :return: The start node of the graph.
        """
        pass

    @property
    @abstractmethod
    def end_node(self) -> "AbstractOperation":
        """
        Abstract property for the end node.

        :return: The end node of the graph.
        """
        pass

    @abstractmethod
    def add_node(self, node: "AbstractOperation"):
        """
        Abstract method to add a node to the graph.

        :param node: The node to add.
        """
        pass

    @abstractmethod
    def add_edge(self, edge: Edge):
        """
        Abstract method to add an edge to the graph.

        :param edge: The edge to add.
        """
        pass
    
    @abstractmethod
    def remove_node(self, node: "AbstractOperation"):
        """
        Abstract method to remove a node from the graph.

        :param node: The node to remove.
        """
        pass
    
    @abstractmethod
    def remove_edge(self, edge: Edge):
        """
        Abstract method to remove an edge from the graph.

        :param edge: The edge to remove.
        """
        pass