from abc import ABC, abstractmethod
import copy
from typing import TYPE_CHECKING, get_origin
import networkx as nx
import matplotlib.pyplot as plt
import json

from pyvis.network import Network


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

    # def view_graph_ipysigma(self,
    #                         show_output_reasoning_states=False,
    #                         show_keys=False,
    #                         show_values=False,
    #                         output_name="debug_graph.html"):
    #     import networkx as nx
    #     from ipysigma import Sigma
    #     import panel as pn

    #     pn.extension('ipywidgets')

    #     G = self._graph

    #     # Create node labels
    #     node_labels = {
    #         node: f"{node.name}\n{node.node_state}" +
    #             (f"\n{node.output_reasoning_states}" if show_output_reasoning_states else "")
    #         for node in G.nodes
    #     }

    #     for node, label in node_labels.items():
    #         G.nodes[node]["label"] = label

    #     # Relabel to strings for Sigma
    #     mapping = {node: str(node) for node in G.nodes}
    #     G_copy = nx.relabel_nodes(G, mapping, copy=True)

    #     for orig, str_id in mapping.items():
    #         G_copy.nodes[str_id]["label"] = G.nodes[orig]["label"]

    #     if show_values:
    #         for u, v, data in G_copy.edges(data=True):
    #             data["label"] = str(data.get("value", ""))
    #     elif show_keys:
    #         for u, v, data in G_copy.edges(data=True):
    #             data["label"] = f"{data.get('from_node_key', '')} -> {data.get('to_node_key', '')}"

    #     viewer = Sigma(G_copy)
    #     pn.panel(viewer, sizing_mode="stretch_both", height=600).save(output_name, embed=True)
    #     print(f"✅ Graph saved to {output_name}")
    
    # def view_graph(self, save_path: str = None, show_output_reasoning_states: bool = False, show_keys: bool = False, show_values: bool = False, use_pyvis: bool = False, edge_length_power: float = 3, output_name: str = "debug_graph.html"):
    #     # Create labels for nodes based on their NodeState
    #     if show_output_reasoning_states:
    #         node_labels = {node: f"{node.name}\n{node.node_state}\n{node.output_reasoning_states}" for node in self._graph.nodes}
    #     else:
    #         node_labels = {node: f"{node.name}\n{node.node_state}" for node in self._graph.nodes}
        
    #     # Create labels for edges based on the 'value' field in edge data
    #     if show_values:
    #         edge_labels = {
    #             (u, v): data.get("value", "") for u, v, data in self._graph.edges(data=True)
    #         }
    #     elif show_keys:
    #         edge_labels = {
    #             (u, v): f"{data.get('from_node_key', '')} -> {data.get('to_node_key', '')}" for u, v, data in self._graph.edges(data=True)
    #         }

    #     # Topological order
    #     topo_order = list(nx.topological_sort(self._graph))
        
    #     # Initialize all distances
    #     dist = {node: float('-inf') for node in self._graph.nodes}
    #     dist[self.start_node] = 0
        
    #     for u in topo_order:
    #         for v in self._graph.successors(u):
    #             if dist[v] < dist[u] + 1:
    #                 dist[v] = dist[u] + 1

    #     # Apply the subset attribute (layer index)
    #     nx.set_node_attributes(self._graph, dist, "subset")

    #     # Use a layout that encourages edges to point to the right
    #     pos = nx.multipartite_layout(self._graph, subset_key="subset", align="vertical")

    #     # Set predefined positions for start and end nodes
    #     if self.start_node in self._graph.nodes:
    #         pos[self.start_node] = (-1, 0)  # Start node on the left
    #     if self.end_node in self._graph.nodes:
    #         pos[self.end_node] = (1, 0)  # End node on the right

    #     # Adjust positions for other nodes
    #     for node in pos:
    #         if node != self.start_node and node != self.end_node:
    #             pos[node] = (pos[node][0], pos[node][1])

    #     # Draw the graph with the node labels
    #     if use_pyvis:
    #         nt = Network(height='750px', width='100%', directed=True)
            
    #         # Relabel nodes
    #         mapping = {node: str(node) for node in self._graph.nodes}
    #         G_copy = nx.relabel_nodes(self._graph, mapping, copy=True)

    #         # Add level to each node in Pyvis based on dist / subset
    #         for relabeled_node, original_node in {v: k for k, v in mapping.items()}.items():
    #             level = self._graph.nodes[original_node].get("subset", 0)
    #             G_copy.nodes[relabeled_node]["level"] = level
            
    #         # Add the 'title' attribute to the relabeled graph
    #         for original_node, relabeled_node in mapping.items():
    #             G_copy.nodes[relabeled_node]['label'] = original_node.name
    #             G_copy.nodes[relabeled_node]['title'] = original_node.name  # Add title for hover functionality
            
    #         # Modify edge attributes in the relabeled graph
    #         for u, v, data in G_copy.edges(data=True):
    #             # Use the mapping to get the original nodes
    #             original_u = next(key for key, value in mapping.items() if value == u)
    #             original_v = next(key for key, value in mapping.items() if value == v)
                
    #             # Get subset values from the original graph
    #             subset_u = self._graph.nodes[original_u].get("subset", 0)
    #             subset_v = self._graph.nodes[original_v].get("subset", 0)
                
    #             # Calculate edge length based on subset difference
    #             edge_length = abs(subset_u - subset_v)
    #             data['length'] = edge_length ** edge_length_power # Add the length parameter to the edge data

    #             # Add keys or values to edge labels based on flags
    #             if show_keys:
    #                 data['label'] = f"{data.get('from_node_key', '')} -> {data.get('to_node_key', '')}"
    #             elif show_values:
    #                 data['label'] = str(data.get('value', ''))
            
    #         # Convert the modified graph to pyvis
    #         options = {
    #             "layout": {
    #                 "hierarchical": {
    #                     "enabled": True,
    #                     "levelSeparation": 150,
    #                     "nodeSpacing": 200,
    #                 }
    #             },
    #             "edges": {
    #                 "smooth": True
    #             },
    #             "physics": {
    #                 "enabled": True
    #             }
    #         }

    #         nt.set_options(json.dumps(options))

    #         nt.from_nx(G_copy)
    #         nt.write_html(output_name, open_browser=True)
    #     else:
    #         nx.draw(self._graph, pos, labels=node_labels, with_labels=True, node_size=300, font_size=5)
        
    #         # Draw the edge labels (only 'value') with adjusted positioning
    #         nx.draw_networkx_edge_labels(
    #             self._graph, pos, edge_labels=edge_labels, font_size=8, label_pos=0.5
    #         )
            
    #         # Save or show the graph
    #         if save_path:
    #             plt.savefig(save_path, bbox_inches="tight")  # Ensure labels fit within the saved image
    #         else:
    #             plt.show()
    #         plt.close()

    def view_graph_debug(self, show_keys: bool = False, show_values: bool = False, output_name: str = "debug_graph.html"):
        nt = Network(height='750px', width='100%', directed=True, cdn_resources="remote")

        # Relabel nodes to strings and remove all parameters
        mapping = {node: str(node) for node in self._graph.nodes}
        inverse_mapping = {v: k for k, v in mapping.items()}
        G_copy = nx.relabel_nodes(self._graph, mapping, copy=True)

        # Remove all attributes from nodes
        for node in G_copy.nodes:
            G_copy.nodes[node].clear()

        # Remove all attributes from edges
        for u, v, data in G_copy.edges(data=True):
            data.clear()

        # Set node label to show only node.name
        for node in G_copy.nodes:
            G_copy.nodes[node]['label'] = inverse_mapping[node].name

        nt.from_nx(G_copy)

        # Save the graph to an HTML file
        output_path = f"graphs/{output_name}"
        nt.save_graph(output_path)
        print(f"Graph saved to {output_path}")

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