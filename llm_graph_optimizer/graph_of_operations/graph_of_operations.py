from typing import TYPE_CHECKING
import networkx as nx
import matplotlib.pyplot as plt

from llm_graph_optimizer.operations.helpers.node_state import NodeState
from .graph_partitions import GraphPartitions
if TYPE_CHECKING:
    from llm_graph_optimizer.operations.abstract_operation import AbstractOperation


class GraphOfOperations:
    """
    Graph of operations.
    """

    def __init__(self, graph: nx.MultiDiGraph = None):
        self._graph = graph if graph else nx.MultiDiGraph()

    @property
    def digraph(self) -> nx.DiGraph:
        return nx.DiGraph(self._graph)
    
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
        self._graph.add_node(node)

    def add_edge(self, from_node: "AbstractOperation", to_node: "AbstractOperation", from_node_key: str | int, to_node_key: str | int):
        if from_node not in self._graph:
            raise ValueError(f"Node {from_node} not found in graph")
        if to_node not in self._graph:
            raise ValueError(f"Node {to_node} not found in graph")
        #check if from_node_key is a valid key
        if from_node_key not in from_node.output_types.keys():
            raise ValueError(f"Key {from_node_key} not found in {from_node.output_types}")
        #check if to_node_key is a valid key
        if to_node_key not in to_node.input_types.keys():
            raise ValueError(f"Key {to_node_key} not found in {to_node.input_types}")
        # Check if there is already an edge with the same to_node_key from any predecessor
        if any(
            edge_data.get("to_node_key") == to_node_key
            for predecessor in self._graph.predecessors(to_node)
            for edge_data in [self._graph.get_edge_data(predecessor, to_node)]
            if edge_data
        ):
            raise ValueError(f"One-to-many relationship violated: Multiple edges to {to_node} with to_node_key '{to_node_key}'")
        self._graph.add_edge(from_node, to_node, from_node_key=from_node_key, to_node_key=to_node_key)
    
    def update_edge_values(self, from_node: "AbstractOperation", value: dict[str | int, any]):
        for edge in self._graph.edges(from_node, data=True):
            edge_data = edge[2]
            if edge_data["from_node_key"] in value:
                edge_data["value"] = value[edge_data["from_node_key"]]

    def get_input_reasoning_states(self, node: "AbstractOperation") -> dict[str | int, any]:
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
    
    
    def view_graph(self, save_path: str = None, show_output_reasoning_states: bool = False, show_keys: bool = False, show_values: bool = False, use_pyvis: bool = False):
        # Create labels for nodes based on their NodeState
        if show_output_reasoning_states:
            node_labels = {node: f"{node.name}\n{node.node_state}\n{node.output_reasoning_states}" for node in self._graph.nodes}
        else:
            node_labels = {node: f"{node.name}\n{node.node_state}" for node in self._graph.nodes}
        
        # Create labels for edges based on the 'value' field in edge data
        if show_values:
            edge_labels = {
                (u, v): data.get("value", "") for u, v, data in self._graph.edges(data=True)
            }
        elif show_keys:
            edge_labels = {
                (u, v): f"{data.get('from_node_key', '')} -> {data.get('to_node_key', '')}" for u, v, data in self._graph.edges(data=True)
            }

        # Topological order
        topo_order = list(nx.topological_sort(self._graph))
        
        # Initialize all distances
        dist = {node: float('-inf') for node in self._graph.nodes}
        dist[self.start_node] = 0
        
        for u in topo_order:
            for v in self._graph.successors(u):
                if dist[v] < dist[u] + 1:
                    dist[v] = dist[u] + 1

        # Apply the subset attribute (layer index)
        nx.set_node_attributes(self._graph, dist, "subset")

        # Use a layout that encourages edges to point to the right
        pos = nx.multipartite_layout(self._graph, subset_key="subset", align="vertical")
        # pos = nx.bfs_layout(self._graph, start=self.start_node)  # Shell layout often aligns nodes circularly, but edges can point outward

        # # Draw the graph with the node labels
        # pos = nx.arf_layout(self._graph)  # Shell layout often aligns nodes circularly, but edges can point outward
        # Draw the graph with the node labels
        if use_pyvis:
            from pyvis.network import Network
            nt = Network(height='750px', width='100%', directed=True)
            
            # Relabel nodes
            mapping = {node: str(node) for node in self._graph.nodes}
            G_copy = nx.relabel_nodes(self._graph, mapping, copy=True)
            
            # Add the 'title' attribute to the relabeled graph
            for original_node, relabeled_node in mapping.items():
                G_copy.nodes[relabeled_node]['label'] = original_node.name
                G_copy.nodes[relabeled_node]['title'] = original_node.name  # Add title for hover functionality
            
            # Modify edge attributes in the relabeled graph
            for u, v, data in G_copy.edges(data=True):
                # Use the mapping to get the original nodes
                original_u = next(key for key, value in mapping.items() if value == u)
                original_v = next(key for key, value in mapping.items() if value == v)
                
                # Get subset values from the original graph
                subset_u = self._graph.nodes[original_u].get("subset", 0)
                subset_v = self._graph.nodes[original_v].get("subset", 0)
                
                # Calculate edge length based on subset difference
                edge_length = abs(subset_u - subset_v)
                data['length'] = edge_length ** 3 # Add the length parameter to the edge data
            
            # Convert the modified graph to pyvis
            nt.from_nx(G_copy)
            nt.show_buttons()
            nt.show("nx.html", notebook=False)
        else:
            nx.draw(self._graph, pos, labels=node_labels, with_labels=True, node_size=300, font_size=5)
        
            # Draw the edge labels (only 'value') with adjusted positioning
            nx.draw_networkx_edge_labels(
                self._graph, pos, edge_labels=edge_labels, font_size=8, label_pos=0.5
            )
            
            # Save or show the graph
            if save_path:
                plt.savefig(save_path, bbox_inches="tight")  # Ensure labels fit within the saved image
            else:
                plt.show()
            plt.close()
    
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
            predecessors=GraphOfOperations(self._graph.subgraph(predecessors_nodes).copy()),
            descendants=GraphOfOperations(self._graph.subgraph(descendants_nodes).copy()),
            exclusive_descendants=GraphOfOperations(self._graph.subgraph(exclusive_descendant_nodes).copy())
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



