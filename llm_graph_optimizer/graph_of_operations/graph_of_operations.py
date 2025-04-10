import copy
import networkx as nx
import matplotlib.pyplot as plt

from llm_graph_optimizer.operations.helpers.node_state import NodeState

from .graph_partitions import GraphPartitions
# from llm_graph_optimizer.operations.abstract_operation import AbstractOperation


class GraphOfOperations:
    """
    Graph of operations.
    """

    def __init__(self, graph: nx.MultiDiGraph = None):
        self._graph = graph if graph else nx.MultiDiGraph()

    # def deepcopy(self) -> "GraphOfOperations":
    #     copied_graph = copy.deepcopy(self._graph)

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

    def add_node(self, node: "AbstractOperation", start_node: bool = False, end_node: bool = False):
        self._graph.add_node(node)
        if start_node:
            self.start_node = node
        if end_node:
            self.end_node = node

    def add_edge(self, from_node: "AbstractOperation", to_node: "AbstractOperation", from_node_key: str, to_node_key: str):
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
    
    def update_edge_values(self, from_node: "AbstractOperation", value: dict[str, any]):
        for edge in self._graph.edges(from_node, data=True):
            edge_data = edge[2]
            if edge_data["from_node_key"] in value:
                edge_data["value"] = value[edge_data["from_node_key"]]

    def get_input_reasoning_states(self, node: "AbstractOperation") -> dict[str, any]:
        if node == self.start_node:
            return node.input_reasoning_states
        predecessors = self._graph.predecessors(node)
        input_reasoning_states = {}
        for predecessor in predecessors:
            if not predecessor.node_state.is_finished:
                raise ValueError(f"Predecessor {predecessor} is not finished")
            edge_data = self._graph.get_edge_data(predecessor, node)
            to_node_key = edge_data[0]["to_node_key"]
            if to_node_key:
                value = edge_data[0].get("value")
                input_reasoning_states[to_node_key] = value
        return input_reasoning_states
    
    def view_graph(self, save_path: str = None, show_output_reasoning_states: bool = False):
        # Create labels for nodes based on their NodeState
        if show_output_reasoning_states:
            node_labels = {node: f"{node.name}\n{node.node_state}\n{node.output_reasoning_states}" for node in self._graph.nodes}
        else:
            node_labels = {node: f"{node.name}\n{node.node_state}" for node in self._graph.nodes}
        
        # Create labels for edges based on the 'value' field in edge data
        edge_labels = {
            (u, v): data.get("value", "") for u, v, data in self._graph.edges(data=True)
        }
        
        # Use a layout that encourages edges to point to the right
        pos = nx.shell_layout(self._graph)  # Shell layout often aligns nodes circularly, but edges can point outward
        
        # Draw the graph with the node labels
        nx.draw(self._graph, pos, labels=node_labels, with_labels=True, node_size=3000, font_size=10)
        
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
    
    @property
    def partitions(self) -> GraphPartitions:
        return None #TODO: fix
        all_nodes = set(self._graph.nodes)

        # Compute predecessors and descendants
        predecessors_nodes = nx.ancestors(self._graph, self._graph.nodes)
        descendants_nodes = nx.descendants(self._graph, self._graph.nodes)

        # Compute exclusive descendants
        exclusive_descendant_nodes = {}
        for node in all_nodes:
            # Get descendants of the current node
            node_descendants = nx.descendants(self._graph, node)

            # Get non-descendant nodes
            non_descendant_nodes = all_nodes - node_descendants - {node}

            # Get descendants of non-descendant nodes
            non_descendant_descendants = set()
            for non_descendant in non_descendant_nodes:
                non_descendant_descendants.update(nx.descendants(self._graph, non_descendant))

            # Filter exclusive descendants
            exclusive_descendant_nodes[node] = node_descendants - non_descendant_descendants

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



