import copy
import logging
from numbers import Number
import pickle
import tempfile
from typing import Callable
from networkx import DiGraph, MultiDiGraph
from pyvis.network import Network
from IPython.display import IFrame
import webbrowser
import json
import base64
import networkx as nx
from llm_graph_optimizer.operations.helpers.node_state import NodeState


class SnapshotGraphs():
    """
    A collection of snapshot graphs representing the state of a graph over time.
    """

    def __init__(self):
        """
        Initialize an empty collection of snapshot graphs.
        """
        self.graphs: list[SnapshotGraph] = []

    def add_snapshot(self, snapshot: "SnapshotGraph"):
        """
        Add a snapshot graph to the collection.

        :param snapshot: The snapshot graph to add.
        """
        self.graphs.append(snapshot)

    def save(self, path: str):
        """
        Save the collection of snapshot graphs to a pickle file.

        :param path: The file path to save the snapshots.
        """
        pickle.dump(self.graphs, open(path, "wb"))

    def load(self, path: str):
        """
        Load a collection of snapshot graphs from a picklefile.

        :param path: The file path to load the snapshots from.
        """
        self.graphs = pickle.load(open(path, "rb"))


class SnapshotGraph():
    """
    Represents a snapshot of a graph at a specific point in time.
    """

    def __init__(self, graph: MultiDiGraph, start_node: str, end_node: str):
        """
        Do not use this constructor directly. Use <BaseGraph>.create_snapshot or <GraphOfOperations>.create_snapshot instead.
        """
        self._graph = graph
        self._start_node = start_node
        self._end_node = end_node

    def save(self, path: str):
        """
        Save the snapshot graph to a pickle file.

        :param path: The file path to save the snapshot.
        """
        pickle.dump(self._graph, open(path, "wb"))

    @classmethod
    def load(cls, path: str):
        """
        Load a snapshot graph from a pickle file.

        :param path: The file path to load the snapshot from.
        :return: The loaded SnapshotGraph instance.
        """
        return cls(pickle.load(open(path, "rb")))
    
    @property
    def digraph(self) -> nx.DiGraph:
        """
        Get a directed graph representation of the snapshot. This represents the topology of the reasoning graph.

        :return: A directed graph (DiGraph) object.
        """
        return nx.DiGraph(self._graph)
    
    def longest_path(self, weight: Callable[[str], Number]) -> Number:
        """
        Calculate the longest path in the snapshot graph based on a weight function.

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

    def visualize(self, show_multiedges: bool = False, show_keys: bool = False, show_values: bool = False, show_state: bool = False, notebook: bool = False, show_keys_on_arrows: bool = False):
        """
        Visualize the snapshot graph.

        :param show_multiedges: Whether to show multiple edges between nodes.
        :param show_keys: Whether to show keys on edges.
        :param show_values: Whether to show values on edges.
        :param show_state: Whether to show node states, color-coded with green for done, yellow for processing, orange for processable, blue for waiting, and red for failed.
        :param notebook: Whether to display the visualization in a notebook.
        :param show_keys_on_arrows: Whether to show keys on arrows. Not recommended.
        :return: An IFrame object if displayed in a notebook, otherwise None.
        """
        if show_multiedges:
            logging.warning("show_multiedges does not work well with hierarchical layout. I recommend not to use it.")
        return self._view_or_save_visualization(show_multiedges, show_keys, show_values, show_state, notebook=notebook, show_keys_on_arrows=show_keys_on_arrows)
    
    def save_visualization(self, show_multiedges: bool = True, show_keys: bool = False, show_values: bool = False, show_state: bool = False, save_path: str = None):
        """
        Save the visualization of the snapshot graph to a file.

        :param show_multiedges: Whether to show multiple edges between nodes.
        :param show_keys: Whether to show keys on edges.
        :param show_values: Whether to show values on edges.
        :param show_state: Whether to show node states. Color-coded with green for done, yellow for processing, orange for processable, blue for waiting, and red for failed.
        :param save_path: The file path to save the visualization.
        """
        return self._view_or_save_visualization(show_multiedges, show_keys, show_values, show_state, save_path=save_path)

    def _view_or_save_visualization(self, show_multiedges: bool = True, show_keys: bool = False, show_values: bool = False, show_state: bool = False, notebook: bool = None, save_path: str = None, show_keys_on_arrows: bool = False) -> IFrame | None:
        """
        Internal method to view or save the visualization of the snapshot graph.

        :param show_multiedges: Whether to show multiple edges between nodes.
        :param show_keys: Whether to show keys on edges.
        :param show_values: Whether to show values on edges.
        :param show_state: Whether to show node states.
        :param notebook: Whether to display the visualization in a notebook.
        :param save_path: The file path to save the visualization.
        :param show_keys_on_arrows: Whether to show keys on arrows.
        :return: An IFrame object if displayed in a notebook, otherwise None.
        """
        nt = self._create_view(show_multiedges, show_keys, show_values, show_state, notebook, show_keys_on_arrows)

        if notebook:
            html_content = nt.generate_html(notebook=False)  # despite it being a notebook, this has to be false
            html_base64 = base64.b64encode(html_content.encode('utf-8')).decode('utf-8')
            html_data_url = f"data:text/html;base64,{html_base64}"
            return IFrame(src=html_data_url, width=nt.width, height=nt.height)
        else:
            if save_path:
                nt.show(save_path, notebook=False)
            else:
                with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as temp_file:
                    temp_path = temp_file.name
                    nt.show(temp_path, notebook=False)
                    webbrowser.open(f"file://{temp_path}")

    def _create_view(self, show_multiedges: bool = True, show_keys: bool = False, show_values: bool = False, show_state: bool = False, notebook: bool = False, show_keys_on_arrows: bool = False) -> Network:
        """
        Internal method to create a visualization view of the snapshot graph.

        :param show_multiedges: Whether to show multiple edges between nodes.
        :param show_keys: Whether to show keys on edges.
        :param show_values: Whether to show values on edges.
        :param show_state: Whether to show node states.
        :param notebook: Whether to display the visualization in a notebook.
        :param show_keys_on_arrows: Whether to show keys on arrows.
        :return: A Network object representing the visualization.
        """
        nt = Network(height='600px', width='100%', directed=True, cdn_resources="remote" if not notebook else "in_line", filter_menu=True, notebook=notebook)
        graph = copy.deepcopy(self._graph)

        # Remove all attributes from edges in the copied graph and set the `title` attribute
        for edge in graph.edges(data=True, keys=True):
            # Access the original edge data from `self._graph`
            original_edge_data = self._graph.get_edge_data(edge[0], edge[1], edge[2])
            
            # Clear all attributes in the copied graph
            edge_data = edge[3]
            edge_data.clear()

            # Set the `title` attribute based on the original edge data
            if show_keys and not show_values:
                if "from_node_key" in original_edge_data and "to_node_key" in original_edge_data:
                    edge_data['title'] = f"{original_edge_data['from_node_key']} -> {original_edge_data['to_node_key']}"
                else:
                    edge_data['title'] = "dependency edge"
                    edge_data['color'] = 'grey'  # Set dependency edge color to grey
            elif show_values:
                if "from_node_key" in original_edge_data and "to_node_key" in original_edge_data:
                    edge_data['title'] = f"{original_edge_data['from_node_key']} -> {original_edge_data['to_node_key']}: {original_edge_data.get('value', 'N/A')}"
                else:
                    edge_data['title'] = "dependency edge"
                    edge_data['color'] = 'grey'  # Set dependency edge color to grey
            if show_keys_on_arrows:
                keys_to_add = f"{original_edge_data['from_node_key']} -> {original_edge_data['to_node_key']}"
                if "label" not in edge_data:
                    edge_data['label'] = keys_to_add
                else:
                    edge_data['label'] = f"{edge_data['label']}, {keys_to_add}"

        # add color to the nodes depending on the state
        if show_state:
            for node in graph.nodes:
                if graph.nodes[node]['state'] == NodeState.DONE:
                    graph.nodes[node]['color'] = 'green'
                elif graph.nodes[node]['state'] == NodeState.PROCESSING:
                    graph.nodes[node]['color'] = 'yellow'
                elif graph.nodes[node]['state'] == NodeState.PROCESSABLE:
                    graph.nodes[node]['color'] = 'orange'
                elif graph.nodes[node]['state'] == NodeState.WAITING:
                    graph.nodes[node]['color'] = 'blue'
                else:
                    graph.nodes[node]['color'] = 'red'

        for node in graph.nodes:
            graph.nodes[node]['state'] = str(graph.nodes[node]['state'])

        # Convert UUIDs to strings for visualization
        graph = nx.relabel_nodes(graph, {node: str(node) for node in graph.nodes})

        # Handle multiedges or single edges
        if show_multiedges:
            nt.from_nx(graph)
        else:
            # Create a DiGraph and concatenate edge data
            digraph = DiGraph(graph)
            for u, v, data in digraph.edges(data=True):
                data['title'] = "\n".join([data.get('title', '') for data in graph.get_edge_data(u, v).values()])

            nt.from_nx(digraph)

        # Configure physics to make edges less springy and allow more space
        physics_options = {
            "layout": {
                "hierarchical": {
                    "enabled": True,
                    "levelSeparation": 100,
                    "nodeSpacing": 200,
                    "treeSpacing": 220,
                    "direction": "LR",
                    "sortMethod": "directed"
                }
            },
            "physics": {
                "hierarchicalRepulsion": {
                    "centralGravity": 0,
                    "springConstant": 0,
                    "nodeDistance": 75,
                    "damping": 0.17,
                    "avoidOverlap": 0
                },
                "minVelocity": 0.75,
                "solver": "hierarchicalRepulsion"
            }
        }
        nt.set_options(json.dumps(physics_options))

        return nt

    def save_graphml(self, path: str, include_values: bool = False):
        """
        Save a sanitized GraphML representation of the snapshot graph.

        This converts all node IDs to strings, coerces node attributes to
        GraphML-supported primitives, removes/strings non-serializable values,
        and sets graph-level attributes to strings.

        :param path: Output file path for the GraphML file.
        :param include_values: If True, includes edge "value" as string (repr, truncated).
        """
        # Build a fresh MultiDiGraph with safe types only
        safe_graph = nx.MultiDiGraph()

        # Graph-level attributes: store start/end as strings only
        if self._start_node is not None:
            safe_graph.graph["start_node"] = str(self._start_node)
        if self._end_node is not None:
            safe_graph.graph["end_node"] = str(self._end_node)

        # Nodes: coerce IDs to strings and attributes to primitives
        for node_id, attrs in self._graph.nodes(data=True):
            safe_node_id = str(node_id)
            safe_attrs: dict[str, any] = {}
            label = attrs.get("label")
            if label is not None:
                safe_attrs["label"] = str(label)
            state = attrs.get("state")
            if state is not None:
                # NodeState has __str__, so this yields its value
                safe_attrs["state"] = str(state)
            safe_graph.add_node(safe_node_id, **safe_attrs)

        # Edges: keep only primitive/safe attributes; optionally stringify "value"
        def _to_primitive(value: any) -> str | int | float | bool | None:
            if value is None:
                return None
            if isinstance(value, (str, int, float, bool)):
                return value
            # Fallback to string
            return str(value)

        for u, v, key, data in self._graph.edges(keys=True, data=True):
            safe_u, safe_v = str(u), str(v)
            safe_data: dict[str, any] = {}
            for k in ("from_node_key", "to_node_key", "order", "idx"):
                if k in data:
                    coerced = _to_primitive(data[k])
                    if coerced is not None:
                        safe_data[k] = coerced
            if include_values and "value" in data:
                # Stringify and truncate overly long values to keep files reasonable
                try:
                    val_str = str(data["value"])
                except Exception:
                    val_str = "<unserializable>"
                if len(val_str) > 2000:
                    val_str = val_str[:2000] + "…"
                safe_data["value"] = val_str
            safe_graph.add_edge(safe_u, safe_v, key=key, **safe_data)

        # Finally, write GraphML
        nx.write_graphml(safe_graph, path)