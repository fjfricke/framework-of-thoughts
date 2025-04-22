import copy
import pickle
import tempfile
from networkx import DiGraph, MultiDiGraph
from pyvis.network import Network
from IPython.display import IFrame
import webbrowser
import json
import base64

from llm_graph_optimizer.operations.helpers.node_state import NodeState


class SnapshotGraphs():
    def __init__(self):
        self.graphs: list[SnapshotGraph] = []

    def add_snapshot(self, snapshot: "SnapshotGraph"):
        self.graphs.append(snapshot)

    def save(self, path: str):
        pickle.dump(self.graphs, open(path, "wb"))

    def load(self, path: str):
        self.graphs = pickle.load(open(path, "rb"))


class SnapshotGraph():
    def __init__(self, graph: MultiDiGraph):
        """
        Do not use this constructor directly. Use <BaseGraph>.create_snapshot or <GraphOfOperations>.create_snapshot instead.
        """
        self._graph = graph

    def save(self, path: str):
        pickle.dump(self._graph, open(path, "wb"))

    @classmethod
    def load(cls, path: str):
        return cls(pickle.load(open(path, "rb")))

    def visualize(self, show_multiedges: bool = True, show_keys: bool = False, show_values: bool = False, show_state: bool = False, notebook: bool = False):
        return self._view_or_save_visualization(show_multiedges, show_keys, show_values, show_state, notebook=notebook)
    
    def save_visualization(self, show_multiedges: bool = True, show_keys: bool = False, show_values: bool = False, show_state: bool = False, save_path: str = None):
        return self._view_or_save_visualization(show_multiedges, show_keys, show_values, show_state, save_path=save_path)

    def _view_or_save_visualization(self, show_multiedges: bool = True, show_keys: bool = False, show_values: bool = False, show_state: bool = False, notebook: bool = None, save_path: str = None) -> IFrame | None:
        nt = self._create_view(show_multiedges, show_keys, show_values, show_state, notebook)

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

    def _create_view(self, show_multiedges: bool = True, show_keys: bool = False, show_values: bool = False, show_state: bool = False, notebook: bool = False) -> Network:
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
                edge_data['title'] = f"{original_edge_data['from_node_key']} -> {original_edge_data['to_node_key']}"
            elif show_values:
                edge_data['title'] = f"{original_edge_data['from_node_key']} -> {original_edge_data['to_node_key']}: {original_edge_data.get('value', 'N/A')}"

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