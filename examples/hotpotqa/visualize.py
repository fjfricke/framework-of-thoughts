import os
from pathlib import Path
from llm_graph_optimizer.graph_of_operations.snapshot_graph import SnapshotGraph


if __name__ == "__main__":
    snapshot_graph = SnapshotGraph.load(Path(os.getcwd()) / "examples" / "hotpotqa" / "output" / "probtree_debug.pkl")
    snapshot_graph.view(show_state=True, show_keys=True, show_values=True, show_multiedges=False)