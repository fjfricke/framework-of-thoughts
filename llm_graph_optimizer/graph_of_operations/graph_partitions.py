from dataclasses import dataclass

@dataclass
class GraphPartitions:
    predecessors: "GraphOfOperations"
    descendants: "GraphOfOperations"
    exclusive_descendants: "GraphOfOperations"

    def __post_init__(self):
        from .graph_of_operations import GraphOfOperations  # Lazy import