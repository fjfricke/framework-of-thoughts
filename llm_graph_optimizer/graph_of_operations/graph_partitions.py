from dataclasses import dataclass

from .graph_of_operations import GraphOfOperations


@dataclass
class GraphPartitions:
    predecessors: GraphOfOperations
    descendants: GraphOfOperations
    exclusive_descendants: GraphOfOperations