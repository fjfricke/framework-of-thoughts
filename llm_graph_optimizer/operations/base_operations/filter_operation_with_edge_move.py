from enum import Enum
from typing import Callable, OrderedDict, get_origin

from llm_graph_optimizer.graph_of_operations.graph_partitions import GraphPartitions
from llm_graph_optimizer.graph_of_operations.types import Dynamic, Edge, ManyToOne, ReasoningState, ReasoningStateType
from llm_graph_optimizer.operations.abstract_operation import AbstractOperation
from llm_graph_optimizer.measurement.measurement import Measurement

class Correspondence(Enum):
    """
    Whether the end nodes are one or num_branches
    """
    MANY_TO_ONE = "many_to_one"
    ONE_TO_ONE = "one_to_one"

def _map_indices_to_edges(edges, indices):
    # Group edges by their order
    order_to_edges = {}
    for edge in edges:
        order = edge.order
        if order not in order_to_edges:
            order_to_edges[order] = []
        order_to_edges[order].append(edge)
    
    # Sort orders to ensure consistent mapping
    sorted_orders = sorted(order_to_edges.keys())
    
    # Map indices to edges based on sorted orders
    index_to_edges = {}
    for index in indices:
        if index < len(sorted_orders):
            order = sorted_orders[index]
            index_to_edges[index] = order_to_edges[order]
    
    return index_to_edges

def _map_order_to_edges(edges: list[Edge]):
    # Create an ordered dictionary to map order to edges
    order_to_edges = OrderedDict()
    for edge in edges:
        order = edge.order
        if order not in order_to_edges:
            order_to_edges[order] = {}
        # Map to_node to edge
        order_to_edges[order][edge.from_node_key] = edge
    
    # Sort the dictionary by order
    return OrderedDict(sorted(order_to_edges.items()))

class FilterOperationWithEdgeMove(AbstractOperation):
    """
    Operation to filter reasoning states based on a custom filter function.

    This operation applies a filter function to the input reasoning states
    and moves the start node of the edges to the top reasoning states. The filter function
    is expected to return the indices of the top reasoning states based
    on a specified criterion (e.g., F1 scores).

    Attributes:
        input_types (ReasoningStateType): Expected types for input reasoning states.
        filter_function (Callable[..., list[int]]): Function to filter reasoning states and return indices of the top states.
        params (dict): Parameters for the operation.
        name (str): Name of the operation.
    """

    def __init__(self, input_types: ReasoningStateType, filter_function: Callable[..., list[int]], correspondence: Correspondence = Correspondence.ONE_TO_ONE, params: dict = None, name: str = None):
        """
        Initialize a FilterOperationWithEdgeMove instance.

        Args:
            input_types (ReasoningStateType): Expected types for input reasoning states. All must be of origin type ManyToOne.
            filter_function (Callable[..., list[int]]): Function to filter reasoning states and return indices of the top states.
            params (dict, optional): Parameters for the operation. Defaults to None.
            name (str, optional): Name of the operation. Defaults to the class name.

        Raises:
            TypeError: If any input type is not of origin type ManyToOne.

        Notes:
            The filter function should return a list of indices corresponding
            to the top reasoning states based on the specified criterion.
        """
        if not all(get_origin(value) is ManyToOne for value in input_types.values()):
            raise TypeError("All input types must have keys of origin type ManyToOne")
        self.filter_function = filter_function
        super().__init__(input_types, Dynamic, params, name)
        self.correspondence = correspondence
    async def _execute(self, partitions: GraphPartitions, input_reasoning_states: ReasoningState) -> tuple[ReasoningState, Measurement | None]:
        indices = self.filter_function(**input_reasoning_states)
        # get predecessor edges and filter by the ones in the filtered indices
        predecessor_edges = partitions.predecessors.predecessor_edges(self)
        index_to_edges = _map_indices_to_edges(predecessor_edges, indices)

        if self.correspondence == Correspondence.ONE_TO_ONE:
            
            # get out edges sorted by their order (if set)
            out_edges = _map_order_to_edges(partitions.descendants.successor_edges(self))

            # move the edges
            out_nodes = set()
            for order, index in zip(out_edges.keys(), indices):
                for edge in index_to_edges[index]:
                    # does any out edge have the same to node key?
                    out_edge = out_edges[order][edge.from_node_key]
                    partitions.move_edge_start_node(out_edge, edge.from_node, edge.from_node_key)
                    out_nodes.add(out_edge.to_node)

            # add dependency edges to the out nodes to mark filter has to run before the out nodes (only for visualisation)
            for out_node in out_nodes:
                partitions.exclusive_descendants.add_dependency_edge(self, out_node)
        elif self.correspondence == Correspondence.MANY_TO_ONE:
            out_edges = partitions.descendants.successor_edges(self)
            out_edge_to_from_node_key = {edge: edge.from_node_key for edge in out_edges}
            out_nodes = set(out_edge.to_node for out_edge in out_edges)
            # if len(out_nodes) != 1:
            #     raise ValueError("Many to one correspondence expected exactly one out node")
            for edge, from_node_key in out_edge_to_from_node_key.items():
                indexed_edges_where_from_node_key_matches = sorted([edge for edges in index_to_edges.values() for edge in edges if edge.from_node_key == from_node_key], key=lambda e: e.order)
                orders = [edge.order for edge in indexed_edges_where_from_node_key_matches]
                partitions.move_start_node_and_duplicate_edges(current_edge=edge, new_from_nodes=[edge.from_node for edge in indexed_edges_where_from_node_key_matches], new_from_node_keys=[from_node_key] * len(indexed_edges_where_from_node_key_matches), orders=orders)
            for out_node in out_nodes:
                partitions.exclusive_descendants.add_dependency_edge(self, out_node)

        return {}, None