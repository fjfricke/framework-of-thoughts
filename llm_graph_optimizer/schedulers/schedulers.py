import networkx as nx
from llm_graph_optimizer.graph_of_operations.graph_of_operations import GraphOfOperations
from llm_graph_optimizer.operations.abstract_operation import AbstractOperation
from llm_graph_optimizer.operations.helpers.node_state import NodeState


class Scheduler:
    """
    Scheduler class with different scheduling strategies for GraphOfOperations.
    """

    @staticmethod
    def BFS(graph_of_operations: GraphOfOperations) -> list[AbstractOperation]:
        """
        Schedule operations using Breadth-First Search (BFS).
        :param graph_of_operations: The graph of operations.
        :return: List of operations in BFS order.
        """
        bfs_order = list(nx.bfs_tree(graph_of_operations._graph, source=graph_of_operations.start_node))
        # remove all nodes that are not processable
        bfs_order = [node for node in bfs_order if node.node_state == NodeState.PROCESSABLE]
        return bfs_order

    @staticmethod
    def DFS(graph_of_operations: GraphOfOperations) -> list[AbstractOperation]:
        """
        Schedule operations using Depth-First Search (DFS).
        :param graph_of_operations: The graph of operations.
        :return: List of operations in DFS order.
        """
        dfs_order = list(nx.dfs_tree(graph_of_operations._graph, source=graph_of_operations.start_node))
        # remove all nodes that are not processable
        dfs_order = [node for node in dfs_order if node.node_state == NodeState.PROCESSABLE]
        return dfs_order