from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar
from networkx.classes.multidigraph import OutMultiEdgeView

if TYPE_CHECKING:
    from llm_graph_optimizer.operations.abstract_operation import AbstractOperation

NodeKey = str | int
"""
Type alias for a solution key, which can be either a string or an integer.
"""

@dataclass
class Edge:
    """
    Represents an edge in a graph, connecting two nodes with specific solution keys.
    """
    from_node: "AbstractOperation"
    to_node: "AbstractOperation"
    from_node_key: NodeKey
    to_node_key: NodeKey

    @classmethod
    def from_edge_view(cls, edge_view: OutMultiEdgeView):
        """
        Create a list of Edge instances from a networkx OutMultiEdgeView.

        :param edge_view: The OutMultiEdgeView to convert.
        :return: A list of Edge instances.
        """
        return [
            cls(
                from_node=from_node,
                to_node=to_node,
                from_node_key=edge_data.get("from_node_key"),
                to_node_key=edge_data.get("to_node_key")
            )
            for from_node, to_node, edge_data in edge_view
        ]

    def __eq__(self, other):
        """
        Check equality between two Edge instances, meaning that they have the same from_node, to_node, from_node_key, and to_node_key.

        :param other: The other Edge instance to compare.
        :return: True if the edges are equal, False otherwise.
        """
        return self.from_node == other.from_node and self.to_node == other.to_node and self.from_node_key == other.from_node_key and self.to_node_key == other.to_node_key


class StateNotSetType:
    """
    Represents a state that has not yet been set. All operations are initialized with this state.
    """
    def __repr__(self):
        return "<StateNotSet>"
StateNotSet = StateNotSetType()


class StateSetFailedType:
    """
    Represents a state that has failed to be set. Set when operations fail.
    """
    def __repr__(self):
        return "<StateSetFailed>"
StateSetFailed = StateSetFailedType()

class DynamicType:
    """
    Represents a dynamic type, used for flexible type definitions. Leads to skipping type checking on graph initalization. Use with caution!
    """
    def __repr__(self):
        return "<Dynamic>"
Dynamic = DynamicType()

T = TypeVar("T")
class ManyToOne(list, Generic[T]):
    """
    Represents a one-to-many relationship, where a single key maps to multiple values. The solution key then contains a list of values from all in-edges. Use the order parameter when connecting the nodes to ensure a predefined order.
    """
    def __repr__(self):
        return "<ManyToOne>"

ReasoningState = dict[NodeKey, any]
"""
Type alias for a reasoning state, represented as a dictionary mapping NodeKey to any value.
"""

ReasoningStateType = dict[NodeKey, type] | DynamicType
"""
Type alias for a reasoning state type, which can be a dictionary mapping NodeKey to a type or a DynamicType.
"""