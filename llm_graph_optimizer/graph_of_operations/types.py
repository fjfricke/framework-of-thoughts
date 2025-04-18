from dataclasses import dataclass
from typing import TYPE_CHECKING
from networkx.classes.multidigraph import OutMultiEdgeView

if TYPE_CHECKING:
    from llm_graph_optimizer.operations.abstract_operation import AbstractOperation

NodeKey = str | int

@dataclass
class Edge:
    from_node: "AbstractOperation"
    to_node: "AbstractOperation"
    from_node_key: NodeKey
    to_node_key: NodeKey

    @classmethod
    def from_edge_view(cls, edge_view: OutMultiEdgeView):
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
        return self.from_node == other.from_node and self.to_node == other.to_node and self.from_node_key == other.from_node_key and self.to_node_key == other.to_node_key


class StateNotSetType:
    def __repr__(self):
        return "<StateNotSet>"
StateNotSet = StateNotSetType()

class DynamicType:
    def __repr__(self):
        return "<Dynamic>"
Dynamic = DynamicType()

class ManyToOne(list):
    def __repr__(self):
        return "<ManyToOne>"
    
class ZeroOrManyToOne(ManyToOne):
    def __repr__(self):
        return "<ZeroOrManyToOne>"

ReasoningStateExecutionType = dict[NodeKey, any]
ReasoningStateType = dict[NodeKey, type] | DynamicType