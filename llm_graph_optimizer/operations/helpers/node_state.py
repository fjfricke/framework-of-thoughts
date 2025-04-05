from enum import Enum


class NodeState(Enum):
    """
    State of a node in the graph.
    """
    WAITING = "waiting"
    PROCESSABLE = "processable"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"
    ABORTED = "aborted"
    EARLY_STOPPED = "early_stopped"

    @property
    def is_finished(self) -> bool:
        return self in [NodeState.DONE, NodeState.FAILED, NodeState.ABORTED, NodeState.EARLY_STOPPED]

    