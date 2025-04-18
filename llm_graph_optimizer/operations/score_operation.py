from typing import Callable

from llm_graph_optimizer.graph_of_operations.graph_partitions import GraphPartitions
from llm_graph_optimizer.graph_of_operations.types import ReasoningStateType, ReasoningStateExecutionType
from llm_graph_optimizer.operations.helpers.exceptions import OperationFailed
from .abstract_operation import AbstractOperation

class ScoreOperation(AbstractOperation):
    """
    Score operation.
    The output_reasoning_state has the signature: {
        "score": float
    }
    """
    
    def __init__(self, input_types: ReasoningStateType, output_type: type, scoring_function: Callable[..., any], params: dict = None, name: str = None):
        output_types = {"score": output_type}
        super().__init__(input_types, output_types, params, name)
        self.scoring_function = scoring_function

    async def _execute(self, partitions: GraphPartitions, input_reasoning_states: ReasoningStateExecutionType) -> ReasoningStateExecutionType:
        try:
            # Call the scoring function with unpacked input_reasoning_states
            score = self.scoring_function(**input_reasoning_states)
            return {"score": score}
        except Exception as e:
            raise OperationFailed(f"Scoring function failed: {e}")