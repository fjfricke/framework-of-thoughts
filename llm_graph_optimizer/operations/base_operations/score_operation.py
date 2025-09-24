from typing import Callable

from llm_graph_optimizer.graph_of_operations.graph_partitions import GraphPartitions
from llm_graph_optimizer.graph_of_operations.types import ReasoningStateType, ReasoningState
from llm_graph_optimizer.operations.abstract_operation import AbstractOperation
from llm_graph_optimizer.operations.helpers.exceptions import OperationFailed
from llm_graph_optimizer.measurement.measurement import Measurement

class ScoreOperation(AbstractOperation):
    """
    Represents a scoring operation within a graph of operations.
    
    This class is responsible for calculating a score based on the provided input reasoning states.
    The output reasoning state will contain a single key-value pair with the key "score" and a value
    of the specified output type.
    """
    
    def __init__(self, input_types: ReasoningStateType, output_type: type, scoring_function: Callable[..., any], params: dict = None, name: str = None):
        """
        Initializes the ScoreOperation with the given input types, output type, and scoring function.

        :param input_types: The expected types of the input reasoning states.
        :param output_type: The type of the score that will be produced.
        :param scoring_function: A callable that computes the score based on the input reasoning states.
        """
        output_types = {"score": output_type}
        super().__init__(input_types, output_types, params, name)
        self.scoring_function = scoring_function

    async def _execute(self, partitions: GraphPartitions, input_reasoning_states: ReasoningState) -> tuple[ReasoningState, Measurement | None]:
        try:
            # Call the scoring function with unpacked input_reasoning_states
            score = self.scoring_function(**input_reasoning_states)
            return {"score": score}, None
        except Exception as e:
            raise OperationFailed(f"Scoring function failed: {e}")