from typing import Callable, get_origin

from llm_graph_optimizer.graph_of_operations.graph_partitions import GraphPartitions
from llm_graph_optimizer.graph_of_operations.types import ManyToOne, ReasoningState, ReasoningStateType
from llm_graph_optimizer.operations.abstract_operation import AbstractOperation
from llm_graph_optimizer.measurement.measurement import Measurement

class FilterOperation(AbstractOperation):
    """
    Operation to filter reasoning states based on a custom filter function.

    This operation applies a filter function to the input reasoning states
    and produces filtered reasoning states as output.

    Attributes:
        output_types (ReasoningStateType): Expected types for output reasoning states.
        input_types (ReasoningStateType): Expected types for input reasoning states.
        filter_function (Callable[..., ReasoningState]): Function to filter reasoning states.
        params (dict): Parameters for the operation.
        name (str): Name of the operation.
    """

    def __init__(self, output_types: ReasoningStateType, input_types: ReasoningStateType, filter_function: Callable[..., ReasoningState], params: dict = None, name: str = None):
        """
        Initialize a FilterOperation instance.

        Args:
            output_types (ReasoningStateType): Expected types for output reasoning states.
            input_types (ReasoningStateType): Expected types for input reasoning states.
            filter_function (Callable[..., ReasoningState]): Function to filter reasoning states.
            params (dict, optional): Parameters for the operation. Defaults to None.
            name (str, optional): Name of the operation. Defaults to the class name.

        Raises:
            TypeError: If any input type is not of origin type ManyToOne.
        """
        if not all(get_origin(value) is ManyToOne for value in input_types.values()):
            raise TypeError("All input types must have keys of origin type ManyToOne")
        self.filter_function = filter_function
        super().__init__(input_types, output_types, params, name)

    async def _execute(self, partitions: GraphPartitions, input_reasoning_states: ReasoningState) -> tuple[ReasoningState, Measurement | None]:
        filtered_reasoning_states = self.filter_function(**input_reasoning_states)
        return filtered_reasoning_states, None