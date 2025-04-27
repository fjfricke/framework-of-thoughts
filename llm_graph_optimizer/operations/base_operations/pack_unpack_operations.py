from typing import get_args, get_origin
from llm_graph_optimizer.graph_of_operations.graph_partitions import GraphPartitions
from llm_graph_optimizer.graph_of_operations.types import ManyToOne, ReasoningState, ReasoningStateType
from llm_graph_optimizer.operations.abstract_operation import AbstractOperation
from llm_graph_optimizer.measurement.measurement import Measurement

class PackOperation(AbstractOperation):
    """
    Operation to pack multiple reasoning states into lists.

    This operation ensures that the output reasoning states are lists, and the input
    reasoning states are converted accordingly.

    Attributes:
        output_types (ReasoningStateType): Expected types for output reasoning states.
        params (dict): Parameters for the operation.
        name (str): Name of the operation.
    """

    def __init__(self, output_types: ReasoningStateType, params: dict = None, name: str = None):
        """
        Initialize a PackOperation instance.

        Args:
            output_types (ReasoningStateType): Expected types for output reasoning states.
            params (dict, optional): Parameters for the operation. Defaults to None.
            name (str, optional): Name of the operation. Defaults to the class name.

        Raises:
            ValueError: If any output type is not a list.
        """
        if not all(get_origin(value) is list for value in output_types.values()):
            raise ValueError("Output types must be lists.")
        input_types = {key: ManyToOne[get_args(value)[0]] for key, value in output_types.items()}
        super().__init__(input_types, output_types, params, name)

    async def _execute(self, partitions: GraphPartitions, input_reasoning_states: ReasoningState) -> tuple[ReasoningState, Measurement | None]:
        """
        Execute the pack operation.

        Args:
            partitions (GraphPartitions): Partitions of the graph.
            input_reasoning_states (ReasoningState): Input reasoning states.

        Returns:
            tuple[ReasoningState, Measurement | None]: Packed reasoning states and measurements.
        """
        return input_reasoning_states, None
