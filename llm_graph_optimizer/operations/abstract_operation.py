from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import Callable, List, get_args, get_origin
import uuid
from typeguard import TypeCheckError, check_type
from llm_graph_optimizer.graph_of_operations.graph_of_operations import GraphOfOperations, GraphPartitions
from llm_graph_optimizer.graph_of_operations.types import Dynamic, ManyToOne, ReasoningState, ReasoningStateType, StateNotSet, StateSetFailed
from llm_graph_optimizer.measurement.measurement import Measurement, MeasurementsWithCache
from llm_graph_optimizer.operations.helpers.exceptions import OperationFailed
from .helpers.node_state import NodeState


class AbstractOperation(ABC):
    """
    Abstract base class for all graph operations.

    This class defines the structure and behavior of operations that can be executed
    within a graph of operations. It includes methods for execution, validation of
    input and output reasoning states, and caching.

    Attributes:
        params (dict): Parameters for the operation.
        cache (dict): Cache for storing intermediate results.
        node_state (NodeState): Current state of the node.
        input_types (ReasoningStateType): Expected types for input reasoning states.
        output_types (ReasoningStateType): Expected types for output reasoning states.
        output_reasoning_states (dict): Resulting reasoning states after execution.
        name (str): Name of the operation.
        logger (logging.Logger): Logger for the operation.
    """

    def __init__(self, input_types: ReasoningStateType, output_types: ReasoningStateType, params: dict = None, name: str = None):
        """
        Initialize an AbstractOperation instance.

        Args:
            input_types (ReasoningStateType): Expected types for input reasoning states.
            output_types (ReasoningStateType): Expected types for output reasoning states.
            params (dict, optional): Parameters for the operation. Defaults to None.
            name (str, optional): Name of the operation. Used in visualization and logging. Defaults to the class name.
        """
        self.params = params
        self.cache = {}
        self.node_state = NodeState.WAITING
        self.input_types = input_types
        self.output_types = output_types
        self.output_reasoning_states = {}
        self.name = name or self.__class__.__name__
        self.uuid = uuid.uuid4()
        self.logger = logging.getLogger(__name__)

    @classmethod
    def factory(cls, **kwargs) -> AbstractOperationFactoryWithParams:
        """
        Create a factory for the operation.

        Args:
            **kwargs: Initial parameters for the operation.

        Returns:
            AbstractOperationFactoryWithParams: A callable factory that can create
            instances of the operation with additional parameters.
        """
        def factory_without_params(**later_kwargs) -> AbstractOperationFactory:
            # Combine initial kwargs with later_kwargs
            combined_kwargs = {**kwargs, **later_kwargs}
            return cls(**combined_kwargs)

        return factory_without_params

    @abstractmethod
    async def _execute(self, partitions: GraphPartitions, input_reasoning_states: ReasoningState) -> tuple[ReasoningState, Measurement | MeasurementsWithCache | None]:
        """
        Abstract method to execute the operation. Needs to be implemented by the operation.

        Args:
            partitions (GraphPartitions): Partitions of the graph.
            input_reasoning_states (ReasoningState): Input reasoning states.

        Returns:
            tuple[ReasoningState, Measurement | MeasurementsWithCache | None]:
            Resulting reasoning states and measurements.
        """
        pass

    async def execute(self, graph: GraphOfOperations) -> Measurement | MeasurementsWithCache:
        """
        Execute the operation within a graph.

        Args:
            graph (GraphOfOperations): The graph containing the operation.

        Returns:
            Measurement | MeasurementsWithCache: Measurements resulting from the execution.

        Raises:
            TypeError: If input or output reasoning states are invalid.
            KeyError: If required keys are missing in input or output reasoning states.
            ValueError: If input reasoning states are not set.
            OperationFailed: If the operation cannot handle failed predecessors.
        """
        input_reasoning_states = graph.get_input_reasoning_states(self)
        
        # Validate input_reasoning_states
        if not isinstance(input_reasoning_states, dict):
            raise TypeError(f"Inputs must be a dictionary, got {type(input_reasoning_states)}")
        
        for key, expected_type in self.input_types.items():
            if key not in input_reasoning_states:
                raise KeyError(f"Missing input key: {key}")
            
            if input_reasoning_states[key] is StateNotSet:
                raise ValueError(f"Input reasoning state for key {key} is not set")
            
            if input_reasoning_states[key] is StateSetFailed or (StateSetFailed in input_reasoning_states[key] if isinstance(input_reasoning_states[key], list) else False):
                try:
                    check_type(input_reasoning_states[key], expected_type)
                except TypeCheckError as e:
                    if get_origin(expected_type) is ManyToOne:
                        inner_type = get_args(expected_type)[0]
                        try:
                            check_type(input_reasoning_states[key], List[inner_type])
                        except TypeCheckError as e:
                            raise OperationFailed(f"Input reasoning state in operation {self.name} for key {key} is set to failed and this operation cannot handle failed predecessors.") from e
                    else:
                        raise OperationFailed(f"Input reasoning state in operation {self.name} for key {key} is set to failed and this operation cannot handle failed predecessors.") from e
            
            try:
                check_type(input_reasoning_states[key], expected_type)
            except TypeCheckError as e:
                if get_origin(expected_type) is ManyToOne:
                    inner_type = get_args(expected_type)[0]
                    try:
                        check_type(input_reasoning_states[key], List[inner_type])
                    except TypeCheckError as e:
                        raise TypeError(f"Input '{key}' must be of type {expected_type}, got {type(input_reasoning_states[key])}") from e
                else:
                    raise TypeError(f"Input '{key}' must be of type {expected_type}, got {type(input_reasoning_states[key])}") from e

        partitions = graph.partitions(self)
        result, measurement_or_measurements_with_cache = await self._execute(partitions, input_reasoning_states)

        if not measurement_or_measurements_with_cache:
            measurement_or_measurements_with_cache = Measurement()

        # Validate result
        if not isinstance(result, dict):
            raise TypeError(f"Outputs must be a dictionary, got {type(result)}")
        
        if self.output_types is Dynamic:
            self.logger.warning(f"Output types are dynamic for operation {self.name}, skipping validation")
        else:
            for key, expected_type in self.output_types.items():
                if key not in result:
                    raise KeyError(f"Missing output key: {key}")
                
                try:
                    check_type(result[key], expected_type)
                except TypeCheckError as e:
                    raise TypeError(f"Output '{key}' must be of type {expected_type}, got {type(result[key])}") from e


        self.output_reasoning_states = result
        self.logger.debug(f"Output reasoning states: {self.output_reasoning_states} for operation {self.name}")
        graph.update_edge_values(self, result)
        return measurement_or_measurements_with_cache

AbstractOperationFactory = Callable[[], AbstractOperation]
AbstractOperationFactoryWithParams = Callable[..., AbstractOperation]