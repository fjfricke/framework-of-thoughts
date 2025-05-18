from typing import Callable, Union, Tuple
from llm_graph_optimizer.graph_of_operations.graph_of_operations import GraphPartitions
from llm_graph_optimizer.graph_of_operations.types import ReasoningStateType, ReasoningState
from llm_graph_optimizer.language_models.abstract_language_model import AbstractLanguageModel
from llm_graph_optimizer.language_models.helpers.language_model_config import LLMResponseType
from llm_graph_optimizer.language_models.cache.types import CacheSeed
from llm_graph_optimizer.measurement.measurement import Measurement, MeasurementsWithCache

from ..helpers.exceptions import OperationFailed
from ..abstract_operation import AbstractOperation
import inspect


class BaseLLMOperation(AbstractOperation):
    """
    LLM operation.
    """

    def __init__(self, llm: AbstractLanguageModel, prompter: Callable[..., str], parser: Callable[[Union[str, Tuple[str, ReasoningState]]], ReasoningState], use_cache: bool = True, params: dict = None, input_types: ReasoningStateType = None, output_types: ReasoningStateType = None, name: str = None, cache_seed: CacheSeed = None):
        """
        Initialize the BaseLLMOperation.

        :param llm: The language model to use.
        :param prompter: A callable that generates a prompt. It can take named arguments corresponding to the input keys.
        :param parser: A callable that parses the response. It takes the response as a single string argument.
        :param params: Additional parameters for the operation.
        :param input_types: Expected input types for the operation.
        :param output_types: Expected output types for the operation.
        """
        self.llm = llm
        if not llm.llm_response_type == LLMResponseType.TEXT:
            raise ValueError(f"Only LLMs that return text are supported. The given LLM {llm} returns {llm.llm_response_type}.")
        self.prompter = prompter
        self.parser = parser
        self.use_cache = use_cache
        self.cache_seed = cache_seed
        super().__init__(input_types=input_types, output_types=output_types, params=params, name=name)

    async def _execute(self, partitions: GraphPartitions, input_reasoning_states: ReasoningState) -> tuple[ReasoningState, MeasurementsWithCache]:
        measurement = Measurement()
        try:
            # Unpack input_reasoning_states into named arguments for the prompter
            prompt = self.prompter(**input_reasoning_states)
            
            # Query the language model
            response, measurement = await self.llm.query(prompt=prompt, use_cache=self.use_cache, cache_seed=self.cache_seed)
            
            # Pass the response to the parser
            parser_signature = inspect.signature(self.parser)
            if len(parser_signature.parameters) == 1:
                # Parser expects only the response
                return self.parser(response), measurement
            elif len(parser_signature.parameters) == 2:
                # Parser expects response and input_reasoning_states
                return self.parser(response, input_reasoning_states), measurement
            else:
                raise ValueError("Parser has an unexpected number of arguments.")
        except Exception as e:
            print(e)
            raise OperationFailed(e, measurement=measurement)
