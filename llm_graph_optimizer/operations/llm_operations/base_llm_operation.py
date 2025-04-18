from typing import Callable
from llm_graph_optimizer.graph_of_operations.graph_of_operations import GraphPartitions
from llm_graph_optimizer.graph_of_operations.types import ReasoningStateType, ReasoningStateExecutionType
from llm_graph_optimizer.language_models.abstract_language_model import AbstractLanguageModel

from ..helpers.exceptions import OperationFailed
from ..abstract_operation import AbstractOperation


class BaseLLMOperation(AbstractOperation):
    """
    LLM operation.
    """

    def __init__(self, llm: AbstractLanguageModel, prompter: Callable[..., str], parser: Callable[[str], dict[str, any]], use_cache: bool = True, params: dict = None, input_types: ReasoningStateType = None, output_types: ReasoningStateType = None, name: str = None):
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
        self.prompter = prompter
        self.parser = parser
        self.use_cache = use_cache
        super().__init__(input_types=input_types, output_types=output_types, params=params, name=name)

    async def _execute(self, partitions: GraphPartitions, input_reasoning_states: ReasoningStateExecutionType) -> ReasoningStateExecutionType:
        try:
            # Unpack input_reasoning_states into named arguments for the prompter
            prompt = self.prompter(**input_reasoning_states)
            
            # Query the language model
            response, query_metadata = await self.llm.query(prompt=prompt, use_cache=self.use_cache)
            
            # Pass the response to the parser
            return self.parser(response)
        except Exception as e:
            print(e)
            raise OperationFailed(e)
