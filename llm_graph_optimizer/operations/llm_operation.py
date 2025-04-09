from typing import Callable
from llm_graph_optimizer.graph_of_operations.graph_of_operations import GraphPartitions
from llm_graph_optimizer.language_models.abstract_language_model import AbstractLanguageModel

from .helpers.exceptions import OperationFailed
from .abstract_operation import AbstractOperation


class LLMOperation(AbstractOperation):
    """
    LLM operation.
    """

    def __init__(self, llm: AbstractLanguageModel, prompter: Callable[[dict[str, any]], str], parser: Callable[[str], dict[str, any]], params: dict = None, input_types: dict[str, type] = None, output_types: dict[str, type] = None):
        self.llm = llm
        self.prompter = prompter
        self.parser = parser
        super().__init__(input_types=input_types, output_types=output_types, params=params)

    async def _execute(self, partitions: GraphPartitions, input_reasoning_states: dict[str, any]) -> dict[str, any]:
        try:
            prompt = self.prompter(input_reasoning_states)
            response, query_metadata = await self.llm.query(prompt)
            return self.parser(response)
        except Exception as e:
            print(e)
            raise OperationFailed(e)
