from typing import Callable
from llm_graph_optimizer.graph_of_operations.graph_of_operations import GraphPartitions
from llm_graph_optimizer.language_models.abstract_language_model import AbstractLanguageModel

from .helpers.exceptions import OperationFailed
from .abstract_operation import AbstractOperation


class LLMOperation(AbstractOperation):
    """
    LLM operation.
    """

    def __init__(self, llm: AbstractLanguageModel, prompter: Callable[[list[any]], str], parser: Callable[[str], list[any]], params: dict = None, input_types: list[type] = None, output_types: list[type] = None):
        self.llm = llm
        self.prompter = prompter
        self.parser = parser
        super().__init__(params, input_types, output_types)

    async def _execute(self, partitions: GraphPartitions, input_reasoning_states: list[any]) -> list[any]:
        try:
            prompt = self.prompter(input_reasoning_states)
            response = await self.llm.generate_text(prompt)
            return self.parser(response)
        except Exception as e:
            raise OperationFailed(e)
