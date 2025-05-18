from examples.game_of_24.programs.prompter_parser import value_parser, value_prompt
from llm_graph_optimizer.language_models.abstract_language_model import AbstractLanguageModel
from llm_graph_optimizer.operations.llm_operations.base_llm_operation import BaseLLMOperation


class ValueOperation(BaseLLMOperation):
    def __init__(self, llm: AbstractLanguageModel, cache_seed: str, params: dict = None, name: str = None):
        super().__init__(
            llm=llm,
            prompter=value_prompt,
            parser=value_parser,
            input_types={"input": list[int]},
            output_types={"value": float | None},
            cache_seed=cache_seed,
            params=params,
            name=name
        )