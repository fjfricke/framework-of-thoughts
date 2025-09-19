from examples.game_of_24.programs.operations.extract_answer_operation import ExtractAnswerOperation
from examples.game_of_24.programs.operations.helpers.find_nodes import find_nodes, FindLastValuesType
from examples.game_of_24.programs.prompter_parser import value_last_step_parser, value_last_step_prompt
from llm_graph_optimizer.graph_of_operations.graph_partitions import GraphPartitions
from llm_graph_optimizer.graph_of_operations.types import Dynamic, Edge, ManyToOne, ReasoningState
from llm_graph_optimizer.language_models.abstract_language_model import AbstractLanguageModel
from llm_graph_optimizer.measurement.measurement import Measurement
from llm_graph_optimizer.operations.abstract_operation import AbstractOperation
from llm_graph_optimizer.operations.base_operations.score_operation import ScoreOperation
from llm_graph_optimizer.operations.llm_operations.base_llm_operation import BaseLLMOperation


class LastStepValueOperation(AbstractOperation):
    def __init__(self, samples: int, llm: AbstractLanguageModel, params: dict = None, name: str = None):
        input_types = {"left": list[int], "expression": str}
        output_types = Dynamic # {"left": list[int]}
        super().__init__(input_types, output_types, params, name)
        self.samples = samples
        self.llm = llm

    async def _execute(self, partitions: GraphPartitions, input_reasoning_states: ReasoningState) -> tuple[ReasoningState, Measurement | None]:

        score_node = ScoreOperation(
            input_types={"values": ManyToOne[float]},
            output_type=float,
            scoring_function=lambda values: sum(values)
        )
        partitions.exclusive_descendants.add_node(score_node)

        extract_answer_node = ExtractAnswerOperation()
        partitions.exclusive_descendants.add_node(extract_answer_node)
        partitions.exclusive_descendants.add_dependency_edge(self, extract_answer_node)

        left_and_expressions_nodes_and_nodekeys = find_nodes(self, partitions, FindLastValuesType.ALL).reverse()
        for i, (expression_node, expression_nodekey) in enumerate(zip(left_and_expressions_nodes_and_nodekeys.expression_nodes, left_and_expressions_nodes_and_nodekeys.expression_nodekeys)):
            partitions.add_edge(Edge(expression_node, extract_answer_node, expression_nodekey, "expressions", i))
        
        for i in range(self.samples):
            value_node = BaseLLMOperation(
                llm=self.llm,
                prompter=value_last_step_prompt,
                parser=value_last_step_parser,
                input_types={"left": list[int], "answer": str},
                output_types={"value": float},
                cache_seed=i,
                name=f"LLMEvaluateLastStep{i}"
            )
            partitions.exclusive_descendants.add_node(value_node)
            partitions.exclusive_descendants.add_dependency_edge(self, value_node)
            partitions.add_edge(Edge(left_and_expressions_nodes_and_nodekeys.left_nodes[0], value_node, left_and_expressions_nodes_and_nodekeys.left_nodekeys[0], "left"))
            partitions.exclusive_descendants.add_edge(Edge(extract_answer_node, value_node, "answer", "answer"))
            partitions.exclusive_descendants.add_edge(Edge(value_node, score_node, "value", "values"))
        
        descendants_edges = partitions.descendants.successor_edges(self)
        descendants_edges = [edge for edge in descendants_edges if edge.from_node_key == "score"]
        assert len(descendants_edges) == 1
        partitions.move_edge_start_node(current_edge=descendants_edges[0], new_from_node=score_node, new_from_node_key="score")

        return {}, None