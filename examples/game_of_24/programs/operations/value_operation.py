from examples.game_of_24.programs.prompter_parser import value_parser, value_prompt
from llm_graph_optimizer.graph_of_operations.graph_partitions import GraphPartitions
from llm_graph_optimizer.graph_of_operations.types import Dynamic, Edge, ManyToOne, ReasoningState
from llm_graph_optimizer.language_models.abstract_language_model import AbstractLanguageModel
from llm_graph_optimizer.measurement.measurement import Measurement
from llm_graph_optimizer.operations.abstract_operation import AbstractOperation
from llm_graph_optimizer.operations.base_operations.score_operation import ScoreOperation
from llm_graph_optimizer.operations.llm_operations.base_llm_operation import BaseLLMOperation


class ValueOperation(AbstractOperation):
    """
    This operation creates samples branches to evaluate a single proposal and rewires its out-edges to the score node (descendant).

    """
    def __init__(self, samples: int, llm: AbstractLanguageModel, params: dict = None, name: str = None):
        input_types = {"expression": str, "left": list[int]}
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

        predecessor_edge = partitions.ancestors.predecessor_edges(self)[0]
        
        for i in range(self.samples):
            value_node = BaseLLMOperation(
                llm=self.llm,
                prompter=value_prompt,
                parser=value_parser,
                input_types={"left": list[int]},
                output_types={"value": float},
                cache_seed=i,
                name=f"LLMEvaluate{i}"
            )
            partitions.exclusive_descendants.add_node(value_node)
            partitions.exclusive_descendants.add_dependency_edge(self, value_node)
            partitions.add_edge(Edge(predecessor_edge.from_node, value_node, predecessor_edge.from_node_key, "left"))
            partitions.exclusive_descendants.add_edge(Edge(value_node, score_node, "value", "values"))
        
        descendants_edges = partitions.descendants.successor_edges(self)
        descendants_edges = [edge for edge in descendants_edges if edge.from_node_key == "score"]
        assert len(descendants_edges) == 1
        partitions.move_edge_start_node(current_edge=descendants_edges[0], new_from_node=score_node, new_from_node_key="score")

        return {}, None