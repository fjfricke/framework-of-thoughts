from examples.game_of_24.programs.operations.value_operation import ValueOperation
from examples.game_of_24.programs.operations.last_step_value_operation import LastStepValueOperation

from llm_graph_optimizer.graph_of_operations.graph_partitions import GraphPartitions
from llm_graph_optimizer.graph_of_operations.types import Dynamic, Edge, ReasoningState
from llm_graph_optimizer.language_models.abstract_language_model import AbstractLanguageModel
from llm_graph_optimizer.measurement.measurement import Measurement
from llm_graph_optimizer.operations.abstract_operation import AbstractOperation


class ParallelEvaluationOperation(AbstractOperation):
    def __init__(self, llm: AbstractLanguageModel, samples: int, branch_index: int, params: dict = None, name: str = None):
        """
        Params:
            value_operation_type: ValueOperation or LastStepValueOperation
        """
        input_types = {"expressions": list[str], "lefts": list[list[int]]}
        output_types = Dynamic # {"expression_{i}": str, "left_{i}": list[int]} for each ValueOperation
        super().__init__(input_types, output_types, params, name)
        self.llm = llm
        self.samples = samples
        self.branch_index = branch_index

    async def _execute(self, partitions: GraphPartitions, input_reasoning_states: ReasoningState) -> tuple[ReasoningState, Measurement | None]:

        assert len(input_reasoning_states["expressions"]) == len(input_reasoning_states["lefts"])

        descendants_edges = partitions.descendants.successor_edges(self)
        descendants_edges = [edge for edge in descendants_edges if edge.from_node_key == "score"]
        assert len(descendants_edges) == 1

        value_nodes = []
        for i in range(len(input_reasoning_states["expressions"])):
            if self.params.get("value_operation_type") == ValueOperation:
                value_node = ValueOperation(samples=self.samples, llm=self.llm, name=f"ValueOperation_{i}")
            elif self.params.get("value_operation_type") == LastStepValueOperation:
                value_node = LastStepValueOperation(samples=self.samples, llm=self.llm, name=f"LastStepValueOperation_{i}")
            else:
                raise ValueError(f"Invalid value operation type: {self.params.get('value_operation_type')}")
            value_nodes.append(value_node)
            partitions.exclusive_descendants.add_node(value_node)
            partitions.exclusive_descendants.add_edge(Edge(self, value_node, f"left_{i}", "left"))
            partitions.exclusive_descendants.add_edge(Edge(self, value_node, f"expression_{i}", "expression"))

        partitions.move_start_node_and_duplicate_edges(current_edge=descendants_edges[0], new_from_nodes=value_nodes, new_from_node_keys=["score"] * len(value_nodes), orders=list(range(self.branch_index*1000, self.branch_index*1000+len(value_nodes))))


        output_reasoning_states = {}
        for i in range(len(input_reasoning_states["expressions"])):
            output_reasoning_states[f"expression_{i}"] = input_reasoning_states["expressions"][i]
            output_reasoning_states[f"left_{i}"] = input_reasoning_states["lefts"][i]
        return output_reasoning_states, None
