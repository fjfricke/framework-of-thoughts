from pathlib import Path
from examples.game_of_24.programs.operations.find_last_values import FindLastValuesOperation, FindLastValuesType
from examples.game_of_24.programs.operations.last_step_value_operation import LastStepValueOperation
from examples.game_of_24.programs.operations.parallel_evaluation import ParallelEvaluationOperation
from examples.game_of_24.programs.operations.value_operation import ValueOperation
from llm_graph_optimizer.controller.controller import Controller
from llm_graph_optimizer.graph_of_operations.graph_of_operations import GraphOfOperations
from llm_graph_optimizer.graph_of_operations.types import Edge, ManyToOne
from llm_graph_optimizer.language_models.abstract_language_model import AbstractLanguageModel

from llm_graph_optimizer.language_models.cache.cache import CacheContainer
from llm_graph_optimizer.language_models.openai_chat import OpenAIChat
from llm_graph_optimizer.measurement.process_measurement import ProcessMeasurement
from llm_graph_optimizer.operations.base_operations.end import End
from llm_graph_optimizer.operations.base_operations.filter_operation_with_edge_move import Correspondence, FilterOperationWithEdgeMove
from llm_graph_optimizer.operations.base_operations.start import Start
from llm_graph_optimizer.operations.llm_operations.base_llm_operation import BaseLLMOperation
from examples.game_of_24.programs.prompter_parser import propose_prompt, propose_parser
from llm_graph_optimizer.schedulers.schedulers import Scheduler

def tot_controller(llm: AbstractLanguageModel, num_examples: int = 10, samples: list[int] = [10, 3, 2], keep_top_n: int = 2):

    start_node = Start(
        input_types={"input_list": list[int]}
    )

    propose_op = lambda: BaseLLMOperation(
        llm=llm,
        prompter=lambda input_list: propose_prompt(num_examples=num_examples, input_list=input_list),
        parser=propose_parser,
        input_types={"input_list": list[int]},
        output_types={"expressions": list[str], "lefts": list[list[int]]},
        name="Propose"
    )

    parallel_evaluation_op = lambda i: ParallelEvaluationOperation(llm=llm, samples=samples[i], params={"value_operation_type": ValueOperation if i < 2 else LastStepValueOperation}, name=f"ParallelEvaluationOperation_{i}")

    filter_op = lambda i: FilterOperationWithEdgeMove(
        input_types={"score": ManyToOne[float]},
        correspondence=Correspondence.ONE_TO_ONE,
        filter_function=lambda score: sorted(range(len(score)), key=lambda i: score[i], reverse=True)[:keep_top_n],
        name=f"FilterOperationWithEdgeMove_{i}"
    )
    find_last_values_op = lambda i: FindLastValuesOperation(name=f"FindLastValuesOperation_{i}", params={"type": FindLastValuesType.ONLY_ONE})

    propose_nodes = [propose_op() for _ in range(3)]
    parallel_evaluation_nodes = [parallel_evaluation_op(i) for i in range(3)]
    filter_nodes = [filter_op(i) for i in range(3)]
    find_last_values_nodes = [find_last_values_op(i) for i in range(2)]

    end_node = End(
        # input_types={"lefts": list[list[int]], "expressions": list[str]}
        input_types={"score": float}
    )

    tot_graph = GraphOfOperations()
    tot_graph.add_node(start_node)
    [tot_graph.add_node(propose_node) for propose_node in propose_nodes]
    [tot_graph.add_node(parallel_evaluation_node) for parallel_evaluation_node in parallel_evaluation_nodes]
    [tot_graph.add_node(filter_node) for filter_node in filter_nodes]
    [tot_graph.add_node(find_last_values_node) for find_last_values_node in find_last_values_nodes]
    tot_graph.add_node(end_node)

    tot_graph.add_edge(Edge(start_node, propose_nodes[0], from_node_key="input_list", to_node_key="input_list"))

    tot_graph.add_edge(Edge(propose_nodes[0], parallel_evaluation_nodes[0], from_node_key="expressions", to_node_key="expressions"))
    tot_graph.add_edge(Edge(propose_nodes[0], parallel_evaluation_nodes[0], from_node_key="lefts", to_node_key="lefts"))
    tot_graph.add_edge(Edge(parallel_evaluation_nodes[0], filter_nodes[0], from_node_key="score", to_node_key="score"))
    tot_graph.add_edge(Edge(filter_nodes[0], find_last_values_nodes[0], from_node_key="score", to_node_key="score"))
    tot_graph.add_edge(Edge(find_last_values_nodes[0], propose_nodes[1], from_node_key="left", to_node_key="input_list"))

    tot_graph.add_edge(Edge(propose_nodes[1], parallel_evaluation_nodes[1], from_node_key="expressions", to_node_key="expressions"))
    tot_graph.add_edge(Edge(propose_nodes[1], parallel_evaluation_nodes[1], from_node_key="lefts", to_node_key="lefts"))
    tot_graph.add_edge(Edge(parallel_evaluation_nodes[1], filter_nodes[1], from_node_key="score", to_node_key="score"))
    tot_graph.add_edge(Edge(filter_nodes[1], find_last_values_nodes[1], from_node_key="score", to_node_key="score"))
    tot_graph.add_edge(Edge(find_last_values_nodes[1], propose_nodes[2], from_node_key="left", to_node_key="input_list"))

    tot_graph.add_edge(Edge(propose_nodes[2], parallel_evaluation_nodes[2], from_node_key="lefts", to_node_key="lefts"))
    tot_graph.add_edge(Edge(propose_nodes[2], parallel_evaluation_nodes[2], from_node_key="expressions", to_node_key="expressions"))
    tot_graph.add_edge(Edge(parallel_evaluation_nodes[2], filter_nodes[2], from_node_key="score", to_node_key="score"))

    tot_graph.add_edge(Edge(filter_nodes[2], end_node, from_node_key="score", to_node_key="score"))

    

    process_measurement = ProcessMeasurement(graph_of_operations=tot_graph)

    tot_controller = Controller(
        graph_of_operations=tot_graph,
        scheduler=Scheduler.BFS,
        max_concurrent=1,
        process_measurement=process_measurement
    )

    return tot_controller

if __name__ == "__main__":
    import asyncio
    cache = CacheContainer.from_persistent_cache_file(Path(__file__).parent / "output" / "dataset_cache.pkl", load_as_virtual_persistent_cache=True, skip_on_file_not_found=True)
    llm = OpenAIChat(model="gpt-4o", cache=cache)
    tot_controller = tot_controller(llm, num_examples=3, samples=[3, 2, 1], keep_top_n=1)
    tot_controller.graph_of_operations.snapshot.visualize(show_multiedges=False, show_values=True, show_keys=True, show_state=True)
    result, measurement = asyncio.run(tot_controller.execute(input={"input_list": [1, 2, 3, 4]}))
    print(result)
    print(measurement)
    tot_controller.graph_of_operations.snapshot.visualize(show_multiedges=False, show_values=True, show_keys=True, show_state=True)
    cache.save_persistent_cache()