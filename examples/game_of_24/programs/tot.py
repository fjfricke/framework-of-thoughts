from pathlib import Path

from examples.game_of_24.programs.operations.find_last_answer_operation import FindLastAnswerOperation
from examples.game_of_24.programs.operations.find_last_values import FindLastValuesOperation, FindLastValuesType
from examples.game_of_24.programs.operations.last_step_value_operation import LastStepValueOperation
from examples.game_of_24.programs.operations.parallel_evaluation import ParallelEvaluationOperation
from examples.game_of_24.programs.operations.value_operation import ValueOperation
from examples.openai_pricing import OPENAI_PRICING
from llm_graph_optimizer.controller.controller import Controller
from llm_graph_optimizer.graph_of_operations.graph_of_operations import GraphOfOperations
from llm_graph_optimizer.graph_of_operations.types import Edge, ManyToOne
from llm_graph_optimizer.language_models.abstract_language_model import AbstractLanguageModel

from llm_graph_optimizer.language_models.cache.cache import CacheContainer
from llm_graph_optimizer.language_models.helpers.openai_rate_limiter import OpenAIRateLimiter
from llm_graph_optimizer.language_models.openai_chat import OpenAIChat
from llm_graph_optimizer.measurement.process_measurement import ProcessMeasurement
from llm_graph_optimizer.operations.base_operations.end import End
from llm_graph_optimizer.operations.base_operations.filter_operation_with_edge_move import Correspondence, FilterOperationWithEdgeMove
from llm_graph_optimizer.operations.base_operations.start import Start
from llm_graph_optimizer.operations.llm_operations.base_llm_operation import BaseLLMOperation
from examples.game_of_24.programs.prompter_parser import propose_prompt, propose_parser
from llm_graph_optimizer.schedulers.schedulers import Scheduler

def tot_controller(llm: AbstractLanguageModel, num_examples: int = 10, samples: list[int] = [10, 3, 2], keep_top_n: list[int] = [2, 2], max_concurrent: int = 20) -> Controller:

    start_node = Start(
        input_types={"input_list": list[int]}
    )

    propose_op = lambda i: BaseLLMOperation(
        llm=llm,
        prompter=lambda input_list: propose_prompt(num_examples=num_examples, input_list=input_list),
        parser=propose_parser,
        input_types={"input_list": list[int]},
        output_types={"expressions": list[str], "lefts": list[list[int]]},
        name=f"Propose_{i}"
    )

    parallel_evaluation_op = lambda i, value_operation_type, branch_index: ParallelEvaluationOperation(llm=llm, samples=samples[i], params={"value_operation_type": value_operation_type}, name=f"ParallelEvaluationOperation_{i}", branch_index=branch_index)

    filter_op = lambda i, keep_top: FilterOperationWithEdgeMove(
        input_types={"score": ManyToOne[float]},
        correspondence=Correspondence.ONE_TO_ONE,
        filter_function=lambda score: sorted(range(len(score)), key=lambda i: score[i], reverse=True)[:keep_top],
        name=f"FilterOperationWithEdgeMove_{i}"
    )
    find_last_values_op = lambda i: FindLastValuesOperation(name=f"FindLastValuesOperation_{i}", params={"type": FindLastValuesType.ONLY_ONE})

    end_node = End(
        # input_types={"lefts": list[list[int]], "expressions": list[str]}
        input_types={"score": float, "answer": str}
    )

    tot_graph = GraphOfOperations()
    tot_graph.add_node(start_node)
    tot_graph.add_node(end_node)

    # first layer
    propose_node_0 = propose_op(0)
    tot_graph.add_node(propose_node_0)
    tot_graph.add_edge(Edge(start_node, propose_node_0, from_node_key="input_list", to_node_key="input_list"))
    parallel_evaluation_node_0 = parallel_evaluation_op(0, ValueOperation, 0)
    tot_graph.add_node(parallel_evaluation_node_0)
    tot_graph.add_edge(Edge(propose_node_0, parallel_evaluation_node_0, from_node_key="expressions", to_node_key="expressions"))
    tot_graph.add_edge(Edge(propose_node_0, parallel_evaluation_node_0, from_node_key="lefts", to_node_key="lefts"))
    filter_node_0 = filter_op(0, keep_top_n[0])
    tot_graph.add_node(filter_node_0)
    tot_graph.add_edge(Edge(parallel_evaluation_node_0, filter_node_0, from_node_key="score", to_node_key="score"))
    find_last_values_nodes_0 = [find_last_values_op(1) for i in range(keep_top_n[0])]
    propose_nodes_1 = [propose_op(1) for i in range(keep_top_n[0])]
    parallel_evaluation_nodes_1 = [parallel_evaluation_op(1, ValueOperation, i) for i in range(keep_top_n[0])]
    filter_node_1 = filter_op(1, keep_top_n[1])
    tot_graph.add_node(filter_node_1)
    for i, (find_last_values_node_0, propose_node_1, parallel_evaluation_node_1) in enumerate(zip(find_last_values_nodes_0, propose_nodes_1, parallel_evaluation_nodes_1)):
        tot_graph.add_node(find_last_values_node_0)
        tot_graph.add_node(propose_node_1)
        tot_graph.add_edge(Edge(filter_node_0, find_last_values_node_0, from_node_key="score", to_node_key="score"), order=i)
        # second layer
        tot_graph.add_edge(Edge(find_last_values_node_0, propose_node_1, from_node_key="left", to_node_key="input_list"))
        tot_graph.add_node(parallel_evaluation_node_1)
        tot_graph.add_edge(Edge(propose_node_1, parallel_evaluation_node_1, from_node_key="expressions", to_node_key="expressions"))
        tot_graph.add_edge(Edge(propose_node_1, parallel_evaluation_node_1, from_node_key="lefts", to_node_key="lefts"))
        tot_graph.add_edge(Edge(parallel_evaluation_node_1, filter_node_1, from_node_key="score", to_node_key="score"), order=i)
    find_last_values_nodes_1 = [find_last_values_op(1) for i in range(keep_top_n[1])]
    propose_nodes_2 = [propose_op(2) for i in range(keep_top_n[1])]
    parallel_evaluation_nodes_2 = [parallel_evaluation_op(2, LastStepValueOperation, i) for i in range(keep_top_n[1])]
    filter_node_2 = filter_op(2, 1)
    tot_graph.add_node(filter_node_2)
    for i, (find_last_values_node_1, propose_node_2, parallel_evaluation_node_2) in enumerate(zip(find_last_values_nodes_1, propose_nodes_2, parallel_evaluation_nodes_2)):
        tot_graph.add_node(find_last_values_node_1)
        tot_graph.add_node(propose_node_2)
        tot_graph.add_edge(Edge(filter_node_1, find_last_values_node_1, from_node_key="score", to_node_key="score"), order=i)
        # third layer
        tot_graph.add_edge(Edge(find_last_values_node_1, propose_node_2, from_node_key="left", to_node_key="input_list"))
        tot_graph.add_node(parallel_evaluation_node_2)
        tot_graph.add_edge(Edge(propose_node_2, parallel_evaluation_node_2, from_node_key="expressions", to_node_key="expressions"))
        tot_graph.add_edge(Edge(propose_node_2, parallel_evaluation_node_2, from_node_key="lefts", to_node_key="lefts"))
        tot_graph.add_edge(Edge(parallel_evaluation_node_2, filter_node_2, from_node_key="score", to_node_key="score"), order=i)
    find_last_answer_node = FindLastAnswerOperation()
    tot_graph.add_node(find_last_answer_node)
    tot_graph.add_edge(Edge(filter_node_2, find_last_answer_node, from_node_key="score", to_node_key="score"))
    tot_graph.add_edge(Edge(find_last_answer_node, end_node, from_node_key="answer", to_node_key="answer"))
    tot_graph.add_edge(Edge(find_last_answer_node, end_node, from_node_key="score", to_node_key="score"))

    
    process_measurement = ProcessMeasurement(graph_of_operations=tot_graph)

    tot_controller = Controller(
        graph_of_operations=tot_graph,
        scheduler=Scheduler.BFS,
        max_concurrent=max_concurrent,
        process_measurement=process_measurement
    )

    return tot_controller

if __name__ == "__main__":
    import asyncio
    cache = CacheContainer.from_persistent_cache_file(Path(__file__).parent / "output" / "dataset_cache.pkl", load_as_virtual_persistent_cache=True, skip_on_file_not_found=True)
    model = "gpt-5-mini"
    llm = OpenAIChat(model=model,
        cache=cache,
        request_price_per_token=OPENAI_PRICING[model]["request_price_per_token"],
        response_price_per_token=OPENAI_PRICING[model]["response_price_per_token"],
        openai_rate_limiter=OpenAIRateLimiter(
            rpm=OPENAI_PRICING[model]["RPM"],
            tpm=OPENAI_PRICING[model]["TPM"])
        )
    tot_controller = tot_controller(llm, num_examples=3, samples=[3, 2, 1], keep_top_n=[2, 2, 1], max_concurrent=20)
    tot_controller.graph_of_operations.snapshot.visualize(show_multiedges=False, show_values=True, show_keys=True, show_state=True)
    result, measurement = asyncio.run(tot_controller.execute(input={"input_list": [1, 2, 3, 4]}, debug_params={"raise_on_operation_failure": True, "visualize_intermediate_graphs": True}))
    print(result)
    print(measurement)
    tot_controller.graph_of_operations.snapshot.visualize(show_multiedges=False, show_values=True, show_keys=True, show_state=True)
    # tot_controller.graph_of_operations.snapshot.save_graphml(Path(__file__).parent / "output" / "tot_graph.graphml", include_values=True)
    cache.save_persistent_cache()