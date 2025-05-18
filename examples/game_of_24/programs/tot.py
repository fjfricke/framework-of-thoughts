from pathlib import Path
from examples.game_of_24.programs.operations.evaluate_and_choose_operation import EvaluateAndChooseOperation
from examples.game_of_24.programs.operations.get_all_predecessor_remainins_and_expressions import GetAllPredecessorRemainingsAndExpressions
from examples.game_of_24.programs.operations.get_predecessor_remaining import GetPredecessorRemaining
from examples.game_of_24.programs.operations.value_operation import ValueOperation
from examples.openai_pricing import OPENAI_PRICING
from llm_graph_optimizer.controller.controller import Controller
from llm_graph_optimizer.graph_of_operations.graph_of_operations import GraphOfOperations
from llm_graph_optimizer.graph_of_operations.types import Edge, ManyToOne
from llm_graph_optimizer.language_models.abstract_language_model import AbstractLanguageModel
from llm_graph_optimizer.language_models.cache.cache import CacheContainer
from llm_graph_optimizer.language_models.helpers.language_model_config import Config
from llm_graph_optimizer.language_models.helpers.openai_rate_limiter import OpenAIRateLimiter
from llm_graph_optimizer.language_models.openai_chat import OpenAIChat
from llm_graph_optimizer.measurement.process_measurement import ProcessMeasurement
from llm_graph_optimizer.operations.base_operations.end import End
from llm_graph_optimizer.operations.base_operations.filter_operation_with_edge_move import FilterOperationWithEdgeMove
from llm_graph_optimizer.operations.base_operations.score_operation import ScoreOperation
from llm_graph_optimizer.operations.base_operations.start import Start
from llm_graph_optimizer.operations.llm_operations.base_llm_operation import BaseLLMOperation
from examples.game_of_24.programs.prompter_parser import cot_parser, cot_prompt, propose_prompt, propose_parser, value_last_step_prompt, value_last_step_parser
from llm_graph_optimizer.schedulers.schedulers import Scheduler

def tot_controller(llm: AbstractLanguageModel, num_layers: int = 3, num_proposals: int = 5, num_value_samples: int = 3) -> Controller:
    start_node = Start(
        input_types={"input": list[int]}
    )

    propose_op = lambda: BaseLLMOperation(
        llm=llm,
        prompter=propose_prompt,
        parser=propose_parser,
        input_types={"input": list[int]},
        output_types={"expressions": list[str], "remainings": list[list[int]]},
        name="Propose",
    )

    final_propose_op = lambda: BaseLLMOperation(
        llm=llm,
        prompter=cot_prompt,
        parser=cot_parser,
        input_types={"input": list[int], "expressions": ManyToOne[str], "remainings": ManyToOne[list[int]]},
        output_types={"answer": str},
        name="FinalPropose",
    )

    value_op = lambda cache_seed: ValueOperation(
        llm=llm,
        cache_seed=cache_seed,
        name="Value",
    )

    value_last_step_op = lambda cache_seed: BaseLLMOperation(
        llm=llm,
        prompter=value_last_step_prompt,
        parser=value_last_step_parser,
        input_types={"input": list[int], "answer": str},
        output_types={"value": float | None},
        name="ValueLastStep",
        cache_seed=cache_seed
    )

    score_op = lambda: ScoreOperation(
        input_types={"values": ManyToOne[float | None]},
        output_type=float,
        scoring_function=lambda values: sum(v for v in values if v is not None),
        name="SumValues"
    )

    def filter_function(values: list[float]) -> list[int]:
        # Sort proposals by their values in descending order
        sorted_values = sorted(values, reverse=True)
        
        # Take the top num_proposals or all if less than num_proposals
        best_values = sorted_values[:min(len(sorted_values), num_proposals)]

        # get the indices of the best values
        best_indices = [values.index(value) for value in best_values]
        
        return best_indices

    filter_op = lambda: FilterOperationWithEdgeMove(
        input_types={"values": ManyToOne[float]},
        filter_function=filter_function,
        name="FilterNProposals"
    )

    def filter_function_last_layer(values: list[float], remainings: list[list[int]]) -> list[int]:
        filtered_indices = [i for i, remaining in enumerate(remainings) if len(remaining) == 1 and 24 in remaining]
        sorted_indices = sorted(filtered_indices, key=lambda i: values[i], reverse=True)[:num_proposals]
        
        return sorted_indices

    filter_op_last_layer = lambda: FilterOperationWithEdgeMove(
        input_types={"values": ManyToOne[float], "remainings": ManyToOne[list[int]]},
        filter_function=filter_function_last_layer,
    )

    # cot_op = lambda: BaseLLMOperation(
    #     llm=llm,
    #     prompter=cot_prompt,
    #     parser=cot_parser,
    #     input_types={"input": list[int], "expressions": list[str], "remainings": list[list[int]]},
    #     output_types={"answer": str},
    #     name="COT",
    # )

    evaluate_and_choose_op = lambda value_op, is_final_layer: EvaluateAndChooseOperation(
        value_op=value_op,
        score_op=score_op,
        is_final_layer=is_final_layer,
        name="EvaluateAndChoose"
    )

    # test_op = lambda: TestOperation(
    #     input_types={"score": float},
    #     output_types={"score": float},
    #     name="Test"
    # )

    get_predecessor_remaining_op = lambda: GetPredecessorRemaining(
        name="GetPredecessorRemaining"
    )

    get_all_predecessor_remainings_and_expressions_op = lambda: GetAllPredecessorRemainingsAndExpressions(
        name="GetAllPredecessorRemainingsAndExpressions"
    )

    end_node = End(
        input_types={"values": ManyToOne[float]}
    )
    
    graph = GraphOfOperations()
    graph.add_node(start_node)
    graph.add_node(end_node)

    # layer 1

    propose_node = propose_op()
    graph.add_node(propose_node)
    graph.add_edge(Edge(start_node, propose_node, "input", "input"))
    value_and_choose_node = evaluate_and_choose_op(value_op, False)
    graph.add_node(value_and_choose_node)
    graph.add_edge(Edge(propose_node, value_and_choose_node, "expressions", "expressions"))
    graph.add_edge(Edge(propose_node, value_and_choose_node, "remainings", "remainings"))
    filter_node = filter_op()
    graph.add_node(filter_node)
    graph.add_edge(Edge(value_and_choose_node, filter_node, "values", "values"), order=100)

    for layer in range(1, num_layers):
        if layer == num_layers-1:
            next_filter_node = filter_op_last_layer()
        else:
            next_filter_node = filter_op()
        graph.add_node(next_filter_node)
        for i in range(num_proposals):
            get_remaining_node = get_predecessor_remaining_op()
            graph.add_node(get_remaining_node)
            graph.add_edge(Edge(filter_node, get_remaining_node, "score", "score"), order=i)
            propose_node = propose_op()
            graph.add_node(propose_node)
            graph.add_edge(Edge(get_remaining_node, propose_node, "remaining", "input"))
            value_and_choose_node = evaluate_and_choose_op(value_op, is_final_layer=layer == num_layers-1)
            graph.add_node(value_and_choose_node)
            graph.add_edge(Edge(propose_node, value_and_choose_node, "expressions", "expressions"))
            graph.add_edge(Edge(propose_node, value_and_choose_node, "remainings", "remainings"))
            graph.add_edge(Edge(value_and_choose_node, next_filter_node, "values", "values"), order=100*i)
            if layer == num_layers-1:
                graph.add_edge(Edge(value_and_choose_node, next_filter_node, "remainings", "remainings"))
        filter_node = next_filter_node

    for i in range(num_proposals):
        get_all_predecessor_remainings_and_expressions_node = get_all_predecessor_remainings_and_expressions_op()
        graph.add_node(get_all_predecessor_remainings_and_expressions_node)
        graph.add_edge(Edge(filter_node, get_all_predecessor_remainings_and_expressions_node, "score", "score"), order=i)
        final_propose_node = final_propose_op()
        graph.add_node(final_propose_node)
        graph.add_edge(Edge(get_all_predecessor_remainings_and_expressions_node, final_propose_node, "remaining", "remainings"))
        graph.add_edge(Edge(get_all_predecessor_remainings_and_expressions_node, final_propose_node, "expression", "expressions"))
        final_value_node = value_last_step_op(cache_seed=i)
        graph.add_node(final_value_node)
        graph.add_edge(Edge(start_node, final_value_node, "input", "input"))
        graph.add_edge(Edge(final_propose_node, final_value_node, "answer", "answer"))
        graph.add_edge(Edge(final_value_node, end_node, "value", "values"), order=i)

    controller = Controller(
        graph_of_operations=graph,
        scheduler=Scheduler.BFS,
        max_concurrent=1,
        process_measurement=ProcessMeasurement(graph_of_operations=graph)
    )

    return controller
        
        
if __name__ == "__main__":
    import asyncio

    cache = CacheContainer.from_persistent_cache_file(
        file_path=Path(__file__).parent.parent / "output" / "cache.pkl",
        load_as_virtual_persistent_cache=True,
        skip_on_file_not_found=True
    )
    model = "gpt-3.5-turbo"
    llm = OpenAIChat(
        model=model,
        config=Config(temperature=1.0),
        cache=cache,
        request_price_per_token=OPENAI_PRICING[model]["request_price_per_token"],
        response_price_per_token=OPENAI_PRICING[model]["response_price_per_token"],
        openai_rate_limiter=OpenAIRateLimiter(
            rpm=OPENAI_PRICING[model]["RPM"],
            tpm=OPENAI_PRICING[model]["TPM"]
        )
    )
    controller = tot_controller(llm=llm, num_layers=3, num_proposals=2, num_value_samples=2)
    # example = [4, 9, 10, 13]
    example = [12, 2, 1, 0]
    result, measurement = asyncio.run(controller.execute(input={"input": example}, debug_params={"visualize_intermediate_graphs": False, "raise_on_operation_failure": False}))
    cache.save_persistent_cache(Path(__file__).parent.parent / "output" / "cache.pkl")