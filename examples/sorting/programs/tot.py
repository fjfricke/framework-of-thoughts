# Initialize the language model
from llm_graph_optimizer.controller.controller import Controller
from llm_graph_optimizer.graph_of_operations.graph_of_operations import GraphOfOperations
from llm_graph_optimizer.language_models.abstract_language_model import AbstractLanguageModel
from llm_graph_optimizer.language_models.openai_chat import OpenAIChat
from llm_graph_optimizer.measurement.process_measurement import ProcessMeasurement
from llm_graph_optimizer.operations.base_operations.filter_operation import FilterOperation
from llm_graph_optimizer.schedulers.schedulers import Scheduler
from examples.sorting.programs.prompter_parser import filter_function, generate_prompt, generate_parser, scoring_function, tot_improve_prompt
from llm_graph_optimizer.operations.llm_operations.base_llm_operation import BaseLLMOperation
from llm_graph_optimizer.operations.base_operations.score_operation import ScoreOperation
from llm_graph_optimizer.operations.base_operations.start import Start
from llm_graph_optimizer.operations.base_operations.end import End
from llm_graph_optimizer.graph_of_operations.types import Edge, ManyToOne


def tot_controller(llm: AbstractLanguageModel, num_branches: int = 20, improvement_levels: int = 2, max_concurrent: int = 5) -> Controller:

    # llm = OpenAIChat(model="gpt-4o")
    # Initialize the start node
    start_node = Start(
        input_types={"input_list": list[int], "expected_output": list[int]}
    )

    generate_op = lambda cache_seed: BaseLLMOperation(
        llm=llm,
        prompter=generate_prompt,
        parser=generate_parser,
        input_types={"input_list": list[int]},
        output_types={"output": list[int]},
        name="Generate",
        cache_seed=cache_seed
    )

    improvement_op = lambda cache_seed: BaseLLMOperation(
        llm=llm,
        prompter=tot_improve_prompt,
        parser=generate_parser,
        input_types={"input_list": list[int], "incorrectly_sorted": list[int]},
        output_types={"output": list[int]},
        name="Improve",
        cache_seed=cache_seed
    )

    score_op = lambda: ScoreOperation(
        input_types={"output": list[int], "expected_output": list[int]},
        output_type=int,
        scoring_function=scoring_function,
        name="Score"
    )

    keep_best_op = lambda: FilterOperation(
        output_types={"output": list[int], "score": int},
        input_types={"outputs": ManyToOne[list[int]], "scores": ManyToOne[int]},
        filter_function=filter_function,
        name="Filter"
    )

    # Initialize the end node
    end_node = End(
        input_types={"score": int, "output": list[int], "expected_output": list[int]}
    )

    # Initialize the graph of operations for IO

    tot_graph = GraphOfOperations()
    tot_graph.add_node(start_node)
    keep_best_nodes = [keep_best_op() for _ in range(improvement_levels + 1)]
    [tot_graph.add_node(keep_best_nodes[i]) for i in range(improvement_levels + 1)]
    for i in range(num_branches):
        generate_node = generate_op(cache_seed=i)
        tot_graph.add_node(generate_node)
        score_node = score_op()
        tot_graph.add_node(score_node)
        tot_graph.add_edge(Edge(start_node, generate_node, "input_list", "input_list"))
        tot_graph.add_edge(Edge(generate_node, score_node, "output", "output"))
        tot_graph.add_edge(Edge(start_node, score_node, "expected_output", "expected_output"))
        tot_graph.add_edge(Edge(score_node, keep_best_nodes[0], "score", "scores"), order=i)
        tot_graph.add_edge(Edge(generate_node, keep_best_nodes[0], "output", "outputs"), order=i)
    for i in range(improvement_levels):
        for j in range(num_branches):
            improvement_node = improvement_op(cache_seed=i * num_branches + j)
            tot_graph.add_node(improvement_node)
            score_node = score_op()
            tot_graph.add_node(score_node)
            tot_graph.add_edge(Edge(keep_best_nodes[i], improvement_node, "output", "incorrectly_sorted"))
            tot_graph.add_edge(Edge(start_node, improvement_node, "input_list", "input_list"))
            tot_graph.add_edge(Edge(improvement_node, score_node, "output", "output"))
            tot_graph.add_edge(Edge(start_node, score_node, "expected_output", "expected_output"))
            tot_graph.add_edge(Edge(score_node, keep_best_nodes[i + 1], "score", "scores"), order=j)
            tot_graph.add_edge(Edge(improvement_node, keep_best_nodes[i + 1], "output", "outputs"), order=j)

    tot_graph.add_node(end_node)
    tot_graph.add_edge(Edge(keep_best_nodes[-1], end_node, "output", "output"))
    tot_graph.add_edge(Edge(keep_best_nodes[-1], end_node, "score", "score"))
    tot_graph.add_edge(Edge(start_node, end_node, "expected_output", "expected_output"))

    process_measurement = ProcessMeasurement(graph_of_operations=tot_graph)
    # Initialize the controller
    tot_controller = Controller(
        graph_of_operations=tot_graph,
        scheduler=Scheduler.BFS,
        max_concurrent=max_concurrent,
        process_measurement=process_measurement
    )

    return tot_controller

if __name__ == "__main__":
    import asyncio
    llm = OpenAIChat(model="gpt-3.5-turbo")
    example = [0, 9, 4, 2, 2, 0, 5, 1]
    expected = sorted(example)
    controller = tot_controller(llm, num_branches=2, improvement_levels=2)
    controller.graph_of_operations.snapshot.visualize(show_multiedges=False, show_values=True, show_keys=True, show_state=False, show_keys_on_arrows=False)
    result, measurement = asyncio.run(controller.execute(input={"input_list": example, "expected_output": expected}))
    print(result)
    print(measurement)
    controller.graph_of_operations.snapshot.visualize(show_multiedges=False, show_values=True, show_keys=True, show_state=True)