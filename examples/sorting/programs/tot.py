# Initialize the language model
from llm_graph_optimizer.controller.controller import Controller
from llm_graph_optimizer.graph_of_operations.graph_of_operations import GraphOfOperations
from llm_graph_optimizer.language_models.helpers.language_model_config import Config
from llm_graph_optimizer.operations.filter_operation import FilterOperation
from llm_graph_optimizer.operations.pack_unpack_operations import PackOperation
from llm_graph_optimizer.schedulers.schedulers import Scheduler
from examples.sorting.programs.prompter_parser import filter_function, generate_prompt, generate_prompt_cot, generate_parser, scoring_function, tot_improve_prompt
from llm_graph_optimizer.language_models.openai_chat import OpenAIChat
from llm_graph_optimizer.operations.llm_operations import BaseLLMOperation
from llm_graph_optimizer.operations.score_operation import ScoreOperation
from llm_graph_optimizer.operations.start import Start
from llm_graph_optimizer.operations.end import End


def tot_controller(num_branches: int = 20, improvement_levels: int = 2) -> Controller:

    # llm = OpenAIChat(model="gpt-4o")
    llm_with_large_temp = OpenAIChat(model="gpt-3.5-turbo", config=Config(temperature=1.9))
    # Initialize the start node
    start_node = Start(
        input_types={"input_list": list[int], "expected_output": list[int]}
    )

    generate_op = lambda: BaseLLMOperation(
        llm=llm_with_large_temp,
        prompter=generate_prompt,
        parser=generate_parser,
        use_cache=False,
        input_types={"input_list": list[int]},
        output_types={"output": list[int]}
    )

    improvement_op = lambda: BaseLLMOperation(
        llm=llm_with_large_temp,
        prompter=tot_improve_prompt,
        parser=generate_parser,
        use_cache=False,
        input_types={"input_list": list[int], "incorrectly_sorted": list[int]},
        output_types={"output": list[int]}
    )

    score_op = lambda: ScoreOperation(
        input_types={"output": list[int], "expected_output": list[int]},
        output_type=int,
        scoring_function=scoring_function
    )

    pack_op = lambda: PackOperation(
        input_types={"output": list[int], "score": int},
        output_key="packed"
    )

    keep_best_op = lambda: FilterOperation(
        output_types={"output": list[int], "score": int},
        filter_function=filter_function,
        length=num_branches
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
        generate_node = generate_op()
        tot_graph.add_node(generate_node)
        score_node = score_op()
        tot_graph.add_node(score_node)
        pack_node = pack_op()
        tot_graph.add_node(pack_node)
        tot_graph.add_edge(start_node, generate_node, "input_list", "input_list")
        tot_graph.add_edge(generate_node, score_node, "output", "output")
        tot_graph.add_edge(start_node, score_node, "expected_output", "expected_output")
        tot_graph.add_edge(score_node, pack_node, "score", "score")
        tot_graph.add_edge(generate_node, pack_node, "output", "output")
        tot_graph.add_edge(pack_node, keep_best_nodes[0], "packed", i)
    for i in range(improvement_levels):
        for j in range(num_branches):
            improvement_node = improvement_op()
            tot_graph.add_node(improvement_node)
            score_node = score_op()
            tot_graph.add_node(score_node)
            pack_node = pack_op()
            tot_graph.add_node(pack_node)
            tot_graph.add_edge(keep_best_nodes[i], improvement_node, "output", "incorrectly_sorted")
            tot_graph.add_edge(start_node, improvement_node, "input_list", "input_list")
            tot_graph.add_edge(improvement_node, score_node, "output", "output")
            tot_graph.add_edge(start_node, score_node, "expected_output", "expected_output")
            tot_graph.add_edge(score_node, pack_node, "score", "score")
            tot_graph.add_edge(improvement_node, pack_node, "output", "output")
            tot_graph.add_edge(pack_node, keep_best_nodes[i + 1], "packed", j)

    tot_graph.add_node(end_node)
    tot_graph.add_edge(keep_best_nodes[-1], end_node, "output", "output")
    tot_graph.add_edge(keep_best_nodes[-1], end_node, "score", "score")
    tot_graph.add_edge(start_node, end_node, "expected_output", "expected_output")


    tot_graph.view_graph(show_keys=True, use_pyvis=True)
    # Initialize the controller
    tot_controller = Controller(
        graph_of_operations=tot_graph,
        scheduler=Scheduler.BFS,
        max_concurrent=1
    )

    return tot_controller

if __name__ == "__main__":
    tot_controller(num_branches=2, improvement_levels=2)