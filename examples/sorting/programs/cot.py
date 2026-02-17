from llm_graph_optimizer.controller.controller import Controller
from llm_graph_optimizer.graph_of_operations.graph_of_operations import GraphOfOperations
from llm_graph_optimizer.measurement.process_measurement import ProcessMeasurement
from llm_graph_optimizer.schedulers.schedulers import Scheduler
from examples.sorting.programs.prompter_parser import generate_prompt_cot, generate_parser, scoring_function
from llm_graph_optimizer.language_models.openai_chat import OpenAIChat
from llm_graph_optimizer.operations.llm_operations.base_llm_operation import BaseLLMOperation
from llm_graph_optimizer.operations.base_operations.score_operation import ScoreOperation
from llm_graph_optimizer.operations.base_operations.start import Start
from llm_graph_optimizer.operations.base_operations.end import End
from llm_graph_optimizer.graph_of_operations.types import Edge


def cot_controller() -> Controller:

    llm = OpenAIChat(model="gpt-3.5-turbo")

    # Initialize the start node
    start_node = Start(
        input_types={"input_list": list[int], "expected_output": list[int]}
    )

    generate_cot_node = BaseLLMOperation(
        llm=llm,
        prompter=generate_prompt_cot,
        parser=generate_parser,
        input_types={"input_list": list[int]},
        output_types={"output": list[int]}
    )

    # Initialize the score operation and node
    score_node = ScoreOperation(
        input_types={"output": list[int], "expected_output": list[int]},
        output_type=int,
        scoring_function=scoring_function
    )

    # Initialize the end node
    end_node = End(
        input_types={"score": int, "output": list[int], "expected_output": list[int]}
    )

    # Initialize the graph of operations for CoT
    cot_graph = GraphOfOperations()
    cot_graph.add_node(start_node)
    cot_graph.add_node(generate_cot_node)
    cot_graph.add_node(score_node)
    cot_graph.add_node(end_node)

    cot_graph.add_edge(Edge(start_node, generate_cot_node, "input_list", "input_list"))
    cot_graph.add_edge(Edge(generate_cot_node, score_node, "output", "output"))
    cot_graph.add_edge(Edge(start_node, score_node, "expected_output", "expected_output"))
    cot_graph.add_edge(Edge(score_node, end_node, "score", "score"))
    cot_graph.add_edge(Edge(generate_cot_node, end_node, "output", "output"))
    cot_graph.add_edge(Edge(start_node, end_node, "expected_output", "expected_output"))

    process_measurement = ProcessMeasurement(graph_of_operations=cot_graph)

    # Initialize the controller
    cot_controller = Controller(
        graph_of_operations=cot_graph,
        scheduler=Scheduler.BFS,
        max_concurrent=1,
        process_measurement=process_measurement
    )

    return cot_controller