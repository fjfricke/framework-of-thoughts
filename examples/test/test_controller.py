import asyncio

from llm_graph_optimizer.controller.controller import Controller
from llm_graph_optimizer.graph_of_operations.graph_of_operations import GraphOfOperations
from llm_graph_optimizer.graph_of_operations.types import Edge, ManyToOne
from llm_graph_optimizer.language_models.cache.cache import CacheContainer
from llm_graph_optimizer.language_models.helpers.language_model_config import Config
from llm_graph_optimizer.language_models.openai_chat import OpenAIChat
from llm_graph_optimizer.measurement.process_measurement import ProcessMeasurement
from llm_graph_optimizer.operations.base_operations.end import End
from llm_graph_optimizer.operations.base_operations.filter_operation import FilterOperation
from llm_graph_optimizer.operations.base_operations.start import Start
from llm_graph_optimizer.operations.llm_operations import BaseLLMOperation

from llm_graph_optimizer.schedulers.schedulers import Scheduler

# Initialize the language model


def controller(cache: CacheContainer, number_of_llm_nodes: int = 3) -> Controller:
    llm = OpenAIChat(model="gpt-3.5-turbo", cache=cache, config=Config(temperature=1.0))

    # Initialize the start node
    start_node = Start(
        input_types={"start": str}
    )

    # Initialize the LLM operation and node
    def prompter(input):
        return f'Answer the following math problem and only answer with the number. Do not think about it, just answer: {input}'
    def parser(x):
        try:
            x_as_number = int(x)
        except ValueError:
            x_as_number = None
        return {"output": x_as_number}
    llm_op = lambda cache_seed: BaseLLMOperation(
        llm=llm,
        prompter=prompter,
        parser=parser,
        input_types={"input": str},
        output_types={"output": int | None},
        cache_seed=cache_seed
    )

    llm_nodes = [llm_op(i) for i in range(number_of_llm_nodes)]

    filter_node = FilterOperation(
        input_types={"outputs": ManyToOne[int | None]},
        output_types={"output": int | None},
        filter_function=lambda outputs: {"output": max(set(outputs), key=outputs.count)}
    )


    # Initialize the end node
    end_node = End(
        input_types={"end": int | None},
    )

    # Initialize the graph
    graph = GraphOfOperations()

    # Add the nodes to the graph
    graph.add_node(start_node)
    graph.add_node(filter_node)
    for i, node in enumerate(llm_nodes):
        graph.add_node(node)
        graph.add_edge(Edge(start_node, node, from_node_key="start", to_node_key="input"))
        graph.add_edge(Edge(node, filter_node, from_node_key="output", to_node_key="outputs"), order=i)
    graph.add_node(end_node)
    graph.add_edge(Edge(filter_node, end_node, from_node_key="output", to_node_key="end"))

    #Initialize the measurement
    process_measurement = ProcessMeasurement(graph)

    # Initialize the scheduler
    scheduler = Scheduler.BFS

    # Initialize the controller
    controller = Controller(graph, scheduler, max_concurrent=1, process_measurement=process_measurement)
    return controller


if __name__ == "__main__":
    cache = CacheContainer()
    controller: Controller = controller(cache, number_of_llm_nodes=5)
    # Run the controller
    async def run_controller():
        answer, measurements = await controller.execute(input={"start": "762**2"})
        print(answer)
        print(measurements)
        controller.graph_of_operations.snapshot.visualize(show_multiedges=False, show_values=True, show_keys=True, show_state=True)

    asyncio.run(run_controller())
