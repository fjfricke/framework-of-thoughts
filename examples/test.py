import logging

import asyncio

from llm_graph_optimizer.controller.controller import Controller
from llm_graph_optimizer.graph_of_operations.graph_of_operations import GraphOfOperations
from llm_graph_optimizer.graph_of_operations.types import Edge
from llm_graph_optimizer.language_models.cache.cache import CacheContainer
from llm_graph_optimizer.language_models.openai_chat import OpenAIChat
from llm_graph_optimizer.measurement.process_measurement import ProcessMeasurement
from llm_graph_optimizer.operations.base_operations.end import End
from llm_graph_optimizer.operations.base_operations.start import Start
from llm_graph_optimizer.operations.llm_operations import BaseLLMOperation

from llm_graph_optimizer.schedulers.schedulers import Scheduler

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('llm_graph_optimizer.language_models.abstract_language_model')
logger.setLevel(logging.DEBUG)

# Initialize the language model
cache = CacheContainer.from_persistent_cache_file("cache.pkl")
llm = OpenAIChat(model="gpt-4o", cache=cache)

# Initialize the start node
start_node = Start(
    input_types={"start": str}
)

# Initialize the LLM operation and node
def prompter(input):
    return f'Answer the following question: {input}'
def parser(x):
    return {"output": x}
llm_op = BaseLLMOperation(
    llm=llm,
    prompter=prompter,
    parser=parser,
    input_types={"input": str},
    output_types={"output": str},
    cache_seed=0
)

llm_op_2 = BaseLLMOperation(
    llm=llm,
    prompter=prompter,
    parser=parser,
    input_types={"input": str},
    output_types={"output": str},
    cache_seed=0
)


# Initialize the end node
end_node = End(
    input_types={"end": str, "end_2": str},
)

# Initialize the graph
graph = GraphOfOperations()

# Add the nodes to the graph
graph.add_node(start_node)
graph.add_node(llm_op)
graph.add_node(llm_op_2)
graph.add_node(end_node)

# Add the edges to the graph
graph.add_edge(Edge(start_node, llm_op, from_node_key="start", to_node_key="input"))
graph.add_edge(Edge(start_node, llm_op_2, from_node_key="start", to_node_key="input"))
graph.add_edge(Edge(llm_op, end_node, from_node_key="output", to_node_key="end"))
graph.add_edge(Edge(llm_op_2, end_node, from_node_key="output", to_node_key="end_2"))

#Initialize the measurement
process_measurement = ProcessMeasurement(graph)

# Initialize the scheduler
scheduler = Scheduler.BFS

# Initialize the controller
controller = Controller(graph, scheduler, max_concurrent=1, process_measurement=process_measurement)

# Run the controller
async def run_controller():
    answer, measurements = await controller.execute(input={"start": "Hello, world!"})
    print(answer)
    print(measurements)
    # cache.save_persistent_cache("cache.pkl")
    # graph.view_graph()

asyncio.run(run_controller())
