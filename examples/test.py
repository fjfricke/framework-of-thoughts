import logging

import asyncio

from llm_graph_optimizer.controller.controller import Controller
from llm_graph_optimizer.graph_of_operations.graph_of_operations import GraphOfOperations
from llm_graph_optimizer.language_models.openai_chat import OpenAIChat
from llm_graph_optimizer.operations.end import End
from llm_graph_optimizer.operations.llm_operations import BaseLLMOperation
from llm_graph_optimizer.operations.start import Start
from llm_graph_optimizer.schedulers.schedulers import Scheduler

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('llm_graph_optimizer')
logger.setLevel(logging.DEBUG)

# Initialize the language model
llm = OpenAIChat(model="gpt-4o")

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
)

# Initialize the end node
end_node = End(
    input_types={"end": str}
)

# Initialize the graph
graph = GraphOfOperations()

# Add the nodes to the graph
graph.add_node(start_node)
graph.add_node(llm_op)
graph.add_node(end_node)

# Add the edges to the graph
graph.add_edge(start_node, llm_op, from_node_key="start", to_node_key="input")
graph.add_edge(llm_op, end_node, from_node_key="output", to_node_key="end")

# Initialize the scheduler
scheduler = Scheduler.BFS

# Initialize the controller
controller = Controller(graph, scheduler, max_concurrent=3)

# Run the controller
async def run_controller():
    answer = await controller.execute(input={"start": "Hello, world!"})
    print(answer)
    # graph.view_graph()

asyncio.run(run_controller())
