import os
import asyncio

from llm_graph_optimizer.controller.controller import Controller
from llm_graph_optimizer.graph_of_operations.graph_of_operations import GraphOfOperations
from llm_graph_optimizer.language_models.openai_chat import OpenAIChat
from llm_graph_optimizer.operations.end import End
from llm_graph_optimizer.operations.llm_operation import LLMOperation
from llm_graph_optimizer.operations.start import Start
from llm_graph_optimizer.schedulers.schedulers import Scheduler

def prompter(x):
    return f'Answer the following question: {x["input"]}'
def parser(x):
    return {"output": x}

llm = OpenAIChat(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))

start_node = Start(
    input_types={"start": str}
)

llm_op = LLMOperation(
    llm=llm,
    prompter=prompter,
    parser=parser,
    input_types={"input": str},
    output_types={"output": str},
)

end_node = End(
    input_types={"end": str}
)
graph = GraphOfOperations()
graph.add_node(start_node)
graph.add_node(llm_op)
graph.add_node(end_node)
graph.add_edge(start_node, llm_op, from_node_key="start", to_node_key="input")
graph.add_edge(llm_op, end_node, from_node_key="output", to_node_key="end")

graph.start_node = start_node
graph.end_node = end_node

scheduler = Scheduler.BFS
controller = Controller(graph, scheduler, max_concurrent=3)

async def run_controller():
    answer = await controller.execute(input={"start": "Hello, world!"})
    print(answer)
    

asyncio.run(run_controller())
