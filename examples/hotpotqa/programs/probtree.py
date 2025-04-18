import logging
import os
from pathlib import Path

from examples.hotpotqa.programs.operations.reasoning.child_aggregate import ChildAggregateReasoning
from examples.hotpotqa.programs.operations.reasoning.closed_book import ClosedBookReasoning
from examples.hotpotqa.programs.operations.reasoning.open_book import OpenBookReasoning, get_retriever
from examples.hotpotqa.programs.operations.understanding import UnderstandingGraphUpdating
from examples.hotpotqa.programs.prompter_parser import understanding_parser, understanding_prompt
from llm_graph_optimizer.controller.controller import Controller
from llm_graph_optimizer.graph_of_operations.graph_of_operations import GraphOfOperations
from llm_graph_optimizer.graph_of_operations.types import Edge, StateNotSet
from llm_graph_optimizer.language_models.helpers.language_model_config import Config
from llm_graph_optimizer.language_models.openai_chat import OpenAIChat
from llm_graph_optimizer.operations.end import End
from llm_graph_optimizer.operations.llm_operations.llm_operation_with_logprobs import LLMOperationWithLogprobs
from llm_graph_optimizer.operations.start import Start
from llm_graph_optimizer.schedulers.schedulers import Scheduler

logging.getLogger().setLevel(logging.CRITICAL)
logging.getLogger('llm_graph_optimizer.controller.controller').setLevel(logging.DEBUG)

def probtree_controller() -> Controller:
    llm = OpenAIChat(model="gpt-4o", config=Config(temperature=0.0))

    start_node = Start(
        input_types={"question": str},
        output_types={"question": str, "question_decomposition_score": float},
        static_outputs={"question_decomposition_score": 0.0}
    )

    end_node = End(
        input_types={"answer": str, "decomposition_score": float},
    )

    generate_hqdt_node = LLMOperationWithLogprobs(
        llm=llm,
        prompter=understanding_prompt,
        parser=understanding_parser,
        input_types={"question": str},
        output_types={"hqdt": dict},
        name="GenerateHQDT"
    )

    open_book_op = lambda: OpenBookReasoning(
        llm=llm,
        retriever=get_retriever(Path(os.getcwd()) / "examples" / "hotpotqa" / "dataset" / "HotpotQA" / "wikipedia_index_bm25"),
        k=5)
    
    closed_book_op = lambda: ClosedBookReasoning(llm=llm)

    child_aggregate_op = lambda: ChildAggregateReasoning(llm=llm)

    understanding_op = lambda: UnderstandingGraphUpdating(
        open_book_op=open_book_op,
        closed_book_op=closed_book_op,
        child_aggregate_op=child_aggregate_op,
        understanding_op=understanding_op
    )

    understanding_node = understanding_op()

    probtree_graph = GraphOfOperations()
    probtree_graph.add_node(start_node)
    probtree_graph.add_node(generate_hqdt_node)
    probtree_graph.add_node(understanding_node)
    probtree_graph.add_node(end_node)
    probtree_graph.add_edge(Edge(start_node, generate_hqdt_node, "question", "question"))
    
    probtree_graph.add_edge(Edge(start_node, understanding_node, "question", "question"))
    probtree_graph.add_edge(Edge(start_node, understanding_node, "question_decomposition_score", "question_decomposition_score"))

    probtree_graph.add_edge(Edge(generate_hqdt_node, understanding_node, "hqdt", "hqdt"))

    probtree_graph.add_edge(Edge(understanding_node, end_node, "answer", "answer"))
    probtree_graph.add_edge(Edge(understanding_node, end_node, "decomposition_score", "decomposition_score"))

    controller = Controller(
        graph_of_operations=probtree_graph,
        scheduler=Scheduler.BFS,
        max_concurrent=5,
    )

    return controller

if __name__ == "__main__":
    controller = probtree_controller()
    import asyncio
    # output = asyncio.run(controller.execute({"question": "What is 1+1?"}))
    output = asyncio.run(controller.execute({"question": "What is the combined population of the population-wise biggest 2 neighbour country of the largest country in Europe by capita?"}))
    controller.graph_of_operations.view_graph_debug(output_name="probtree_debug_2.html")
    snapshot_graph = controller.graph_of_operations.snapshot
    save_path = Path(os.getcwd()) / "examples" / "hotpotqa" / "output"
    snapshot_graph.save(save_path / "probtree_debug.pkl")
    print(output)
