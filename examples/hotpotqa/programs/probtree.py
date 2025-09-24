import logging
from pathlib import Path

from examples.hotpotqa.programs.operations.reasoning.child_aggregate import ChildAggregateReasoning
from examples.hotpotqa.programs.operations.reasoning.closed_book import ClosedBookReasoning
from examples.hotpotqa.programs.operations.reasoning.filter import filter_function
from examples.hotpotqa.programs.operations.reasoning.open_book import OpenBookReasoning, get_retriever
from examples.hotpotqa.programs.operations.unterstanding.understanding import UnderstandingGraphUpdating
from examples.hotpotqa.programs.operations.unterstanding.prompter_parser import understanding_parser, understanding_prompt_hotpotqa, understanding_prompt_musique

from examples.hotpotqa.programs.operations.reasoning.open_book import prompter_hotpotqa as open_book_prompt_hotpotqa
from examples.hotpotqa.programs.operations.reasoning.open_book import prompter_musique as open_book_prompt_musique

from examples.hotpotqa.programs.operations.reasoning.closed_book import prompter_hotpotqa as closed_book_prompt_hotpotqa
from examples.hotpotqa.programs.operations.reasoning.closed_book import prompter_musique as closed_book_prompt_musique

from examples.openai_pricing import OPENAI_PRICING
from llm_graph_optimizer.controller.controller import Controller
from llm_graph_optimizer.graph_of_operations.graph_of_operations import GraphOfOperations
from llm_graph_optimizer.graph_of_operations.types import Edge, ManyToOne
from llm_graph_optimizer.language_models.cache.cache import CacheContainer
from llm_graph_optimizer.language_models.helpers.language_model_config import Config
from llm_graph_optimizer.language_models.openai_chat_with_logprobs import OpenAIChatWithLogprobs
from llm_graph_optimizer.measurement.process_measurement import ProcessMeasurement
from llm_graph_optimizer.operations.base_operations.end import End
from llm_graph_optimizer.operations.base_operations.filter_operation import FilterOperation
from llm_graph_optimizer.operations.base_operations.start import Start
from llm_graph_optimizer.operations.llm_operations.llm_operation_with_logprobs import LLMOperationWithLogprobs
from llm_graph_optimizer.schedulers.schedulers import Scheduler

retriever = get_retriever(Path().resolve() / "examples" / "hotpotqa" / "dataset" / "HotpotQA" / "wikipedia_index_bm25")
cache = CacheContainer.from_persistent_cache_file(Path(__file__).parent.parent / "output" / "cache.pkl", skip_on_file_not_found=True, load_as_virtual_persistent_cache=True)

def probtree_controller(llm: OpenAIChatWithLogprobs, max_concurrent: int = 1, dataset: str = "hotpotqa") -> Controller:
    """
    This function creates a controller for the ProbTree algorithm.
    It takes a language model, a number of retrieved documents, and optional scaling and shifting factors. Do not create the LLM inside the controller generator function. It needs to be shared across controllers to use the same shared cache.
    It returns a controller that can be used to execute the ProbTree algorithm.
    """

    if dataset == "hotpotqa":
        understanding_prompter = understanding_prompt_hotpotqa
        open_book_prompter = open_book_prompt_hotpotqa
        closed_book_prompter = closed_book_prompt_hotpotqa
    elif dataset == "musique":
        understanding_prompter = understanding_prompt_musique
        open_book_prompter = open_book_prompt_musique
        closed_book_prompter = closed_book_prompt_musique
    else:
        raise ValueError(f"Dataset {dataset} not supported")

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
        prompter=understanding_prompter,
        parser=understanding_parser,
        input_types={"question": str},
        output_types={"hqdt": dict},
        name="GenerateTree"
    )

    open_book_op = lambda: OpenBookReasoning(
        llm=llm,
        retriever=retriever,
        k=5,
        name="OB",
        prompter=open_book_prompter
    )
    
    closed_book_op = lambda: ClosedBookReasoning(llm=llm, name="CB", prompter=closed_book_prompter)

    child_aggregate_op = lambda: ChildAggregateReasoning(llm=llm, name="CA")

    filter_op = lambda: FilterOperation(output_types={"answer": str, "decomposition_score": float}, input_types={"answers": ManyToOne[str], "decomposition_scores": ManyToOne[float]}, filter_function=filter_function, name="Filter")

    understanding_op = lambda: UnderstandingGraphUpdating(
        open_book_op=open_book_op,
        closed_book_op=closed_book_op,
        child_aggregate_op=child_aggregate_op,
        understanding_op=understanding_op,
        filter_op=filter_op,
        name="BuildTree"
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

    process_measurement = ProcessMeasurement(graph_of_operations=probtree_graph)

    controller = Controller(
        graph_of_operations=probtree_graph,
        scheduler=Scheduler.BFS,
        max_concurrent=max_concurrent,
        process_measurement=process_measurement,
        store_intermediate_snapshots=True
    )

    return controller

if __name__ == "__main__":
    import asyncio

    logging.getLogger().setLevel(logging.CRITICAL)
    logging.getLogger('llm_graph_optimizer.controller.controller').setLevel(logging.DEBUG)

    model = "gpt-4o-mini"
    llm = OpenAIChatWithLogprobs(model=model, config=Config(temperature=0.0), request_price_per_token=OPENAI_PRICING[model]["request_price_per_token"], response_price_per_token=OPENAI_PRICING[model]["response_price_per_token"])
    controller = probtree_controller(llm=llm)

    controller.graph_of_operations.snapshot.visualize(show_multiedges=False, show_values=True, show_keys=True, show_state=True)  # Visualizes the operation graph before execution

    output, process_measurement = asyncio.run(controller.execute(input={"question": "Are both Superdrag and Collective Soul rock bands?"}, debug_params={"raise_on_operation_failure": True, "visualize_intermediate_graphs": True}))

    snapshot_graph = controller.graph_of_operations.snapshot
    snapshot_graph.visualize(show_multiedges=False, show_values=True, show_keys=True, show_state=True)

    print(output)
    print(process_measurement)