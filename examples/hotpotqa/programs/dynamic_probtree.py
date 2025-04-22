from pathlib import Path
from examples.hotpotqa.programs.operations.one_layer_understanding.one_layer_reasoning import OneLayerReasoning
from examples.hotpotqa.programs.operations.one_layer_understanding.one_layer_understanding import OneLayerUnderstanding
from examples.hotpotqa.programs.operations.reasoning.child_aggregate import ChildAggregateReasoning
from examples.hotpotqa.programs.operations.reasoning.closed_book import ClosedBookReasoning
from examples.hotpotqa.programs.operations.reasoning.filter import filter_function
from examples.hotpotqa.programs.operations.reasoning.open_book import OpenBookReasoning, get_retriever
from llm_graph_optimizer.controller.controller import Controller
from llm_graph_optimizer.graph_of_operations.graph_of_operations import GraphOfOperations
from llm_graph_optimizer.graph_of_operations.types import Edge, ManyToOne
from llm_graph_optimizer.language_models.helpers.language_model_config import Config
from llm_graph_optimizer.language_models.openai_chat_with_logprobs import OpenAIChatWithLogprobs
from llm_graph_optimizer.measurement.process_measurement import ProcessMeasurement
from llm_graph_optimizer.operations.base_operations.end import End
from llm_graph_optimizer.operations.base_operations.filter_operation import FilterOperation
from llm_graph_optimizer.operations.base_operations.start import Start
from llm_graph_optimizer.operations.llm_operations.llm_operation_with_logprobs import LLMOperationWithLogprobs
from examples.hotpotqa.programs.operations.one_layer_understanding.branch import BranchOperation, prompter as branch_prompt, parser as branch_parser
from examples.hotpotqa.programs.operations.one_layer_understanding.decompose import prompter as decompose_prompt, parser as decompose_parser
from llm_graph_optimizer.schedulers.schedulers import Scheduler

import logging
logging.getLogger().setLevel(logging.CRITICAL)
logging.getLogger('llm_graph_optimizer.controller.controller').setLevel(logging.DEBUG)

def dynamic_probtree_controller(max_depth: int, min_branch_certainty_threshold: float) -> Controller:
    llm = OpenAIChatWithLogprobs(model="gpt-4o", config=Config(temperature=0.0))

    start_node = Start(
        input_types={"question": str},
        output_types={"question": str, "max_depth": int, "min_branch_certainty_threshold": float, "question_decomposition_score": float},
        static_outputs={"max_depth": max_depth, "min_branch_certainty_threshold": min_branch_certainty_threshold, "question_decomposition_score": 0.0}
    )

    end_node = End(
        input_types={"answer": str, "decomposition_score": float},
    )
    
    branch_op = BranchOperation.factory(
        llm=llm,
        prompter=branch_prompt,
        parser=branch_parser,
        input_types={"question": str, "dependency_answers": ManyToOne[str]},
        output_types={"should_decompose": bool, "decomposition_score": float},
    )

    open_book_op = OpenBookReasoning.factory(
        llm=llm,
        retriever=get_retriever(Path().resolve() / "examples" / "hotpotqa" / "dataset" / "HotpotQA" / "wikipedia_index_bm25"),
        k=5)
    
    closed_book_op = ClosedBookReasoning.factory(llm=llm)

    child_aggregate_op = ChildAggregateReasoning.factory(llm=llm, use_many_to_one=False)

    filter_op = FilterOperation.factory(output_types={"answer": str, "decomposition_score": float}, input_types={"answers": ManyToOne[str], "decomposition_scores": ManyToOne[float]}, filter_function=filter_function)

    decompose_op = lambda: LLMOperationWithLogprobs(
        llm=llm,
        prompter=decompose_prompt,
        parser=decompose_parser,
        input_types={"question": str, "dependency_answers": list[str]},
        output_types={"subquestions": list[str], "question_decomposition_score": float},
    )

    understanding_op = lambda: OneLayerUnderstanding(
        branch_op=branch_op,
        reasoning_op=reasoning_op,
    )

    reasoning_op = lambda: OneLayerReasoning(
        open_book_op=open_book_op,
        closed_book_op=closed_book_op,
        child_aggregate_op=child_aggregate_op,
        filter_op=filter_op,
        decompose_op=decompose_op,
        understanding_op=understanding_op,
        min_branch_certainty_threshold=min_branch_certainty_threshold
    )

    probtree_graph = GraphOfOperations()
    probtree_graph.add_node(start_node)
    probtree_graph.add_node(end_node)
    branch_node = branch_op()
    probtree_graph.add_node(branch_node)
    reasoning_node = reasoning_op()
    probtree_graph.add_node(reasoning_node)

    probtree_graph.add_edge(Edge(start_node, branch_node, "question", "question"))
    
    probtree_graph.add_edge(Edge(start_node, reasoning_node, "question", "question"))
    probtree_graph.add_edge(Edge(start_node, reasoning_node, "max_depth", "max_depth"))
    probtree_graph.add_edge(Edge(start_node, reasoning_node, "question_decomposition_score", "question_decomposition_score"))

    probtree_graph.add_edge(Edge(branch_node, reasoning_node, "should_decompose", "should_decompose"))
    probtree_graph.add_edge(Edge(branch_node, reasoning_node, "decomposition_score", "should_decompose_score"))
    
    probtree_graph.add_edge(Edge(reasoning_node, end_node, "answer", "answer"))
    probtree_graph.add_edge(Edge(reasoning_node, end_node, "decomposition_score", "decomposition_score"))

    process_measurement = ProcessMeasurement(graph_of_operations=probtree_graph)

    controller = Controller(
        graph_of_operations=probtree_graph,
        scheduler=Scheduler.BFS,
        max_concurrent=1,
        process_measurement=process_measurement,
        store_intermediate_snapshots=True
    )
    
    return controller

if __name__ == "__main__":
    import asyncio
    controller = dynamic_probtree_controller(max_depth=1, min_branch_certainty_threshold=0.5)
    answer, measurement = asyncio.run(controller.execute({"question": "What is the capital of the population-wise largest country in the EU? Please decompose."}))
    print(answer)
    print(measurement)
    controller.graph_of_operations.snapshot.visualize(show_multiedges=False, show_keys=True, show_values=True, show_state=True)