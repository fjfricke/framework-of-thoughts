"""
1. Merge the 4 NDAs into a single one 5 times; Score each attempt and keep the best 3
2. Aggregate the merge attempts into a single one 5 times; Score each aggregation attempt and keep the overall best attempt (including Step 1)
3. Improve the merged NDA 10 times; Score each and keep the best
"""

from pathlib import Path
import dspy
from examples.doc_merge.dataloader import DocMergeDataloader, Split
from examples.doc_merge.programs.prompter_parser import aggregate_parser, aggregate_prompt, improve_parser, improve_prompt, merge_parser, merge_prompt, score_parser, score_prompt
from examples.openai_pricing import OPENAI_PRICING
from llm_graph_optimizer.controller.controller import Controller
from llm_graph_optimizer.graph_of_operations.graph_of_operations import GraphOfOperations
from llm_graph_optimizer.graph_of_operations.types import Edge, ManyToOne
from llm_graph_optimizer.language_models.abstract_language_model import AbstractLanguageModel
from llm_graph_optimizer.language_models.cache.cache import CacheContainer
from llm_graph_optimizer.language_models.helpers.language_model_config import Config
from llm_graph_optimizer.language_models.helpers.openai_rate_limiter import OpenAIRateLimiter
from llm_graph_optimizer.language_models.openai_chat import OpenAIChat
from llm_graph_optimizer.measurement.process_measurement import ProcessMeasurement
from llm_graph_optimizer.operations.base_operations.filter_operation_with_edge_move import Correspondence, FilterOperationWithEdgeMove
from llm_graph_optimizer.operations.base_operations.start import Start
from llm_graph_optimizer.operations.base_operations.end import End
from llm_graph_optimizer.operations.llm_operations.base_llm_operation import BaseLLMOperation
from llm_graph_optimizer.operations.llm_operations.dspy.shared_prompt_llm_operation import SharedPromptLLMOperation
from llm_graph_optimizer.schedulers.schedulers import Scheduler


def got_controller(
    *,
    llm_gen: AbstractLanguageModel,
    llm_score: AbstractLanguageModel,
    num_merges: int = 4,
    keep_best_merges: int = 3,
    num_aggregations: int = 5,
    num_improvements: int = 10,
    max_concurrent: int = 1,
    use_dspy: bool = False,
    save_to_cache_after_execution: CacheContainer = None,
    optimized_improve_prompter = None
) -> Controller:
    
    start_node = Start(
        input_types={"docs": list[str]},
    )

    end_node = End(
        input_types={"merged": ManyToOne[str], "redundancies": ManyToOne[float], "retentions": ManyToOne[float], "f1_scores": ManyToOne[float]},
    )
    
    def merge_op(cache_seed: int) -> BaseLLMOperation:
        return BaseLLMOperation(
            llm=llm_gen,
            prompter=merge_prompt,
            parser=merge_parser,
            cache_seed=cache_seed,
            input_types={"docs": list[str]},
            output_types={"merged": str},
            name="Merge",
        )

    def aggregate_op(cache_seed: int) -> BaseLLMOperation:
        return BaseLLMOperation(
            llm=llm_gen,
            prompter=aggregate_prompt,
            parser=aggregate_parser,
            cache_seed=cache_seed,
            input_types={"summaries": ManyToOne[str], "docs": list[str]},
            output_types={"merged": str},
            name="Aggregate",
        )
    
    def improve_op(cache_seed: int) -> BaseLLMOperation:
        if not use_dspy:
            return BaseLLMOperation(
                llm=llm_gen,
                prompter=improve_prompt if not optimized_improve_prompter else optimized_improve_prompter,
                parser=improve_parser,
                cache_seed=cache_seed,
                input_types={"summaries": ManyToOne[str], "docs": list[str]},  # only one summary will be passed
                output_types={"merged": str},
                name="Improve",
            )
        else:
            class ImproveSignature(dspy.Signature):
                """Improves the summary of NDAs. Output needs to be between the two tags <Merged> and </Merged>"""
                summaries: list[str] = dspy.InputField(description="The summaries of the NDAs to improve")
                docs: list[str] = dspy.InputField(description="The original NDAs")
                merged: str = dspy.OutputField(description="The improved summary of the NDAs")

            return SharedPromptLLMOperation(
                group_id="improve",
                llm=llm_gen,
                parser=improve_parser,
                signature=ImproveSignature,
                input_types={"summaries": ManyToOne[str], "docs": list[str]},  # only one summary will be passed
                output_types={"merged": str},
                cache_seed=cache_seed,
                name="Improve",
            )
    
    def score_op() -> BaseLLMOperation:
        return BaseLLMOperation(
            llm=llm_score,
            prompter=score_prompt,
            parser=score_parser,
            input_types={"summary": str, "docs": list[str]},
            output_types={"redundancy": float, "retention": float, "f1_score": float},
            name="Score",
        )
    
    def get_top_f1_indices(f1_scores: list[float], keep_best_merges: int) -> list[int]:
        """
        Get the indices of the top `keep_best_merges` F1 scores.

        :param f1_scores: List of F1 scores.
        :param keep_best_merges: Number of top scores to keep.
        :return: Indices of the top `keep_best_merges` F1 scores.
        """
        # Pair each F1 score with its index and sort by score (descending) and index (ascending for ties)
        sorted_indices = sorted(range(len(f1_scores)), key=lambda i: (-f1_scores[i], i))

        # Return the top `keep_best_merges` indices
        return sorted_indices[:keep_best_merges]
    
    def keep_best_op(keep_best_merges: int = keep_best_merges) -> FilterOperationWithEdgeMove:
        return FilterOperationWithEdgeMove(
            input_types={"summaries": ManyToOne[str], "redundancies": ManyToOne[float], "retentions": ManyToOne[float], "f1_scores": ManyToOne[float]},
            filter_function=lambda summaries, redundancies, retentions, f1_scores: get_top_f1_indices(f1_scores, keep_best_merges),
            correspondence=Correspondence.MANY_TO_ONE,
            name="KeepBestN",
        )


    graph = GraphOfOperations()
    graph.add_node(start_node)
    graph.add_node(end_node)

    score_nodes = []
    merge_nodes = []
    for i in range(num_merges):
        merge_node = merge_op(i)
        graph.add_node(merge_node)
        graph.add_edge(Edge(start_node, merge_node, "docs", "docs"))
        score_node = score_op()
        graph.add_node(score_node)
        graph.add_edge(Edge(merge_node, score_node, "merged", "summary"))
        graph.add_edge(Edge(start_node, score_node, "docs", "docs"))
        score_nodes.append(score_node)
        merge_nodes.append(merge_node)
    keep_best_n_node = keep_best_op(keep_best_merges=keep_best_merges)
    graph.add_node(keep_best_n_node)
    for i, (score_node, merge_node)  in enumerate(zip(score_nodes, merge_nodes)):
        graph.add_edge(Edge(score_node, keep_best_n_node, "f1_score", "f1_scores"), order=i)
        graph.add_edge(Edge(merge_node, keep_best_n_node, "merged", "summaries"), order=i)

    score2_nodes = []
    aggregate_nodes = []
    for i in range(num_aggregations):
        aggregate_node = aggregate_op(i)
        graph.add_node(aggregate_node)
        graph.add_edge(Edge(keep_best_n_node, aggregate_node, "merged", "summaries"))
        graph.add_edge(Edge(start_node, aggregate_node, "docs", "docs"))
        score_node = score_op()
        graph.add_node(score_node)
        graph.add_edge(Edge(aggregate_node, score_node, "merged", "summary"))
        graph.add_edge(Edge(start_node, score_node, "docs", "docs"))
        score2_nodes.append(score_node)
        aggregate_nodes.append(aggregate_node)
    keep_best_n_node = keep_best_op(keep_best_merges=1)
    graph.add_node(keep_best_n_node)
    for i, (score_node, merge_or_aggregate_node) in enumerate(zip(score_nodes + score2_nodes, merge_nodes + aggregate_nodes)):
        graph.add_edge(Edge(score_node, keep_best_n_node, "f1_score", "f1_scores"), order=i)
        graph.add_edge(Edge(merge_or_aggregate_node, keep_best_n_node, "merged", "summaries"), order=i)

    score3_nodes = []
    improve_nodes = []
    for i in range(num_improvements):
        improve_node = improve_op(i)
        graph.add_node(improve_node)
        graph.add_edge(Edge(keep_best_n_node, improve_node, "merged", "summaries"))
        graph.add_edge(Edge(start_node, improve_node, "docs", "docs"))
        score_node = score_op()
        graph.add_node(score_node)
        graph.add_edge(Edge(improve_node, score_node, "merged", "summary"))
        graph.add_edge(Edge(start_node, score_node, "docs", "docs"))
        score3_nodes.append(score_node)
        improve_nodes.append(improve_node)

    keep_best_n_node = keep_best_op(keep_best_merges=1)
    graph.add_node(keep_best_n_node)
    for i, (score_node, improve_node) in enumerate(zip(score3_nodes, improve_nodes)):
        graph.add_edge(Edge(score_node, keep_best_n_node, "f1_score", "f1_scores"), order=i)
        graph.add_edge(Edge(score_node, keep_best_n_node, "redundancy", "redundancies"), order=i)
        graph.add_edge(Edge(score_node, keep_best_n_node, "retention", "retentions"), order=i)
        graph.add_edge(Edge(improve_node, keep_best_n_node, "merged", "summaries"), order=i)

    graph.add_edge(Edge(keep_best_n_node, end_node, "merged", "merged"))
    graph.add_edge(Edge(keep_best_n_node, end_node, "redundancy", "redundancies"))
    graph.add_edge(Edge(keep_best_n_node, end_node, "retention", "retentions"))
    graph.add_edge(Edge(keep_best_n_node, end_node, "f1_score", "f1_scores"))

    return Controller(
        graph_of_operations=graph,
        scheduler=Scheduler.BFS,
        max_concurrent=max_concurrent,
        process_measurement=ProcessMeasurement(graph_of_operations=graph),
        save_to_cache_after_execution=save_to_cache_after_execution,
    )

if __name__ == "__main__":
    import asyncio

    cache = CacheContainer.from_persistent_cache_file(
        file_path=Path(__file__).parent.parent / "output" / "cache.pkl",
        load_as_virtual_persistent_cache=True,
        skip_on_file_not_found=True
    )

    model = "gpt-3.5-turbo"
    openai_rate_limiter = OpenAIRateLimiter(
        rpm=OPENAI_PRICING[model]["RPM"],
        tpm=OPENAI_PRICING[model]["TPM"]
    )
    llm_generator = lambda temperature: OpenAIChat(
        model=model,
        config=Config(temperature=temperature),
        cache=cache,
        request_price_per_token=OPENAI_PRICING[model]["request_price_per_token"],
        response_price_per_token=OPENAI_PRICING[model]["response_price_per_token"],
        openai_rate_limiter=openai_rate_limiter
    )

    # Example of very short NDAs for mock testing
    dataloader = DocMergeDataloader(
        dataset_path=Path(__file__).parent.parent / "dataset" / "documents.csv",
        execution_mode=Split.TEST
    )
    short_ndas = next(dataloader)[0]

    
    llm_gen = llm_generator(1)
    llm_score = llm_generator(0)
    controller = got_controller(
        llm_gen=llm_gen,
        llm_score=llm_score,
        num_merges=4,
        keep_best_merges=3,
        num_aggregations=5,
        num_improvements=10,
        max_concurrent=10,
        use_dspy=False,
        save_to_cache_after_execution=cache
    )
    result, measurement = asyncio.run(controller.execute(input=short_ndas, debug_params={"raise_on_operation_failure": True}))
    controller.graph_of_operations.snapshot.visualize(show_multiedges=False, show_values=True, show_keys=True, show_state=True)
    print(result)
    print(measurement)
    cache.save_persistent_cache(Path(__file__).parent.parent / "output" / "cache.pkl")