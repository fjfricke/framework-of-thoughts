from __future__ import annotations

from typing import List

from llm_graph_optimizer.controller.controller import Controller
from llm_graph_optimizer.graph_of_operations.graph_of_operations import GraphOfOperations
from llm_graph_optimizer.graph_of_operations.types import Edge, ManyToOne
from llm_graph_optimizer.language_models.abstract_language_model import AbstractLanguageModel
from llm_graph_optimizer.language_models.openai_chat import OpenAIChat
from llm_graph_optimizer.operations.base_operations.end import End
from llm_graph_optimizer.operations.base_operations.filter_operation import FilterOperation
from llm_graph_optimizer.operations.base_operations.score_operation import ScoreOperation
from llm_graph_optimizer.operations.base_operations.start import Start
from llm_graph_optimizer.operations.llm_operations.base_llm_operation import BaseLLMOperation
from llm_graph_optimizer.measurement.process_measurement import ProcessMeasurement
from llm_graph_optimizer.schedulers.schedulers import Scheduler

from examples.sorting.programs.prompter_parser import (
    generate_prompt,
    generate_parser,
    got_split_prompt,
    got_split_parser,
    got_aggregate_prompt,
    tot_improve_prompt,
    scoring_function,
    filter_function,
)

# Scoring helpers
def _score_sublist(output: List[int], unsorted_sublist: List[int]) -> int:
    return scoring_function(output, sorted(unsorted_sublist))


def _score_merge(output: List[int], input1: List[int], input2: List[int]) -> int:
    return scoring_function(output, sorted(input1 + input2))


# Graph builder
def got_controller(
    llm: AbstractLanguageModel,
    num_sort_branches: int = 5,
    num_merge_branches: int = 5,
    global_improvement_rounds: int = 2,
    max_concurrent: int = 5,
) -> Controller:

    # Node factories
    def split_op() -> BaseLLMOperation:
        return BaseLLMOperation(
            llm=llm,
            prompter=got_split_prompt,
            parser=got_split_parser,
            input_types={"input_list": List[int]},
            output_types={f"output{i}": List[int] for i in range(1, 9)},
            name="SplitInto8",
        )

    def sort_op(cache_seed: int = 0) -> BaseLLMOperation:
        return BaseLLMOperation(
            llm=llm,
            prompter=generate_prompt,
            parser=generate_parser,
            cache_seed=cache_seed,
            input_types={"input_list": List[int]},
            output_types={"output": List[int]},
            name="SortSlice",
        )

    def merge_op(cache_seed: int = 0) -> BaseLLMOperation:
        return BaseLLMOperation(
            llm=llm,
            prompter=got_aggregate_prompt,
            parser=generate_parser,
            cache_seed=cache_seed,
            input_types={"input1": List[int], "input2": List[int]},
            output_types={"output": List[int]},
            name="MergeSorted",
        )

    def improve_op(cache_seed: int = 0) -> BaseLLMOperation:
        return BaseLLMOperation(
            llm=llm,
            prompter=tot_improve_prompt,
            parser=generate_parser,
            cache_seed=cache_seed,
            input_types={"input_list": List[int], "incorrectly_sorted": List[int]},
            output_types={"output": List[int]},
            name="RepairList",
        )

    def score_sublist() -> ScoreOperation:
        return ScoreOperation(
            input_types={"output": List[int], "unsorted_sublist": List[int]},
            output_type=int,
            scoring_function=_score_sublist,
        )

    def score_merge() -> ScoreOperation:
        return ScoreOperation(
            input_types={"output": List[int], "input1": List[int], "input2": List[int]},
            output_type=int,
            scoring_function=_score_merge,
            name="ScoreMerge",
        )

    def score_final() -> ScoreOperation:
        return ScoreOperation(
            input_types={"output": List[int], "expected_output": List[int]},
            output_type=int,
            scoring_function=scoring_function,
            name="ScoreFinal",
        )

    def filter_op() -> FilterOperation:
        return FilterOperation(
            input_types={"outputs": ManyToOne[List[int]], "scores": ManyToOne[int]},
            output_types={"output": List[int], "score": int},
            filter_function=filter_function,
        )

    # Graph
    graph = GraphOfOperations()
    start = Start(input_types={"input_list": List[int], "expected_output": List[int]})
    end = End(input_types={"output": List[int], "score": int, "expected_output": List[int]})
    graph.add_node(start)

    # Split
    split = split_op()
    graph.add_node(split)
    graph.add_edge(Edge(start, split, "input_list", "input_list"))

    # Sort slices
    slice_best: List[FilterOperation] = []
    for idx in range(1, 9):
        keep_best = filter_op()
        graph.add_node(keep_best)
        slice_best.append(keep_best)
        key = f"output{idx}"
        for br in range(num_sort_branches):
            sorter = sort_op(br)
            scorer = score_sublist()
            graph.add_node(sorter)
            graph.add_node(scorer)
            graph.add_edge(Edge(split, sorter, key, "input_list"))
            graph.add_edge(Edge(split, scorer, key, "unsorted_sublist"))
            graph.add_edge(Edge(sorter, scorer, "output", "output"))
            graph.add_edge(Edge(scorer, keep_best, "score", "scores"), order=br)
            graph.add_edge(Edge(sorter, keep_best, "output", "outputs"), order=br)

    # Merge neighbouring pairs
    pair_best: List[FilterOperation] = []
    for (pair_idx, (a, b)) in enumerate([(1, 2), (3, 4), (5, 6), (7, 8)], start=1):
        keep_best = filter_op()
        graph.add_node(keep_best)
        pair_best.append(keep_best)
        left, right = slice_best[a - 1], slice_best[b - 1]
        for br in range(num_merge_branches):
            merger = merge_op(br)
            scorer = score_merge()
            graph.add_node(merger)
            graph.add_node(scorer)
            graph.add_edge(Edge(left, merger, "output", "input1"))
            graph.add_edge(Edge(right, merger, "output", "input2"))
            graph.add_edge(Edge(merger, scorer, "output", "output"))
            graph.add_edge(Edge(left, scorer, "output", "input1"))
            graph.add_edge(Edge(right, scorer, "output", "input2"))
            graph.add_edge(Edge(scorer, keep_best, "score", "scores"), order=br)
            graph.add_edge(Edge(merger, keep_best, "output", "outputs"), order=br)

    # Merge halves
    half_best: List[FilterOperation] = []
    for (idx, (a, b)) in enumerate([(1, 2), (3, 4)], start=1):
        keep_best = filter_op()
        graph.add_node(keep_best)
        half_best.append(keep_best)
        left, right = pair_best[a - 1], pair_best[b - 1]
        for br in range(num_merge_branches):
            merger = merge_op(br)
            scorer = score_merge()
            graph.add_node(merger)
            graph.add_node(scorer)
            graph.add_edge(Edge(left, merger, "output", "input1"))
            graph.add_edge(Edge(right, merger, "output", "input2"))
            graph.add_edge(Edge(merger, scorer, "output", "output"))
            graph.add_edge(Edge(left, scorer, "output", "input1"))
            graph.add_edge(Edge(right, scorer, "output", "input2"))
            graph.add_edge(Edge(scorer, keep_best, "score", "scores"), order=br)
            graph.add_edge(Edge(merger, keep_best, "output", "outputs"), order=br)

    # Final merge
    final_keep = filter_op()
    graph.add_node(final_keep)
    left, right = half_best
    for br in range(num_merge_branches):
        merger = merge_op(br)
        scorer = score_merge()
        graph.add_node(merger)
        graph.add_node(scorer)
        graph.add_edge(Edge(left, merger, "output", "input1"))
        graph.add_edge(Edge(right, merger, "output", "input2"))
        graph.add_edge(Edge(merger, scorer, "output", "output"))
        graph.add_edge(Edge(left, scorer, "output", "input1"))
        graph.add_edge(Edge(right, scorer, "output", "input2"))
        graph.add_edge(Edge(scorer, final_keep, "score", "scores"), order=br)
        graph.add_edge(Edge(merger, final_keep, "output", "outputs"), order=br)

    # Global repair rounds
    last_best_node: FilterOperation | BaseLLMOperation = final_keep
    for r in range(global_improvement_rounds):
        repair = improve_op(r)
        scorer = score_final()
        keep_best = filter_op()
        graph.add_node(repair)
        graph.add_node(scorer)
        graph.add_node(keep_best)
        graph.add_edge(Edge(start, repair, "input_list", "input_list"))
        graph.add_edge(Edge(last_best_node, repair, "output", "incorrectly_sorted"))
        graph.add_edge(Edge(repair, scorer, "output", "output"))
        graph.add_edge(Edge(start, scorer, "expected_output", "expected_output"))
        graph.add_edge(Edge(scorer, keep_best, "score", "scores"), order=-1)
        graph.add_edge(Edge(repair, keep_best, "output", "outputs"), order=-1)
        graph.add_edge(Edge(last_best_node, keep_best, "output", "outputs"), order=-2)
        graph.add_edge(Edge(last_best_node, keep_best, "score", "scores"), order=-2)
        last_best_node = keep_best

    # Final score
    graph.add_node(end)
    graph.add_edge(Edge(last_best_node, end, "output", "output"))
    graph.add_edge(Edge(last_best_node, end, "score", "score"))
    graph.add_edge(Edge(start, end, "expected_output", "expected_output"))

    measurement = ProcessMeasurement(graph_of_operations=graph)
    return Controller(graph_of_operations=graph, scheduler=Scheduler.BFS, max_concurrent=max_concurrent, process_measurement=measurement)

if __name__ == "__main__":
    import asyncio

    llm = OpenAIChat(model="gpt-3.5-turbo")

    SAMPLE = [0, 9, 4, 2, 2, 0, 5, 1] * 16  # 128 numbers, easy to verify
    EXPECTED = sorted(SAMPLE)

    controller = got_controller(llm=llm, num_sort_branches=2, num_merge_branches=2, global_improvement_rounds=1, max_concurrent=1)
    result, measurement = asyncio.run(controller.execute(input={"input_list": SAMPLE, "expected_output": EXPECTED}, debug_params={"raise_on_operation_failure": True}))
    controller.graph_of_operations.snapshot.visualize(show_multiedges=False, show_values=True, show_keys=True, show_state=True)
    print("Final result:", result)
    print("Measurement:\n", measurement)