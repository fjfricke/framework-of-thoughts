
import logging
from pathlib import Path
import asyncio

import numpy as np

from examples.openai_pricing import OPENAI_PRICING
from examples.sorting.dataloader import SortingDataloader, Split
from examples.sorting.programs.got import got_controller
from examples.sorting.programs.tot import tot_controller
from llm_graph_optimizer.graph_of_operations.types import ReasoningState
from llm_graph_optimizer.language_models.cache.cache import CacheContainer
from llm_graph_optimizer.language_models.helpers.language_model_config import Config
from llm_graph_optimizer.language_models.helpers.openai_rate_limiter import OpenAIRateLimiter
from llm_graph_optimizer.language_models.openai_chat import OpenAIChat
from llm_graph_optimizer.measurement.dataset_measurement import DatasetEvaluatorParameters, ScoreParameter
from llm_graph_optimizer.measurement.process_measurement import ProcessMeasurement
from llm_graph_optimizer.optimizer.dataset_evaluator import DatasetEvaluator


dataset_path = Path(__file__).parent / "dataset" / "sorting_128.csv"

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

accuracy_score = ScoreParameter(
    name="mistakes",
    confidence_interval_width=0.95,
    # acceptable_ci_width=0.05  # This sets the early stopping acceptable confidence interval width of each dataset evaluation
)
parameters = DatasetEvaluatorParameters(
    min_runs=10,
    score_parameters=[accuracy_score]
)

def calculate_score(reasoning_state: ReasoningState, measurement: ProcessMeasurement, ground_truth: list[int]) -> dict[ScoreParameter, float]:
    return {accuracy_score: reasoning_state["score"]}


def run_dataset_evaluation(process: str, original_or_optimized: str, split: Split):

    cache = CacheContainer.from_persistent_cache_file(Path(__file__).parent / "output" / "dataset_cache.pkl", load_as_virtual_persistent_cache=True, skip_on_file_not_found=False)
    dataloader_factory = lambda: SortingDataloader(split, dataset_path, split=0.5, seed=42)  # noqa: F821
    model = "gpt-3.5-turbo"
    llm = OpenAIChat(model=model,
                     config=Config(temperature=1.0),
                     cache=cache,
                     request_price_per_token=OPENAI_PRICING[model]["request_price_per_token"],
                     response_price_per_token=OPENAI_PRICING[model]["response_price_per_token"],
                     openai_rate_limiter=OpenAIRateLimiter(
                        rpm = OPENAI_PRICING[model]["RPM"],
                        tpm = OPENAI_PRICING[model]["TPM"]
                    )
    )
    if process == "tot":
        if original_or_optimized == "original":
            controller_factory = lambda: tot_controller(llm=llm, num_branches=20, improvement_levels=4, max_concurrent=1)
        elif original_or_optimized == "optimized":
            controller_factory = lambda: tot_controller(llm=llm, num_branches=14, improvement_levels=6, max_concurrent=1)
        else:
            raise ValueError("Invalid original or optimized mode")
    elif process == "got":
        if original_or_optimized == "original":
            controller_factory = lambda: got_controller(llm=llm, num_sort_branches=5, num_merge_branches=10, global_improvement_rounds=1, max_concurrent=1)
        elif original_or_optimized == "optimized":
            controller_factory = lambda: got_controller(llm=llm, num_sort_branches=2, num_merge_branches=13, global_improvement_rounds=2, max_concurrent=1)
        else:
            raise ValueError("Invalid original or optimized mode")
    dataset_evaluator = DatasetEvaluator(
        controller_factory=controller_factory,
        calculate_score=calculate_score,
        dataloader_factory=dataloader_factory,
        parameters=parameters,
        save_cache_on_completion_to=cache
    )
    scores = asyncio.run(dataset_evaluator.evaluate_dataset(max_concurrent=20))
    dataset_measurement = dataset_evaluator.dataset_measurement
    dataset_measurement.to_excel(Path(__file__).parent / "output" / "dataset_measurement.xlsx", maps_for_measurements={"mean": np.mean, "sum": np.sum})
    dataset_measurement.save(Path(__file__).parent / "output" / "dataset_measurement.pkl")
    cache.save_persistent_cache(Path(__file__).parent / "output" / "dataset_cache.pkl")
    print(scores)

if __name__ == "__main__":
    PROCESS = "got"
    ORIGINAL_OR_OPTIMIZED = "optimized"  # original or optimized
    SPLIT = Split.VALIDATION  # train or validation dataset
    run_dataset_evaluation(PROCESS, ORIGINAL_OR_OPTIMIZED, SPLIT)