import logging
from pathlib import Path
import re
import numexpr as ne
import numpy as np

from examples.game_of_24.dataloader import GameOf24Dataloader, Split
from examples.game_of_24.programs.tot import tot_controller

from llm_graph_optimizer.graph_of_operations.types import ReasoningState, StateSetFailed
from llm_graph_optimizer.language_models.helpers.openai_rate_limiter import OpenAIRateLimiter
from llm_graph_optimizer.language_models.openai_chat import OpenAIChat
from llm_graph_optimizer.measurement.process_measurement import ProcessMeasurement
from llm_graph_optimizer.optimizer.dataset_evaluator import DatasetEvaluator, DatasetEvaluatorParameters, ScoreParameter
from llm_graph_optimizer.language_models.cache.cache import CacheContainer
from examples.openai_pricing import OPENAI_PRICING

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

success_score = ScoreParameter(
    name="success",
    confidence_interval_width=0.95,
)

parameters = DatasetEvaluatorParameters(
    min_runs=10,
    score_parameters=[success_score]
)

def get_score_from_reasoning_state(reasoning_state: ReasoningState, ground_truth: list[int]) -> float:
    if reasoning_state is None or "answer" not in reasoning_state:
        return 0.0
    # ckeck if reasoning state answer expression on the left only contains the numbers in the ground truth
    answer = reasoning_state["answer"]
    if answer == StateSetFailed:
        return 0.0
    answer_expression = answer.split("=")[0].strip()
    # exctract all numbers from the answer expression
    answer_numbers = [int(num) for num in re.findall(r'\d+', answer_expression)]
    if not sorted(answer_numbers) == sorted(ground_truth):
        return 0.0
    # check if the expression on the left evaluates to 24 using safe eval
    try:
        result = ne.evaluate(answer_expression)
        if result != 24:
            return 0.0
    except Exception:
        return 0.0
    return 1.0

def calculate_score(reasoning_state: ReasoningState | None, measurement: ProcessMeasurement, ground_truth: list[int]) -> dict[ScoreParameter, float]:
    return {success_score: get_score_from_reasoning_state(reasoning_state, ground_truth)}


def run_dataset_evaluation(original_or_optimized: str, split: Split):
    import asyncio
    cache = CacheContainer.from_persistent_cache_file(Path(__file__).parent / "output" / "dataset_cache_3.pkl", load_as_virtual_persistent_cache=True, skip_on_file_not_found=True)
    dataloader_factory = lambda: GameOf24Dataloader(execution_mode=split)
    model = "gpt-4o"
    llm = OpenAIChat(model=model,
        cache=cache,
        request_price_per_token=OPENAI_PRICING[model]["request_price_per_token"],
        response_price_per_token=OPENAI_PRICING[model]["response_price_per_token"],
        openai_rate_limiter=OpenAIRateLimiter(
            rpm=OPENAI_PRICING[model]["RPM"],
            tpm=OPENAI_PRICING[model]["TPM"]
        )
    )
    if original_or_optimized == "original":
        controller_factory = lambda: tot_controller(llm, num_examples=8, samples=[3, 3, 3], keep_top_n=[5, 5, 1], max_concurrent=40)
    elif original_or_optimized == "optimized":
        controller_factory = lambda: tot_controller(llm, num_examples=11, samples=[3, 2, 2], keep_top_n=[5, 3, 1], max_concurrent=40)
    else:
        raise ValueError("Invalid original or optimized mode")
    dataset_evaluator = DatasetEvaluator(
        controller_factory=controller_factory,
        calculate_score=calculate_score,
        dataloader_factory=dataloader_factory,
        parameters=parameters,
        save_cache_on_completion_to=cache
    )
    scores = asyncio.run(dataset_evaluator.evaluate_dataset(max_concurrent=10))
    dataset_measurement = dataset_evaluator.dataset_measurement
    dataset_measurement.to_excel(Path(__file__).parent / "output" / "dataset_measurement.xlsx", maps_for_measurements={"mean": np.mean, "sum": np.sum})
    dataset_measurement.save(Path(__file__).parent / "output" / "dataset_measurement.pkl")
    cache.save_persistent_cache()
    print(scores)

if __name__ == "__main__":
    ORIGINAL_OR_OPTIMIZED = "original"
    SPLIT = Split.VALIDATION
    run_dataset_evaluation(ORIGINAL_OR_OPTIMIZED, SPLIT)