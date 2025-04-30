
import logging
from pathlib import Path

import numpy as np

from examples.openai_pricing import OPENAI_PRICING
from examples.sorting.dataloader import SortingDataloader
from examples.sorting.programs.tot import tot_controller
from llm_graph_optimizer.graph_of_operations.types import ReasoningState
from llm_graph_optimizer.language_models.cache.cache import CacheContainer
from llm_graph_optimizer.language_models.helpers.language_model_config import Config
from llm_graph_optimizer.language_models.helpers.openai_rate_limiter import OpenAIRateLimiter
from llm_graph_optimizer.language_models.openai_chat import OpenAIChat
from llm_graph_optimizer.measurement.dataset_measurement import DatasetEvaluatorParameters, ScoreParameter
from llm_graph_optimizer.measurement.process_measurement import ProcessMeasurement
from llm_graph_optimizer.optimizer.dataset_evaluator import DatasetEvaluator


dataset_path = Path(__file__).parent / "dataset" / "sorting_064.csv"
dataloader_factory = lambda: SortingDataloader(dataset_path)

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
# logging.getLogger('llm_graph_optimizer.controller.controller').setLevel(logging.DEBUG)

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



if __name__ == "__main__":
    import asyncio
    cache = CacheContainer(Path(__file__).parent / "output" / "cache.pkl")
    model = "gpt-3.5-turbo"
    llm = OpenAIChat(model=model, config=Config(temperature=1.0), cache=cache,
                     request_price_per_token=OPENAI_PRICING[model]["request_price_per_token"],
                     response_price_per_token=OPENAI_PRICING[model]["response_price_per_token"],
                     openai_rate_limiter=OpenAIRateLimiter(
                        rpm = OPENAI_PRICING[model]["RPM"],
                        tpm = OPENAI_PRICING[model]["TPM"]
                    )
    )
    controller_factory = lambda: tot_controller(llm, num_branches=4, improvement_levels=2, max_concurrent=5)
    dataset_evaluator = DatasetEvaluator(
        controller_factory=controller_factory,
        calculate_score=calculate_score,
        dataloader_factory=dataloader_factory,
        parameters=parameters
    )
    scores = asyncio.run(dataset_evaluator.evaluate_dataset(max_concurrent=4))
    dataset_measurement = dataset_evaluator.dataset_measurement
    dataset_measurement.to_excel(Path(__file__).parent / "output" / "dataset_measurement.xlsx", maps_for_measurements={"mean": np.mean, "sum": np.sum})
    dataset_measurement.save(Path(__file__).parent / "output" / "dataset_measurement.pkl")
    cache.save_persistent_cache(Path(__file__).parent / "output" / "dataset_cache.pkl")
    print(scores)