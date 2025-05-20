import asyncio
import logging
from pathlib import Path

import numpy as np

from examples.doc_merge.dataloader import DocMergeDataloader, Split
from examples.doc_merge.programs.got import got_controller
from examples.openai_pricing import OPENAI_PRICING
from llm_graph_optimizer.graph_of_operations.types import ReasoningState
from llm_graph_optimizer.language_models.cache.cache import CacheContainer
from llm_graph_optimizer.language_models.helpers.language_model_config import Config
from llm_graph_optimizer.language_models.helpers.openai_rate_limiter import OpenAIRateLimiter
from llm_graph_optimizer.language_models.openai_chat import OpenAIChat
from llm_graph_optimizer.measurement.dataset_measurement import DatasetEvaluatorParameters, ScoreParameter
from llm_graph_optimizer.measurement.process_measurement import ProcessMeasurement
from llm_graph_optimizer.optimizer.dataset_evaluator import DatasetEvaluator

dataset_path = Path(__file__).parent / "dataset" / "documents.csv"
dataloader_factory = lambda: DocMergeDataloader(dataset_path=dataset_path, execution_mode=Split.VALIDATION, split=0.5, seed=42)

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

redundancy_score = ScoreParameter(
    name="redundancy",
    confidence_interval_width=0.95
)

retention_score = ScoreParameter(
    name="retention",
    confidence_interval_width=0.95
)

f1_score = ScoreParameter(
    name="f1_score",
    confidence_interval_width=0.95,
    acceptable_ci_width=0.05
)

parameters = DatasetEvaluatorParameters(
    score_parameters=[redundancy_score, retention_score, f1_score],
    min_runs=10,
)

def calculate_score(reasoning_state: ReasoningState, measurement: ProcessMeasurement, ground_truth: None) -> float:
    return {
        redundancy_score: reasoning_state["redundancies"][0],
        retention_score: reasoning_state["retentions"][0],
        f1_score: reasoning_state["f1_scores"][0]
    }


if __name__ == "__main__":
    cache = CacheContainer.from_persistent_cache_file(
        file_path=Path(__file__).parent / "output" / "cache.pkl",
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
    llm_gen = llm_generator(1)
    llm_score = llm_generator(0)
    
    controller_factory = lambda: got_controller(
        llm_gen=llm_gen,
        llm_score=llm_score,
        num_merges=4,
        keep_best_merges=3,
        num_aggregations=5,
        num_improvements=10,
        max_concurrent=1,
    )
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
    print(scores)