from pathlib import Path
import logging

import numpy as np

from examples.test.dataloader import TestDatasetLoaderWithYield
from examples.test.test_controller import controller

from llm_graph_optimizer.graph_of_operations.types import ReasoningState
from llm_graph_optimizer.language_models.cache.cache import CacheContainer
from llm_graph_optimizer.measurement.process_measurement import ProcessMeasurement
from llm_graph_optimizer.optimizer.dataset_evaluator import DatasetEvaluator, DatasetEvaluatorParameters, ScoreParameter

dataset_path = Path(__file__).parent / "dataset" / "test_dataset.txt"
dataloader = TestDatasetLoaderWithYield(dataset_path)

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

accuracy_score = ScoreParameter(
    name="accuracy",
    confidence_interval_width=0.95,
    acceptable_ci_width=0.05
)
parameters = DatasetEvaluatorParameters(
    min_runs=10,
    max_runs=500,
    score_parameters=[accuracy_score]
)

def calculate_score(reasoning_state: ReasoningState, measurement: ProcessMeasurement, ground_truth: int) -> dict[ScoreParameter, float]:
    return {accuracy_score: 1} if reasoning_state["end"] == ground_truth else {accuracy_score: 0}



if __name__ == "__main__":
    import asyncio
    cache = CacheContainer.from_persistent_cache_file(Path(__file__).parent / "output" / "test_dataset_cache.pkl", skip_on_file_not_found=True)
    controller_factory = lambda: controller(cache, number_of_llm_nodes=5)
    dataset_evaluator = DatasetEvaluator(
        controller_factory=controller_factory,
        calculate_score=calculate_score,
        dataloader=dataloader,
        parameters=parameters
    )
    scores = asyncio.run(dataset_evaluator.evaluate_dataset(max_concurrent=10))
    dataset_measurement = dataset_evaluator.dataset_measurement
    mean_measurements = dataset_measurement.calculate_dataset_measurement(np.mean)
    dataset_measurement.to_excel(Path(__file__).parent / "output" / "test_dataset_measurement.xlsx", maps_for_measurements={"mean": np.mean})
    cache.save_persistent_cache(Path(__file__).parent / "output" / "test_dataset_cache.pkl")
    print(scores)