import logging
from pathlib import Path
import numpy as np
import optuna

from examples.test.dataloader import TestDatasetLoaderWithYield
from examples.test.test_controller import controller
from llm_graph_optimizer.graph_of_operations.types import ReasoningState
from llm_graph_optimizer.language_models.cache.cache import CacheContainer
from llm_graph_optimizer.measurement.dataset_measurement import DatasetEvaluatorParameters, ScoreParameter
from llm_graph_optimizer.measurement.process_measurement import ProcessMeasurement
from llm_graph_optimizer.measurement.study_measurement import StudyMeasurement
from llm_graph_optimizer.optimizer.dataset_evaluator import DatasetEvaluator
from llm_graph_optimizer.optimizer.study_optuna import Study

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    dataset_path = Path(__file__).parent / "dataset" / "test_dataset.txt"
    dataloader = TestDatasetLoaderWithYield(dataset_path)

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

    def calculate_score(reasoning_state: ReasoningState, measurement: ProcessMeasurement, ground_truth: int) -> dict[ScoreParameter, float]:  # noqa: F821
        return {accuracy_score: 1} if reasoning_state["end"] == ground_truth else {accuracy_score: 0}


    cache = CacheContainer.from_persistent_cache_file(Path(__file__).parent / "output" / "test_optimizer_cache.pkl", skip_on_file_not_found=True)
    controller_factory_with_optimization_parameters = lambda number_of_llm_nodes: controller(cache=cache, number_of_llm_nodes=number_of_llm_nodes)

    def objective(trial: optuna.Trial):
        number_of_llm_nodes = trial.suggest_int("number_of_llm_nodes", 1, 21)
        controller_factory = lambda: controller_factory_with_optimization_parameters(number_of_llm_nodes=number_of_llm_nodes)
        return controller_factory

    optuna.delete_study(study_name="test_optimizer", storage="sqlite:///db.sqlite3")
    optuna_study = optuna.create_study(
        direction="maximize",
        storage="sqlite:///db.sqlite3",
        study_name="test_optimizer",
        )
    study_measurement = StudyMeasurement()
    study = Study(
        optuna_study=optuna_study,
        metrics=[accuracy_score],
        dataset_evaluator=DatasetEvaluator(
            calculate_score=calculate_score,
            dataloader=dataloader,
            parameters=parameters),
        max_concurrent=10,
        study_measurement=study_measurement
    )
    study.set_objective(objective)
    study.optimize(n_trials=10)
    study_measurement.to_excel(Path(__file__).parent / "output" / "test_optimizer_study_measurement.xlsx")
    study_measurement.best_run.to_excel(
        Path(__file__).parent / "output" / "test_optimizer_best_run.xlsx",
        maps_for_measurements={
            "mean": np.mean,
            "sum": np.sum
        }
    )
    cache.save_persistent_cache(Path(__file__).parent / "output" / "test_optimizer_cache.pkl")
    print(optuna_study.best_trial)