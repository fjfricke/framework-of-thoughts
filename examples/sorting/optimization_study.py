import logging
from pathlib import Path

import numpy as np
import optuna

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
from llm_graph_optimizer.measurement.study_measurement import StudyMeasurement
from llm_graph_optimizer.optimizer.dataset_evaluator import DatasetEvaluator
from llm_graph_optimizer.optimizer.study_optuna import Study

logging.getLogger().setLevel(logging.ERROR)

dataset_path = Path(__file__).parent / "dataset" / "sorting_128.csv"

mistakes = ScoreParameter(
    name="mistakes",
    confidence_interval_width=0.95,
    # acceptable_ci_width=0.1
)

mistakes_penalized = ScoreParameter(
    name="mistakes_penalized",
    confidence_interval_width=0.95,
    # acceptable_ci_width=0.1
)

parameters = DatasetEvaluatorParameters(
    min_runs=10,
    # max_runs=10,
    score_parameters=[mistakes, mistakes_penalized]
)

def calculate_score(reasoning_state: ReasoningState, measurement: ProcessMeasurement, ground_truth: list[int]) -> dict[ScoreParameter, float]:
    if reasoning_state is None or "score" not in reasoning_state:
        return {mistakes: 128, mistakes_penalized: 128}

    base_score = reasoning_state["score"]
    cost = measurement.total_sequential_cost().with_process_cache.total_cost

    # Parameters for exponential penalty
    cost_start = 0.05
    cost_end = 0.1
    max_penalty = 2 * base_score
    B = 60  # Steepness factor

    if cost <= cost_start:
        penalty = 0
    else:
        shifted_cost = cost - cost_start
        cost_range = cost_end - cost_start

        # Compute A to normalize the curve to reach max_penalty at cost_end
        A = max_penalty / (np.exp(B * cost_range) - 1)
        penalty = A * (np.exp(B * shifted_cost) - 1)
        penalty = min(penalty, max_penalty)

    total_score = min(base_score + penalty, 128)
    return {mistakes: base_score, mistakes_penalized: total_score}

def sorting_study(process: str):

    dataloader_factory = lambda: SortingDataloader(Split.TRAIN, dataset_path, split=0.5, seed=42)

    cache = CacheContainer.from_persistent_cache_file(Path(__file__).parent / "output" / "dataset_cache.pkl", load_as_virtual_persistent_cache=True)
    model_name = "gpt-3.5-turbo"
    llm = OpenAIChat(
        model=model_name,
        config=Config(temperature=1.0),
        cache=cache,
        request_price_per_token=OPENAI_PRICING[model_name]["request_price_per_token"],
        response_price_per_token=OPENAI_PRICING[model_name]["response_price_per_token"],
        openai_rate_limiter=OpenAIRateLimiter(
            rpm = OPENAI_PRICING[model_name]["RPM"],
            tpm = OPENAI_PRICING[model_name]["TPM"]
        )
    )

    def objective(trial: optuna.Trial):
        if process == "tot":
            num_branches = trial.suggest_int("num_branches", 5, 20)
            improvement_levels = trial.suggest_int("improvement_levels", 1, 6)
            controller_factory = lambda: tot_controller(llm=llm, num_branches=num_branches, improvement_levels=improvement_levels, max_concurrent=1)
        elif process == "got":
            num_sort_branches = trial.suggest_int("num_sort_branches", 1, 10)
            num_merge_branches = trial.suggest_int("num_merge_branches", 5, 15)
            global_improvement_rounds = trial.suggest_int("global_improvement_rounds", 1, 3)
            controller_factory = lambda: got_controller(llm=llm, num_sort_branches=num_sort_branches, num_merge_branches=num_merge_branches, global_improvement_rounds=global_improvement_rounds, max_concurrent=1)
        return controller_factory
    
    study_measurement = StudyMeasurement(save_file_path=Path(__file__).parent / "output" / f"{process}_sorting_study.pkl")

    # optuna.delete_study(study_name=f"sorting_study_{process}_2", storage="sqlite:///db.sqlite3")
    optuna_study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=42),
        direction="minimize",
        storage="sqlite:///db.sqlite3",
        study_name=f"sorting_study_{process}_2",
        load_if_exists=True
    )

    study = Study(
        optuna_study=optuna_study,
        metrics=[mistakes_penalized],
        dataset_evaluator=DatasetEvaluator(
            calculate_score=calculate_score,
            dataloader_factory=dataloader_factory,
            parameters=parameters,
            save_cache_on_completion_to=cache,
        ),
        max_concurrent=20,
        study_measurement=study_measurement,
        save_study_measurement_after_each_trial=True
    )
    study.set_objective(objective)
    study.optimize(n_trials=50)
    study_measurement.save(Path(__file__).parent / "output" / f"{process}_sorting_study_measurement.pkl")
    cache.save_persistent_cache(Path(__file__).parent / "output" / "dataset_cache.pkl")
    study_measurement.to_excel(Path(__file__).parent / "output" / f"{process}_sorting_study_measurement.xlsx")
    study_measurement.best_run.to_excel(
        Path(__file__).parent / "output" / f"{process}_sorting_study_best_run.xlsx",
        maps_for_measurements={
            "mean": np.mean,
            "sum": np.sum
        }
    )

if __name__ == "__main__":
    # sorting_study("tot")
    sorting_study("got")