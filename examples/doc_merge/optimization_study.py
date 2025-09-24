# study_run.py
import logging
from pathlib import Path
import optuna
import numpy as np

from examples.doc_merge.dataloader import DocMergeDataloader, Split
from examples.doc_merge.programs.got import got_controller
from examples.openai_pricing import OPENAI_PRICING
from llm_graph_optimizer.graph_of_operations.types import ReasoningState, StateSetFailed
from llm_graph_optimizer.language_models.cache.cache import CacheContainer
from llm_graph_optimizer.language_models.helpers.language_model_config import Config
from llm_graph_optimizer.language_models.helpers.openai_rate_limiter import OpenAIRateLimiter
from llm_graph_optimizer.language_models.openai_chat import OpenAIChat
from llm_graph_optimizer.measurement.dataset_measurement import DatasetEvaluatorParameters, ScoreParameter
from llm_graph_optimizer.measurement.process_measurement import ProcessMeasurement
from llm_graph_optimizer.measurement.study_measurement import StudyMeasurement
from llm_graph_optimizer.optimizer.dataset_evaluator import DatasetEvaluator
from llm_graph_optimizer.optimizer.study_optuna import Study


logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

dataset_path = Path(__file__).parent / "dataset" / "documents.csv"

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
    # acceptable_ci_width=0.05
)

cost_score = ScoreParameter(
    name="cost",
    confidence_interval_width=0.95,
    # acceptable_ci_width=0.1
)

parameters = DatasetEvaluatorParameters(
    score_parameters=[redundancy_score, retention_score, f1_score, cost_score],
    min_runs=10,
)

def calculate_score(reasoning_state: ReasoningState, measurement: ProcessMeasurement, ground_truth: None) -> dict[ScoreParameter, float]:

    cost = measurement.total_sequential_cost().with_process_cache.total_cost

    if reasoning_state is None or any(reasoning_state.get(score.name, 0) == StateSetFailed for score in [redundancy_score, retention_score, f1_score]):
        return {redundancy_score: 0.0, retention_score: 0.0, f1_score: 0.0, cost_score: cost}

    return {
        redundancy_score: reasoning_state["redundancies"][0],
        retention_score: reasoning_state["retentions"][0],
        f1_score: reasoning_state["f1_scores"][0],
        cost_score: cost
    }

def doc_merge_study():

    COST_BOUNDARY = 0.066

    dataloader_factory = lambda: DocMergeDataloader(dataset_path=dataset_path, execution_mode=Split.TRAIN, split=0.5, seed=42)

    cache = CacheContainer.from_persistent_cache_file(Path(__file__).parent / "output" / "cache.pkl", load_as_virtual_persistent_cache=True, skip_on_file_not_found=True)

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

    def objective(trial: optuna.Trial):
        # gen_temperature = trial.suggest_float("gen_temperature", 0.0, 1.5)
        num_merges = trial.suggest_int("num_merges", 1, 20)
        keep_best_merges = trial.suggest_int("keep_best_merges", 1, 10)
        num_aggregations = trial.suggest_int("num_aggregations", 1, 10)
        num_improvements = trial.suggest_int("num_improvements", 1, 30)
        controller_factory = lambda: got_controller(
            llm_gen=llm_generator(1),
            llm_score=llm_generator(0),
            num_merges=num_merges,
            keep_best_merges=keep_best_merges,
            num_aggregations=num_aggregations,
            num_improvements=num_improvements,
            max_concurrent=1,
        )
        return controller_factory
    
    def constraints(trial: optuna.Trial):
        cost_attr = trial.user_attrs.get(cost_score.name, 0)
        return [cost_attr - COST_BOUNDARY]

    study_measurement = StudyMeasurement(save_file_path=Path(__file__).parent / "output" / "doc_merge_study.pkl")

    optuna_study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=42, constraints_func=constraints),
        direction="maximize",
        storage="sqlite:///db.sqlite3",
        study_name="doc_merge_study_2",
        load_if_exists=True
    )

    study = Study(
        optuna_study=optuna_study,
        metrics=[f1_score],
        dataset_evaluator=DatasetEvaluator(
            calculate_score=calculate_score,
            dataloader_factory=dataloader_factory,
            parameters=parameters,
            save_cache_on_completion_to=cache,
        ),
        max_concurrent=10,
        study_measurement=study_measurement,
        save_study_measurement_after_each_trial=True
    )
    study.set_objective(objective)
    study.optimize(n_trials=50)
    study_measurement.save(Path(__file__).parent / "output" / "doc_merge_study_measurement.pkl")
    cache.save_persistent_cache(Path(__file__).parent / "output" / "cache.pkl")
    study_measurement.to_excel(Path(__file__).parent / "output" / "doc_merge_study_measurement.xlsx")
    study_measurement.best_run.to_excel(
        Path(__file__).parent / "output" / "doc_merge_study_best_run.xlsx",
        maps_for_measurements={"mean": np.mean, "sum": np.sum}
    )

if __name__ == "__main__":
    doc_merge_study()