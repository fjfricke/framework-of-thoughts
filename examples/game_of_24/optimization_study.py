import logging
from pathlib import Path
from examples.game_of_24.dataloader import GameOf24Dataloader, Split
from examples.game_of_24.dataset_evaluation import get_score_from_reasoning_state
from examples.game_of_24.programs.tot import tot_controller
from examples.openai_pricing import OPENAI_PRICING
from llm_graph_optimizer.graph_of_operations.types import ReasoningState
from llm_graph_optimizer.language_models.cache.cache import CacheContainer
from llm_graph_optimizer.language_models.openai_chat import OpenAIChat, OpenAIRateLimiter
from llm_graph_optimizer.measurement.dataset_measurement import DatasetEvaluatorParameters, ScoreParameter
from llm_graph_optimizer.measurement.process_measurement import ProcessMeasurement
from llm_graph_optimizer.measurement.study_measurement import StudyMeasurement
from llm_graph_optimizer.optimizer.dataset_evaluator import DatasetEvaluator
from llm_graph_optimizer.optimizer.study_optuna import Study
import optuna
import numpy as np

logging.getLogger().setLevel(logging.ERROR)

success_score = ScoreParameter(
    name="success",
    confidence_interval_width=0.95,  # Take confidence interval width with caution as it is a binary variable.
)

cost_score = ScoreParameter(
    name="cost",
    confidence_interval_width=0.95,
)

parameters = DatasetEvaluatorParameters(
    min_runs=10,
    score_parameters=[success_score, cost_score]
)

def calculate_score(reasoning_state: ReasoningState, measurement: ProcessMeasurement, ground_truth: list[int]) -> dict[ScoreParameter, float]:

    cost = measurement.total_sequential_cost().with_process_cache.total_cost

    if reasoning_state is None:
        return {success_score: 0.0, cost_score: cost}
    
    return {success_score: get_score_from_reasoning_state(reasoning_state, ground_truth), cost_score: cost}

def game_of_24_study():
    COST_BOUNDARY = 0.205

    dataloader_factory = lambda: GameOf24Dataloader(execution_mode=Split.TRAIN)

    cache = CacheContainer.from_persistent_cache_file(Path(__file__).parent / "output" / "dataset_cache.pkl", load_as_virtual_persistent_cache=True)

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

    def objective(trial: optuna.Trial):
        num_examples = trial.suggest_int("num_examples", 4, 12)
        samples = [
            trial.suggest_int("samples_layer_1", 1, 5),
            trial.suggest_int("samples_layer_2", 1, 5),
            trial.suggest_int("samples_layer_3", 1, 5)
        ]
        max_first_keep_top_n = min(num_examples, 7)
        keep_top_n = [
            trial.suggest_int("keep_top_n_layer_1", 2, max_first_keep_top_n),
            trial.suggest_int("keep_top_n_layer_2", 2, 7),
        ]
        controller_factory = lambda: tot_controller(llm=llm, num_examples=num_examples, samples=samples, keep_top_n=keep_top_n, max_concurrent=40)
        return controller_factory
    
    def constraints(trial: optuna.Trial):
        cost_attr = trial.user_attrs.get(cost_score.name, 0)
        return [cost_attr - COST_BOUNDARY]

    study_measurement = StudyMeasurement(save_file_path=Path(__file__).parent / "output" / "game_of_24_study.pkl")

    optuna_study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=42, constraints_func=constraints),
        direction="maximize",
        storage="sqlite:///db.sqlite3",
        study_name="game_of_24_study",
        load_if_exists=True
    )

    study = Study(
        optuna_study=optuna_study,
        metrics=[success_score],
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
    study.optimize(n_trials=25)
    study_measurement.save(Path(__file__).parent / "output" / "game_of_24_study_measurement.pkl")
    cache.save_persistent_cache(Path(__file__).parent / "output" / "dataset_cache.pkl")
    study_measurement.to_excel(Path(__file__).parent / "output" / "game_of_24_study_measurement.xlsx")
    study_measurement.best_run.to_excel(
        Path(__file__).parent / "output" / "game_of_24_study_best_run.xlsx",
        maps_for_measurements={"mean": np.mean, "sum": np.sum}
    )

if __name__ == "__main__":
    game_of_24_study()