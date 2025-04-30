from pathlib import Path

import numpy as np
import optuna

from examples.hotpotqa.programs.dataloader import HotpotQADatasetLoader, Split
from examples.hotpotqa.programs.operations.utils import calculate_exact_match_score, calculate_f1_score
from examples.hotpotqa.programs.probtree import probtree_controller
from examples.openai_pricing import OPENAI_PRICING
from llm_graph_optimizer.graph_of_operations.types import ReasoningState
from llm_graph_optimizer.language_models.cache.cache import CacheContainer
from llm_graph_optimizer.language_models.helpers.language_model_config import Config
from llm_graph_optimizer.language_models.helpers.openai_rate_limiter import OpenAIRateLimiter
from llm_graph_optimizer.language_models.openai_chat_with_logprobs import OpenAIChatWithLogprobs
from llm_graph_optimizer.measurement.dataset_measurement import DatasetEvaluatorParameters, ScoreParameter
from llm_graph_optimizer.measurement.process_measurement import ProcessMeasurement
from llm_graph_optimizer.measurement.study_measurement import StudyMeasurement
from llm_graph_optimizer.optimizer.dataset_evaluator import DatasetEvaluator
from llm_graph_optimizer.optimizer.study import Study

import logging
logging.getLogger().setLevel(logging.ERROR)

dataset_path = Path(__file__).parent.parent / "dataset" / "HotpotQA" / "hotpot_dev_fullwiki_v1.json"
dataloader = lambda: HotpotQADatasetLoader(execution_mode=Split.TRAIN, dataset_path=dataset_path, split=0.3, seed=42)  # Loads the dataset and sets training and test split. Note that for the paper, the first 30% where used  as training and the last 30%*30% as test set (implementation error.) 

accuracy_score = ScoreParameter(
    name="accuracy",
)

f1_score = ScoreParameter(
    name="f1",
    confidence_interval_width=0.95,
    acceptable_ci_width=0.01  # This sets the early stopping acceptable confidence interval widthof each dataset evaluation
)

precision_score = ScoreParameter(
    name="precision"
)

recall_score = ScoreParameter(
    name="recall"
)

parameters = DatasetEvaluatorParameters(
    min_runs=10,  # Should be > 10 when using the early stopping criterion from the acceptable_ci_width parameter
    # max_runs=950,  # Can be set for an additional early stopping criterion
    score_parameters=[accuracy_score, f1_score, precision_score, recall_score]
)

def calculate_score(reasoning_state: ReasoningState | None, measurement: ProcessMeasurement, ground_truth: str) -> dict[ScoreParameter, float]:
    if reasoning_state is None:
        return {accuracy_score: 0.0, f1_score: 0.0, precision_score: 0.0, recall_score: 0.0}
    answer = reasoning_state["answer"]
    try:
        f1, precision, recall = calculate_f1_score(answer, ground_truth)
        accuracy = calculate_exact_match_score(answer, ground_truth)
    except Exception as e:
        logging.error(f"Error calculating exact match score: {e}")
        f1, precision, recall = 0.0, 0.0, 0.0
        accuracy = 0.0
    return {accuracy_score: accuracy, f1_score: f1, precision_score: precision, recall_score: recall}

def probtree_study():
    cache = CacheContainer.from_persistent_cache_file(Path(__file__).parent.parent / "output" / "cache.pkl", skip_on_file_not_found=True, load_as_virtual_persistent_cache=True)
    model_name = "gpt-4.1-mini"
    llm = OpenAIChatWithLogprobs(
        model=model_name,
        config=Config(temperature=0.0),
        request_price_per_token=OPENAI_PRICING[model_name]["request_price_per_token"],
        response_price_per_token=OPENAI_PRICING[model_name]["response_price_per_token"],  # Set these two for correct pricing measurements
        cache=cache,
        openai_rate_limiter=OpenAIRateLimiter(
            rpm = OPENAI_PRICING[model_name]["RPM"],
            tpm = OPENAI_PRICING[model_name]["TPM"]
        )  # Set this to use the OpenAI rate limiter
    )

    def controller_factory_with_params(n_retrieved_docs: int, scaling_factors: list[float], shifting_factors: list[float]):
        return probtree_controller(llm, n_retrieved_docs=n_retrieved_docs, scaling_factors=scaling_factors, shifting_factors=shifting_factors)

    def objective(trial: optuna.Trial):
        n_retrieved_docs = trial.suggest_int("n_retrieved_docs", 1, 5)
        scaling_factors = [1.0]
        shifting_factors = [0.0]
        for name in ["closedbook", "aggregate"]:
            scaling_factors.append(trial.suggest_float(f"scaling_factor_{name}", 0.8, 1.2))
            shifting_factors.append(trial.suggest_float(f"shifting_factor_{name}", -0.3, 0.3))
        controller_factory = lambda: controller_factory_with_params(n_retrieved_docs=n_retrieved_docs, scaling_factors=scaling_factors, shifting_factors=shifting_factors)
        return controller_factory

    #to restart a new study
    # optuna.delete_study(study_name="final_probtree_study_parallelism", storage="sqlite:///db.sqlite3")
    study_measurement = StudyMeasurement(save_file_path=Path(__file__).parent.parent / "output" / "final_probtree_study.pkl")


    optuna_study = optuna.create_study(
        direction="maximize",
        storage="sqlite:///db.sqlite3",
        study_name="final_probtree_study_parallelism",
        load_if_exists=True  # Set this to load the study from the database and continue the optimization when stopped (e.g. after a crash)
    )

    study = Study(
        optuna_study=optuna_study,
        metrics=[f1_score],
        dataset_evaluator=DatasetEvaluator(
            calculate_score=calculate_score,
            dataloader_factory=dataloader,
            parameters=parameters,
            save_cache_on_completion_to=cache,
        ),
        max_concurrent=40,  # Set this to the number of concurrent datapoints evaluated. This is not the same as the concurrency in the controller.
        study_measurement=study_measurement,
        save_study_measurement_after_each_trial=True
    )
    study.set_objective(objective)
    study.optimize(n_trials=50)
    study_measurement.save(Path(__file__).parent.parent / "output" / "final_probtree_study_measurement.pkl")
    cache.save_persistent_cache(Path(__file__).parent.parent / "output" / "cache.pkl")
    study_measurement.to_excel(Path(__file__).parent.parent / "output" / "final_probtree_study_measurement.xlsx")
    study_measurement.best_run.to_excel(
        Path(__file__).parent.parent / "output" / "final_probtree_study_best_run.xlsx",
        maps_for_measurements={
            "mean": np.mean,
            "sum": np.sum
        }
    )

def test_dataset_evaluation():
    import asyncio
    dataloader = lambda: HotpotQADatasetLoader(execution_mode=Split.VALIDATION, dataset_path=dataset_path, split=0.3, seed=42)
    cache = CacheContainer.from_persistent_cache_file(Path(__file__).parent.parent / "output" / "cache.pkl", skip_on_file_not_found=True, load_as_virtual_persistent_cache=True)
    llm = OpenAIChatWithLogprobs(
        model="gpt-4.1-mini",
        config=Config(temperature=0.0),
        request_price_per_token=OPENAI_PRICING["gpt-4.1-mini"]["request_price_per_token"],
        response_price_per_token=OPENAI_PRICING["gpt-4.1-mini"]["response_price_per_token"],
        cache=cache
    )
    controller_factory = lambda: probtree_controller(llm, n_retrieved_docs=8, scaling_factors=[1.0, 1.172188886586379, 0.8973330433368379], shifting_factors=[0.0, 0.05253448002817633, -0.16389013949880604])
    dataset_evaluator = DatasetEvaluator(
        controller_factory=controller_factory,
        calculate_score=calculate_score,
        dataloader_factory=dataloader,
        parameters=parameters
    )
    scores = asyncio.run(dataset_evaluator.evaluate_dataset(max_concurrent=20))
    print(scores)
if __name__ == "__main__":
    # probtree_study()
    test_dataset_evaluation()
