from pathlib import Path

import numpy as np
import optuna

from examples.hotpotqa.programs.dataloader import HotpotQADatasetLoader
from examples.hotpotqa.programs.old_studies.dynamic_probtree import dynamic_probtree_controller
from examples.hotpotqa.programs.operations.utils import calculate_exact_match_score, calculate_f1_score
from examples.hotpotqa.programs.probtree import probtree_controller
from examples.openai_pricing import OPENAI_PRICING
from llm_graph_optimizer.graph_of_operations.types import ReasoningState
from llm_graph_optimizer.language_models.cache.cache import CacheContainer
from llm_graph_optimizer.language_models.helpers.language_model_config import Config
from llm_graph_optimizer.language_models.openai_chat_with_logprobs import OpenAIChatWithLogprobs
from llm_graph_optimizer.measurement.dataset_measurement import DatasetEvaluatorParameters, ScoreParameter
from llm_graph_optimizer.measurement.process_measurement import ProcessMeasurement
from llm_graph_optimizer.measurement.study_measurement import StudyMeasurement
from llm_graph_optimizer.optimizer.dataset_evaluator import DatasetEvaluator
from llm_graph_optimizer.optimizer.study_optuna import Study

dataset_path = Path(__file__).parent.parent / "dataset" / "HotpotQA" / "hotpot_dev_fullwiki_v1.json"
dataloader = lambda: HotpotQADatasetLoader(dataset_path)

accuracy_score = ScoreParameter(
    name="accuracy",
)

f1_score = ScoreParameter(
    name="f1",
    confidence_interval_width=0.95,
    acceptable_ci_width=0.05
)

precision_score = ScoreParameter(
    name="precision"
)

recall_score = ScoreParameter(
    name="recall"
)

parameters = DatasetEvaluatorParameters(
    min_runs=10,
    # max_runs=5,
    # max_runs=1000,
    score_parameters=[accuracy_score, f1_score, precision_score, recall_score]
)

def calculate_score(reasoning_state: ReasoningState | None, measurement: ProcessMeasurement, ground_truth: str) -> dict[ScoreParameter, float]:
    if reasoning_state is None:
        return {accuracy_score: 0.0, f1_score: 0.0, precision_score: 0.0, recall_score: 0.0}
    answer = reasoning_state["answer"]
    f1, precision, recall = calculate_f1_score(answer, ground_truth)
    accuracy = calculate_exact_match_score(answer, ground_truth)
    return {accuracy_score: accuracy, f1_score: f1, precision_score: precision, recall_score: recall}

def test_dataset_evaluation():
    import asyncio
    cache = CacheContainer.from_persistent_cache_file(Path(__file__).parent.parent / "output" / "hotpotqa_dataset_cache.pkl", skip_on_file_not_found=True)
    llm = OpenAIChatWithLogprobs(
        model="gpt-4.1-mini",
        config=Config(temperature=0.0),
        request_price_per_token=OPENAI_PRICING["gpt-4.1-mini"]["request_price_per_token"],
        response_price_per_token=OPENAI_PRICING["gpt-4.1-mini"]["response_price_per_token"],
        cache=cache
    )
    controller_factory = lambda: dynamic_probtree_controller(llm, max_depth=3, min_branch_certainty_threshold=0.5, n_retrieved_docs=10)
    dataset_evaluator = DatasetEvaluator(
        controller_factory=controller_factory,
        calculate_score=calculate_score,
        dataloader_factory=dataloader,
        parameters=parameters
    )
    scores = asyncio.run(dataset_evaluator.evaluate_dataset(max_concurrent=20))
    print(scores)


def dynamic_probtree_study():
    cache = CacheContainer.from_persistent_cache_file(Path(__file__).parent.parent / "output" / "hotpotqa_study_cache.pkl", skip_on_file_not_found=True)
    llm = OpenAIChatWithLogprobs(
        model="gpt-4.1-mini",
        config=Config(temperature=0.0),
        request_price_per_token=OPENAI_PRICING["gpt-4.1-mini"]["request_price_per_token"],
        response_price_per_token=OPENAI_PRICING["gpt-4.1-mini"]["response_price_per_token"],
        cache=cache
    )
    
    def controller_factory_with_params(max_depth: int, min_branch_certainty_threshold: float, n_retrieved_docs: int):
        return dynamic_probtree_controller(llm, max_depth=max_depth, min_branch_certainty_threshold=min_branch_certainty_threshold, n_retrieved_docs=n_retrieved_docs)
    
    def objective(trial: optuna.Trial):
        max_depth = trial.suggest_int("max_depth", 0, 4)
        n_retrieved_docs = trial.suggest_int("n_retrieved_docs", 1, 10)
        if max_depth > 0:
            min_branch_certainty_threshold = trial.suggest_float("min_branch_certainty_threshold", 0.0, 1.0)
        else:
            min_branch_certainty_threshold = 0.5 # not used
            trial.set_user_attr("min_branch_certainty_threshold", None)
        controller_factory = lambda: controller_factory_with_params(max_depth=max_depth, min_branch_certainty_threshold=min_branch_certainty_threshold, n_retrieved_docs=n_retrieved_docs)
        return controller_factory

    optuna.delete_study(study_name="hotpotqa_dynamic_probtree", storage="sqlite:///db.sqlite3")
    optuna_study = optuna.create_study(
        direction="maximize",
        storage="sqlite:///db.sqlite3",
        study_name="hotpotqa_dynamic_probtree",
        load_if_exists=True
    )
    # last_trial = sorted(optuna_study.trials, key=lambda t: t.number)[-1]
    # optuna_study.enqueue_trial(last_trial.params)
    study_measurement = StudyMeasurement()
    study = Study(
        optuna_study=optuna_study,
        metrics=[f1_score],
        dataset_evaluator=DatasetEvaluator(
            calculate_score=calculate_score,
            dataloader_factory=dataloader,
            parameters=parameters,
            save_cache_on_completion_to=cache,
        ),
        max_concurrent=20,
        study_measurement=study_measurement
    )
    study.set_objective(objective)
    study.optimize(n_trials=100)
    cache.save_persistent_cache(Path(__file__).parent.parent / "output" / "hotpotqa_study_cache.pkl")
    study_measurement.to_excel(Path(__file__).parent.parent / "output" / "hotpotqa_study_measurement.xlsx")
    study_measurement.best_run.to_excel(
        Path(__file__).parent.parent / "output" / "hotpotqa_study_best_run.xlsx",
        maps_for_measurements={
            "mean": np.mean,
            "sum": np.sum
        }
    )

def probtree_study():
    cache = CacheContainer.from_persistent_cache_file(Path(__file__).parent.parent / "output" / "hotpotqa_probtree_study_cache.pkl", skip_on_file_not_found=True)
    llm = OpenAIChatWithLogprobs(
        model="gpt-4.1-mini",
        config=Config(temperature=0.0),
        request_price_per_token=OPENAI_PRICING["gpt-4.1-mini"]["request_price_per_token"],
        response_price_per_token=OPENAI_PRICING["gpt-4.1-mini"]["response_price_per_token"],
        cache=cache
    )

    def controller_factory_with_params(n_retrieved_docs: int):
        return probtree_controller(llm, n_retrieved_docs=n_retrieved_docs)

    def objective(trial: optuna.Trial):
        n_retrieved_docs = trial.suggest_int("n_retrieved_docs", 1, 10)
        controller_factory = lambda: controller_factory_with_params(n_retrieved_docs=n_retrieved_docs)
        return controller_factory

    # optuna.delete_study(study_name="hotpotqa_probtree", storage="sqlite:///db.sqlite3")
    optuna_study = optuna.create_study(
        direction="maximize",
        storage="sqlite:///db.sqlite3",
        study_name="hotpotqa_probtree",
        load_if_exists=True
    )
    # last_trial = sorted(optuna_study.trials, key=lambda t: t.number)[-1]
    # optuna_study.enqueue_trial(last_trial.params)
    study_measurement = StudyMeasurement()
    study = Study(
        optuna_study=optuna_study,
        metrics=[f1_score],
        dataset_evaluator=DatasetEvaluator(
            calculate_score=calculate_score,
            dataloader_factory=dataloader,
            parameters=parameters,
            save_cache_on_completion_to=cache,
        ),
        max_concurrent=20,
        study_measurement=study_measurement
    )
    study.set_objective(objective)
    study.optimize(n_trials=10)
    cache.save_persistent_cache(Path(__file__).parent.parent / "output" / "hotpotqa_probtree_study_cache.pkl")
    study_measurement.to_excel(Path(__file__).parent.parent / "output" / "hotpotqa_probtree_study_measurement.xlsx")
    study_measurement.best_run.to_excel(
        Path(__file__).parent.parent / "output" / "hotpotqa_probtree_study_best_run.xlsx",
        maps_for_measurements={
            "mean": np.mean,
            "sum": np.sum
        }
    )
if __name__ == "__main__":
    # dynamic_probtree_study()
    # probtree_study()
    test_dataset_evaluation()