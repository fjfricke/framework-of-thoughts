import logging
from pathlib import Path
import optuna
import asyncio

from examples.test.test_controller import controller
from examples.test.test_dataset_evaluator import calculate_score, dataloader, parameters
from llm_graph_optimizer.language_models.cache.cache import CacheContainer
from llm_graph_optimizer.optimizer.dataset_evaluator import DatasetEvaluator

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

cache = CacheContainer.from_persistent_cache_file(Path(__file__).parent / "output" / "cache.pkl", skip_on_file_not_found=True)
controller_factory_with_optimization_parameters = lambda number_of_llm_nodes: controller(cache, number_of_llm_nodes=number_of_llm_nodes)


def objective(trial: optuna.Trial):
    number_of_llm_nodes = trial.suggest_int("number_of_llm_nodes", 1, 21)
    controller_factory = lambda: controller_factory_with_optimization_parameters(number_of_llm_nodes)
    dataset_evaluator = DatasetEvaluator(
        controller_factory=controller_factory,
        calculate_score=calculate_score,
        dataloader=dataloader,
        parameters=parameters
    )
    scores = asyncio.run(dataset_evaluator.evaluate_dataset(max_concurrent=10))
    return scores[0]

if __name__ == "__main__":
    optuna.delete_study(study_name="test_optimizer", storage="sqlite:///db.sqlite3")
    study = optuna.create_study(
        direction="maximize",
        storage="sqlite:///db.sqlite3",
        study_name="test_optimizer",
        )
    study.optimize(objective, n_trials=10)
    cache.save_persistent_cache(Path(__file__).parent / "output" / "cache.pkl")
    print(study.best_trial)
