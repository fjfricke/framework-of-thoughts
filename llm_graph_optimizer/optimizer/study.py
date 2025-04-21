import asyncio
from typing import Callable
from optuna import Study as OptunaStudy, Trial

from llm_graph_optimizer.controller.controller import ControllerFactory
from llm_graph_optimizer.measurement.dataset_measurement import ScoreParameter
from llm_graph_optimizer.measurement.study_measurement import StudyMeasurement
from llm_graph_optimizer.optimizer.dataset_evaluator import DatasetEvaluator

class Study:
    def __init__(self,
                 optuna_study: OptunaStudy,
                 metrics: list[ScoreParameter],
                 dataset_evaluator: DatasetEvaluator,
                 max_concurrent: int = 10,
                 study_measurement: StudyMeasurement = None
                 ):
        self.optuna_study = optuna_study
        self.optuna_study.set_metric_names([metric.name for metric in metrics])
        self.metrics = metrics
        self.dataset_evaluator = dataset_evaluator
        self.max_concurrent = max_concurrent
        self.objective = None
        self.study_measurement = study_measurement
    def set_objective(self, trial_controller_factory: Callable[[Trial], ControllerFactory]):
        def objective(trial: Trial):
            controller_factory = trial_controller_factory(trial)
            self.dataset_evaluator.set_controller_factory(controller_factory)
            scores = asyncio.run(self.dataset_evaluator.evaluate_dataset(max_concurrent=self.max_concurrent))
            if self.study_measurement:
                self.study_measurement.add_dataset_measurement(self.dataset_evaluator.dataset_measurement)
            return tuple(scores[metric] for metric in self.metrics)
        self.objective = objective

    def optimize(self, n_trials: int):
        self.optuna_study.optimize(self.objective, n_trials=n_trials)
        if self.study_measurement:
            self.study_measurement.set_best_run(self.best_trial.number)

    @property
    def best_trial(self):
        return self.optuna_study.best_trial
