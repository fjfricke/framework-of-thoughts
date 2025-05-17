import asyncio
from typing import Callable
from optuna import Study as OptunaStudy, Trial

from llm_graph_optimizer.controller.controller import ControllerFactory
from llm_graph_optimizer.measurement.dataset_measurement import ScoreParameter
from llm_graph_optimizer.measurement.study_measurement import StudyMeasurement
from llm_graph_optimizer.optimizer.dataset_evaluator import DatasetEvaluator

class Study:
    """
    Manages optimization of a dataset evaluator using Optuna.

    This class integrates Optuna for hyperparameter optimization and coordinates
    the evaluation of datasets using a dataset evaluator. It supports concurrent
    evaluations and measurement tracking.

    Attributes:
        optuna_study (OptunaStudy): Optuna study object for optimization.
        metrics (list[ScoreParameter]): List of metrics to optimize.
        dataset_evaluator (DatasetEvaluator): Evaluator for the dataset.
        max_concurrent (int): Maximum number of concurrent evaluations.
        study_measurement (StudyMeasurement): Measurement object for tracking study results.
        save_study_measurement_after_each_trial (bool): Whether to save measurements after each trial.
    """

    def __init__(self,
                 optuna_study: OptunaStudy,
                 metrics: list[ScoreParameter],
                 dataset_evaluator: DatasetEvaluator,
                 max_concurrent: int = 10,
                 study_measurement: StudyMeasurement = None,
                 save_study_measurement_after_each_trial: bool = False,
                 ):
        """
        Initialize a Study instance.

        Args:
            optuna_study (OptunaStudy): Optuna study object for optimization.
            metrics (list[ScoreParameter]): List of metrics to optimize.
            dataset_evaluator (DatasetEvaluator): Evaluator for the dataset.
            max_concurrent (int, optional): Maximum number of concurrent evaluations. Defaults to 10.
            study_measurement (StudyMeasurement, optional): Measurement object for tracking study results. Defaults to None.
            save_study_measurement_after_each_trial (bool, optional): Whether to save measurements after each trial. Defaults to False.
        """
        self.optuna_study = optuna_study
        self.optuna_study.set_metric_names([metric.name for metric in metrics])
        self.metrics = metrics
        self.dataset_evaluator = dataset_evaluator
        self.max_concurrent = max_concurrent
        self.objective = None
        self.study_measurement = study_measurement
        self.save_study_measurement_after_each_trial = save_study_measurement_after_each_trial

    def _run_dataset_evaluator_async(self):
        def safe_asyncio_run(coro):
            try:
                return asyncio.run(coro)
            except RuntimeError as e:
                if "cannot be called from a running event loop" in str(e):
                    import nest_asyncio
                    nest_asyncio.apply()
                    loop = asyncio.get_event_loop()
                    return loop.run_until_complete(coro)
                else:
                    raise
        return safe_asyncio_run(self.dataset_evaluator.evaluate_dataset(max_concurrent=self.max_concurrent))

    def set_objective(self, trial_controller_factory: Callable[[Trial], ControllerFactory]):
        """
        Set the objective function for the Optuna study.

        Args:
            trial_controller_factory (Callable[[Trial], ControllerFactory]): Factory to create controllers for each trial. See examples for implementation.
        """
        def objective(trial: Trial):
            controller_factory = trial_controller_factory(trial)
            self.dataset_evaluator.set_controller_factory(controller_factory)
            scores = self._run_dataset_evaluator_async()
            if self.study_measurement:
                self.study_measurement.add_dataset_measurement(self.dataset_evaluator.dataset_measurement)
            if self.save_study_measurement_after_each_trial:
                self.study_measurement.save()
            return tuple(scores[metric] for metric in self.metrics)
        self.objective = objective

    def optimize(self, n_trials: int):
        """
        Run the optimization process.

        Args:
            n_trials (int): Number of trials to run.

        Raises:
            ValueError: If the study measurement save file path is not set.
        """
        if not self.study_measurement.save_file_path and self.save_study_measurement_after_each_trial:
            raise ValueError("Study measurement save file path is not set. Please set it to a valid path.")
        self.optuna_study.optimize(self.objective, n_trials=n_trials)
        if self.study_measurement:
            self.study_measurement.set_best_run(self.best_trial.number)

    @property
    def best_trial(self):
        """
        Get the best trial from the Optuna study.

        Returns:
            Trial: The best trial.
        """
        return self.optuna_study.best_trial
