from sys import maxsize
from typing import Callable, Iterator
import numpy as np
from scipy.stats import t
from tqdm import tqdm
import asyncio

from llm_graph_optimizer.controller.controller import ControllerFactory
from llm_graph_optimizer.graph_of_operations.types import ReasoningState
from llm_graph_optimizer.language_models.cache.cache import CacheContainer
from llm_graph_optimizer.measurement.dataset_measurement import DatasetEvaluatorParameters, DatasetMeasurement, GlobalEvaluationMeasurements, Score, ScoreParameter
from llm_graph_optimizer.measurement.process_measurement import ProcessMeasurement
from llm_graph_optimizer.operations.helpers.exceptions import GraphExecutionFailed

class DatasetEvaluator:
    """
    Evaluates a dataset using a graph of operations and calculates scores.

    This class manages the evaluation of a dataset by executing a graph of operations, measuring performance, and calculating scores
    based on user-defined metrics.

    Attributes:
        controller_factory (ControllerFactory): Function that creates a controller.
        calculate_score (Callable[[ReasoningState | None, ProcessMeasurement, any], dict[ScoreParameter, float]]):
            Function to calculate scores from reasoning states and measurements.
        dataloader_factory (Callable[[], Iterator[tuple[ReasoningState, any]]]): Function that creates a dataloader.
        parameters (DatasetEvaluatorParameters): Parameters for dataset evaluation.
        dataset_measurement (DatasetMeasurement): Stores measurements and scores for the dataset.
        save_cache_on_completion_to (CacheContainer): Cache container to save persistent cache upon completion.
    """

    def __init__(
        self,
        calculate_score: Callable[[ReasoningState | None, ProcessMeasurement, any], dict[ScoreParameter, float]],
        dataloader_factory: Callable[[], Iterator[tuple[ReasoningState, any]]],
        parameters: DatasetEvaluatorParameters,
        controller_factory: ControllerFactory = None,
        save_cache_on_completion_to: CacheContainer = None,
    ):
        """
        Initialize a DatasetEvaluator instance.

        Args:
            calculate_score (Callable): Function to calculate scores from reasoning states and measurements.
            dataloader_factory (Callable): Factory to create a dataloader.
            parameters (DatasetEvaluatorParameters): Parameters for dataset evaluation.
            controller_factory (ControllerFactory, optional): Factory to create controllers for graph execution. Defaults to None.
            save_cache_on_completion_to (CacheContainer, optional): Cache container to save persistent cache upon completion. Defaults to None.
        """
        self.controller_factory = controller_factory
        self.calculate_score = calculate_score
        self.dataloader_factory = dataloader_factory
        self.dataloader = None
        self.parameters = parameters
        self.dataset_measurement = None
        self.save_cache_on_completion_to = save_cache_on_completion_to

    def set_controller_factory(self, controller_factory: ControllerFactory):
        self.controller_factory = controller_factory

    def _confidence_interval_width(self, values: list[float], confidence: float) -> float:
        n = len(values)
        if n < 2:
            return np.float64('inf')
        std_err = np.std(values, ddof=1) / np.sqrt(n)
        t_score = t.ppf((1 + confidence) / 2, df=n - 1)
        return 2 * t_score * std_err

    async def evaluate_dataset(self, max_concurrent: int = 1) -> dict[ScoreParameter, np.float64]:
        """
        Evaluate the dataset by executing reasoning states through the graph of operations.

        Args:
            max_concurrent (int, optional): Maximum number of concurrent workers. Defaults to 1.

        Returns:
            dict[ScoreParameter, np.float64]: Calculated scores for each score parameter.

        Raises:
            ValueError: If the controller factory is not set.
        """
        self.dataset_measurement = DatasetMeasurement()
        self.dataset_measurement.add_dataset_evaluator_parameters(self.parameters)

        if self.controller_factory is None:
            raise ValueError("Controller factory is not set")
        self.dataloader = self.dataloader_factory()
        score_params_and_values_up_to_current_iteration: dict[ScoreParameter, list[np.float64]] = {score_param: np.array([], dtype=np.float64) for score_param in self.parameters.score_parameters}

        if hasattr(self.dataloader, '__len__'):
            total = min(self.parameters.max_runs, len(self.dataloader))
        else:
            total = self.parameters.max_runs if self.parameters.max_runs is not maxsize else None

        stop_event = asyncio.Event()

        async def producer(queue: asyncio.Queue):
            for i, item in enumerate(self.dataloader):
                if stop_event.is_set():
                    while not queue.empty():
                        queue.get_nowait()
                        queue.task_done()
                    break
                if i == self.parameters.max_runs:
                    break
                await queue.put((i, item))
            # Add termination signals for each worker
            for _ in range(max_concurrent):
                await queue.put(None)

        iteration_counter = {"value": 0}

        async def worker(queue: asyncio.Queue):
            while True:
                item = await queue.get()
                if item is None:  # Termination signal
                    break

                i, (input_reasoning_state, ground_truth) = item
                controller = self.controller_factory()
                try:
                    output_reasoning_state, measurement = await controller.execute(input_reasoning_state)
                except GraphExecutionFailed as e:
                    output_reasoning_state = None
                    measurement = e.process_measurement
                if self.save_cache_on_completion_to is not None:
                    self.save_cache_on_completion_to.clear_process_cache()
                self.dataset_measurement.add_measurement(i, measurement)
                scores = self.calculate_score(output_reasoning_state, measurement, ground_truth)
                for param, score in scores.items():
                    score_params_and_values_up_to_current_iteration[param] = np.append(score_params_and_values_up_to_current_iteration[param], np.float64(score))

                iteration_counter["value"] += 1
                current_iteration = iteration_counter["value"]

                description = f"Iteration {current_iteration}: "
                for param, values in score_params_and_values_up_to_current_iteration.items():
                    ci_width = self._confidence_interval_width(values, param.confidence_interval_width)
                    mean_score = np.mean(values)
                    mapped_score = param.map(values)
                    if param.map == np.mean:
                        description += f"({param.name} = {mapped_score:.4f}, CI width = {ci_width:.4f}), "
                    else:
                        description += f"({param.name} = {mapped_score:.4f}, mean = {mean_score:.4f}, CI width = {ci_width:.4f}), "
                    if param.acceptable_ci_width is not None and self.parameters.min_runs <= current_iteration:
                        confident = bool(ci_width <= param.acceptable_ci_width)
                        if confident:
                            stop_event.set()
                pbar.set_description(description)

                pbar.update(1)
                queue.task_done()

        queue = asyncio.Queue(maxsize=max_concurrent + 1)

        with tqdm(total=total) as pbar:
            producer_task = asyncio.create_task(producer(queue))
            workers = [worker(queue) for _ in range(max_concurrent)]
            await asyncio.gather(producer_task, *workers)

        self.dataset_measurement.add_global_evaluation_measurement(GlobalEvaluationMeasurements(pbar.format_dict["elapsed"]))
        if self.save_cache_on_completion_to is not None:
            self.save_cache_on_completion_to.save_persistent_cache()
        scores = [Score(param.name, param.map(values), self._confidence_interval_width(values, param.confidence_interval_width)) for param, values in score_params_and_values_up_to_current_iteration.items()]
        self.dataset_measurement.add_scores(scores)
        return {param: param.map(values) for param, values in score_params_and_values_up_to_current_iteration.items()}
