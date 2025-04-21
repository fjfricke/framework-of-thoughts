from typing import Callable, Iterator
import numpy as np
from scipy.stats import t
from tqdm import tqdm
import asyncio

from llm_graph_optimizer.controller.controller import ControllerFactory
from llm_graph_optimizer.graph_of_operations.types import ReasoningState
from llm_graph_optimizer.measurement.dataset_measurement import DatasetEvaluatorParameters, DatasetMeasurement, GlobalEvaluationMeasurements, Score, ScoreParameter
from llm_graph_optimizer.measurement.process_measurement import ProcessMeasurement

class DatasetEvaluator:
    def __init__(
        self,
        calculate_score: Callable[[ReasoningState, ProcessMeasurement, any], dict[ScoreParameter, float]],
        dataloader: Iterator[tuple[ReasoningState, any]],
        parameters: DatasetEvaluatorParameters,
        controller_factory: ControllerFactory = None,
    ):
        self.controller_factory = controller_factory
        self.calculate_score = calculate_score
        self.dataloader = dataloader
        self.parameters = parameters
        self.dataset_measurement = DatasetMeasurement()
        self.dataset_measurement.add_dataset_evaluator_parameters(parameters)

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
        if self.controller_factory is None:
            raise ValueError("Controller factory is not set")
        score_params_and_values_up_to_current_iteration: dict[ScoreParameter, list[np.float64]] = {score_param: np.array([], dtype=np.float64) for score_param in self.parameters.score_parameters}

        if hasattr(self.dataloader, '__len__'):
            total = min(self.parameters.max_runs, len(self.dataloader))
        else:
            total = self.parameters.max_runs

        stop_event = asyncio.Event()

        async def producer(queue: asyncio.Queue):
            for i, item in enumerate(self.dataloader):
                if i == self.parameters.max_runs or stop_event.is_set():
                    break
                await queue.put((i, item))

            # Add termination signals for each worker
            for _ in range(max_concurrent):
                await queue.put(None)

        async def worker(queue: asyncio.Queue):
            while True:
                item = await queue.get()
                if item is None:  # Termination signal
                    break

                i, (input_reasoning_state, ground_truth) = item
                controller = self.controller_factory()
                output_reasoning_state, measurement = await controller.execute(input_reasoning_state)
                self.dataset_measurement.add_measurement(i, measurement)
                scores = self.calculate_score(output_reasoning_state, measurement, ground_truth)
                for param, score in scores.items():
                    score_params_and_values_up_to_current_iteration[param] = np.append(score_params_and_values_up_to_current_iteration[param], np.float64(score))

                description = f"Iteration {i}: "
                for param, values in score_params_and_values_up_to_current_iteration.items():
                    ci_width = self._confidence_interval_width(values, param.confidence_interval_width)
                    mean_score = np.mean(values)
                    mapped_score = param.map(values)
                    description += f"mean = {mean_score:.4f}, CI width = {ci_width:.4f}, {param.name} = {mapped_score:.4f}, "
                    if param.acceptable_ci_width is not None and self.parameters.min_runs <= i + 1:
                        confident = ci_width <= param.acceptable_ci_width
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

        scores = [Score(param.name, param.map(values), self._confidence_interval_width(values, param.confidence_interval_width)) for param, values in score_params_and_values_up_to_current_iteration.items()]
        self.dataset_measurement.add_scores(scores)
        return {param: param.map(values) for param, values in score_params_and_values_up_to_current_iteration.items()}
