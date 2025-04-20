from dataclasses import dataclass
from sys import maxsize
from typing import Callable, Iterator
import numpy as np
from scipy.stats import t
from tqdm import tqdm
import asyncio

from llm_graph_optimizer.controller.controller import ControllerFactory
from llm_graph_optimizer.graph_of_operations.types import ReasoningState
from llm_graph_optimizer.measurement.process_measurement import ProcessMeasurement

Map = Callable[[list[float]], float]

@dataclass
class DatasetEvaluatorParameters:
    min_runs: int = 1
    max_runs: int = maxsize
    confidence_level: float = None              # e.g., [0.95, 0.99]
    acceptable_ci_width: float = None           # e.g., [0.02, 0.01]

class DatasetEvaluator:
    def __init__(
        self,
        controller_factory: ControllerFactory,
        calculate_score: Callable[[ReasoningState, ProcessMeasurement, any], float],
        dataloader: Iterator[tuple[ReasoningState, any]],
        parameters: DatasetEvaluatorParameters = DatasetEvaluatorParameters(),
    ):
        self.controller_factory = controller_factory
        self.calculate_score = calculate_score
        self.dataloader = dataloader
        self.parameters = parameters

    def _confidence_interval_width(self, values: list[float], confidence: float) -> float:
        n = len(values)
        if n < 2:
            return float('inf')
        std_err = np.std(values, ddof=1) / np.sqrt(n)
        t_score = t.ppf((1 + confidence) / 2, df=n - 1)
        return 2 * t_score * std_err

    async def evaluate_dataset(self, max_concurrent: int = 1, map: list[Map] = [np.mean]) -> list[float]:
        scores = []

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
                score = self.calculate_score(output_reasoning_state, measurement, ground_truth)
                scores.append(score)

                if self.parameters.confidence_level is not None:
                    ci_width = self._confidence_interval_width(scores, self.parameters.confidence_level)
                mean_score = np.mean(scores)
                mapped_scores = [m(scores) for m in map]
                description = f"Iteration {i}: mean = {mean_score:.4f}, mapped scores = {mapped_scores}"
                if self.parameters.confidence_level is not None:
                    description += f", CI width = {ci_width:.4f}"
                pbar.set_description(description)

                if (
                    self.parameters.confidence_level is not None and
                    self.parameters.acceptable_ci_width is not None and
                    len(scores) >= self.parameters.min_runs
                ):
                    confident = ci_width <= self.parameters.acceptable_ci_width

                    if confident:
                        stop_event.set()

                pbar.update(1)
                queue.task_done()

        queue = asyncio.Queue(maxsize=max_concurrent + 1)

        with tqdm(total=total) as pbar:
            producer_task = asyncio.create_task(producer(queue))
            workers = [worker(queue) for _ in range(max_concurrent)]
            await asyncio.gather(producer_task, *workers)

        return [m(scores) for m in map]
