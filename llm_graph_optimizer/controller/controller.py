from __future__ import annotations

import asyncio
from typing import Callable

from llm_graph_optimizer.graph_of_operations.graph_of_operations import GraphOfOperations
from llm_graph_optimizer.graph_of_operations.snapshot_graph import SnapshotGraphs
from llm_graph_optimizer.graph_of_operations.types import ReasoningState
from llm_graph_optimizer.language_models.cache.cache import CacheContainer
from llm_graph_optimizer.measurement.process_measurement import ProcessMeasurement
from llm_graph_optimizer.operations.abstract_operation import AbstractOperation
from llm_graph_optimizer.operations.helpers.exceptions import GraphExecutionFailed, OperationFailed
from llm_graph_optimizer.operations.helpers.node_state import NodeState
import logging

class Controller:
    def __init__(self, graph_of_operations: GraphOfOperations, scheduler: Callable[[GraphOfOperations], list[AbstractOperation]], max_concurrent: int = 3, process_measurement: ProcessMeasurement = None, store_intermediate_snapshots: bool = False, save_to_cache_after_execution: CacheContainer = None):
        """
        Initialize the Controller with a graph of operations, a scheduler, and optional parameters.

        :param graph_of_operations: The graph of operations to be executed.
        :param scheduler: A callable that schedules operations from the graph.
        :param max_concurrent: Maximum number of concurrent operations.
        :param process_measurement: Optional measurement store for the individual process.
        :param store_intermediate_snapshots: Whether to store intermediate snapshots of the graph.
        """
        self.graph_of_operations = graph_of_operations
        self.scheduler = scheduler
        self.max_concurrent = max_concurrent
        self.logger = logging.getLogger(__name__)
        self.process_measurement = process_measurement
        self.store_intermediate_snapshots = store_intermediate_snapshots
        self.save_to_cache_after_execution = save_to_cache_after_execution
        if self.store_intermediate_snapshots:
            self.intermediate_snapshots = SnapshotGraphs()

    @classmethod
    def factory(cls, **kwargs) -> ControllerFactoryWithParams:
        """
        Create a factory for the Controller class with pre-defined parameters.

        :param kwargs: Initial parameters for the Controller.
        :return: A callable factory that accepts additional parameters.
        """
        def factory_without_params(**later_kwargs) -> ControllerFactory:
            # Combine initial kwargs with later_kwargs
            combined_kwargs = {**kwargs, **later_kwargs}
            return cls(**combined_kwargs)

        return factory_without_params

    def _initialize_input(self, input: dict[str, any]):
        self.graph_of_operations.start_node.node_state = NodeState.PROCESSABLE
        self.graph_of_operations.start_node.set_input_reasoning_states(input)

    async def execute(self, input: dict[str, any], debug_params: dict[str, bool] = {}) -> tuple[ReasoningState, ProcessMeasurement]:
        """
        Execute operations in the graph using the scheduler and an async queue.

        :param input: Initial input reasoning state to the graph.
        :param debug_params: Optional debugging parameters dict. Keys can be:
            - "visualize_intermediate_graphs": Whether to visualize the intermediate graphs.
            - "raise_on_operation_failure": Whether to raise an exception on operation failure.
        :return: A tuple containing the final reasoning state and process measurements.
        """
        self.logger.debug("Initializing graph with input: %s", input)

        # Initialize the graph with the input
        self._initialize_input(input)

        if self.store_intermediate_snapshots:
            self.intermediate_snapshots.add_snapshot(self.graph_of_operations.snapshot)

        # Create an async queue for operations
        operation_queue = asyncio.Queue()

        queue_lock = asyncio.Lock()

        workers_failed_to_enqueue = {"value": 0}

        async def worker():
            """
            Worker function to process operations from the queue.
            """
            while True:
                operation = await operation_queue.get()
                if operation is not None:
                    operation.node_state = NodeState.PROCESSING
                if operation is None:  # Sentinel value to stop the worker
                    self.logger.debug("Worker received sentinel value. Adding sentinel back and exiting.")
                    operation_queue.task_done()
                    await operation_queue.put(None)
                    break
                self.logger.debug("Processing operation: %s", operation.name)
                try:
                    measurement_or_measurements_with_cache = await operation.execute(self.graph_of_operations)
                    if self.process_measurement:
                        self.process_measurement.add_measurement(operation, measurement_or_measurements_with_cache)
                    if debug_params.get("visualize_intermediate_graphs", False):
                        self.graph_of_operations.snapshot.visualize(show_multiedges=False, show_values=True, show_keys=True, show_state=True)
                    self.logger.debug("Operation %s completed successfully.", operation.name)
                    operation.node_state = NodeState.DONE
                except OperationFailed as e:
                    if self.process_measurement:
                        self.process_measurement.add_measurement(operation, e.measurement)
                    self.logger.warning("Operation %s failed. Error: %s", operation.name, e)
                    operation.node_state = NodeState.FAILED
                    if debug_params.get("raise_on_operation_failure", False):
                        raise e
                finally:
                    if self.store_intermediate_snapshots:
                        self.intermediate_snapshots.add_snapshot(self.graph_of_operations.snapshot)
                    operation_queue.task_done()

                if self.graph_of_operations.all_processed:
                    self.logger.debug("All operations processed. Adding sentinel to stop other workers. Then exiting.")
                    await operation_queue.put(None)
                    break

                async with queue_lock:
                    # Set next processable operations
                    self.graph_of_operations.set_next_processable()
                    self.logger.debug("Set next processable operations.")

                    # Re-run the scheduler and enqueue the next operations
                
                    next_operations = [
                        op for op in self.scheduler(self.graph_of_operations)
                        if op in self.graph_of_operations.processable_nodes and op not in operation_queue._queue
                    ]
                    self.logger.debug("Next operations to enqueue: %s", [op.name for op in next_operations])
                    if not next_operations and all(item is None for item in operation_queue._queue):
                        workers_failed_to_enqueue["value"] += 1
                        if workers_failed_to_enqueue["value"] >= self.max_concurrent:
                            self.logger.error("No executable operations found but not all done. Please check the graph structure here. Exiting.")
                            await operation_queue.put(None)
                            break
                    else:
                        workers_failed_to_enqueue["value"] = 0
                    for op in next_operations:
                        await operation_queue.put(op)

        # Enqueue initial operations
        initial_operations = self.scheduler(self.graph_of_operations)
        # self.graph_of_operations.view_graph_debug(show_keys=True, output_name=f"initial_debug_{time.time()}.html")
        self.logger.debug("Initial operations to enqueue: %s", [op.name for op in initial_operations])
        for operation in initial_operations:
            await operation_queue.put(operation)

        # Start workers
        workers = [asyncio.create_task(worker()) for _ in range(self.max_concurrent)]
        self.logger.debug("Started %d workers.", self.max_concurrent)

        try:
            await asyncio.gather(*workers)
        except Exception as e:
            self.logger.error("An exception occurred in one of the workers: %s", e)

        # Enqueue sentinel values to stop workers
        for _ in range(self.max_concurrent):
            await operation_queue.put(None)

        # Cancel workers
        for worker_task in workers:
            try:
                worker_task.cancel()
                await worker_task
            except asyncio.CancelledError:
                self.logger.debug("Worker task cancelled.")

        for worker_task in workers:
            worker_task.cancel()

        self.logger.debug("Returning final input reasoning states.")

        # Update snapshot graph in the process measurement
        if self.process_measurement:
            self.process_measurement.snapshot_graph = self.graph_of_operations.snapshot
        
        if self.save_to_cache_after_execution:
            self.save_to_cache_after_execution.save_persistent_cache()
        
        try:
            if self.graph_of_operations.end_node.node_state == NodeState.FAILED:
                self.logger.error("The output of the final operation failed for input %s.", input)
                return self.graph_of_operations.get_input_reasoning_states(self.graph_of_operations.end_node), self.process_measurement
        except ValueError as e:
            self.logger.error("The output of the final operation failed for input %s with error %s.", input, e)
            raise GraphExecutionFailed(e, process_measurement=self.process_measurement)
        
        return self.graph_of_operations.get_input_reasoning_states(self.graph_of_operations.end_node), self.process_measurement

ControllerFactory = Callable[[], Controller]
ControllerFactoryWithParams = Callable[..., Controller]