import asyncio
import random
import time
from typing import Callable
from llm_graph_optimizer.graph_of_operations.graph_of_operations import GraphOfOperations
from llm_graph_optimizer.operations.abstract_operation import AbstractOperation
from llm_graph_optimizer.operations.helpers.exceptions import OperationFailed
from llm_graph_optimizer.operations.helpers.node_state import NodeState
import logging

class Controller:
    def __init__(self, graph_of_operations: GraphOfOperations, scheduler: Callable[[GraphOfOperations], list[AbstractOperation]], max_concurrent: int = 3):
        self.graph_of_operations = graph_of_operations
        self.scheduler = scheduler
        # self.graph_over_time = []
        self.max_concurrent = max_concurrent
        self.logger = logging.getLogger(__name__)

    def initialize_input(self, input: dict[str, any]):
        self.graph_of_operations.start_node.node_state = NodeState.PROCESSABLE
        self.graph_of_operations.start_node.set_input_reasoning_states(input)

    async def execute(self, input: dict[str, any], debug_params = {}):
        """
        Execute operations in the graph using the scheduler and an async queue.
        :param input: Initial input for the graph.
        """
        self.logger.debug("Initializing graph with input: %s", input)

        # Initialize the graph with the input
        self.initialize_input(input)

        # Create an async queue for operations
        operation_queue = asyncio.Queue()

        queue_lock = asyncio.Lock()

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
                    await operation.execute(self.graph_of_operations)
                    self.logger.debug("Operation %s completed successfully.", operation.name)
                    operation.node_state = NodeState.DONE
                except OperationFailed as e:
                    self.logger.error("Operation %s failed. Error: %s", operation.name, e)
                    operation.node_state = NodeState.FAILED
                    raise e
                finally:
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

        for worker_task in workers:
            try:
                worker_task.cancel()
                await worker_task
            except asyncio.CancelledError:
                self.logger.debug("Worker task cancelled.")

        # Cancel workers
        for worker_task in workers:
            worker_task.cancel()

        self.logger.debug("Returning final input reasoning states.")
        return self.graph_of_operations.get_input_reasoning_states(self.graph_of_operations.end_node)
