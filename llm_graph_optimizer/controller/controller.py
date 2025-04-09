import asyncio
from typing import Callable
from llm_graph_optimizer.graph_of_operations.graph_of_operations import GraphOfOperations
from llm_graph_optimizer.operations.abstract_operation import AbstractOperation
from llm_graph_optimizer.operations.helpers.exceptions import OperationFailed
from llm_graph_optimizer.operations.helpers.node_state import NodeState


class Controller:
    def __init__(self, graph_of_operations: GraphOfOperations, scheduler: Callable[[GraphOfOperations], list[AbstractOperation]], max_concurrent: int = 3):
        self.graph_of_operations = graph_of_operations
        self.scheduler = scheduler
        # self.graph_over_time = []
        self.max_concurrent = max_concurrent

    def initialize_input(self, input: dict[str, any]):
        self.graph_of_operations.start_node.node_state = NodeState.PROCESSABLE
        self.graph_of_operations.start_node.set_input_reasoning_states(input)

    async def execute(self, input: dict[str, any]):
        """
        Execute operations in the graph using the scheduler and an async queue.
        :param input: Initial input for the graph.
        """
        # self.graph_over_time.append(self.graph_of_operations)

        # Initialize the graph with the input
        self.initialize_input(input)

        # Create an async queue for operations
        operation_queue = asyncio.Queue()

        async def worker():
            """
            Worker function to process operations from the queue.
            """
            while True:
                operation = await operation_queue.get()
                try:
                    await operation.execute(self.graph_of_operations)
                    operation.node_state = NodeState.DONE
                except OperationFailed as e:
                    print(f"Operation {operation.name} failed. Error: {e}")
                    operation.node_state = NodeState.FAILED                    

                self.graph_of_operations.set_next_processable()
                # Re-run the scheduler and enqueue the next operations
                next_operations = [
                    op for op in self.scheduler(self.graph_of_operations)
                    if op in self.graph_of_operations.processable_nodes and op not in operation_queue._queue
                ]
                for op in next_operations:
                    await operation_queue.put(op)
                operation_queue.task_done()

        # Enqueue initial operations
        initial_operations = self.scheduler(self.graph_of_operations)
        for operation in initial_operations:
            await operation_queue.put(operation)

        # Start workers
        workers = [asyncio.create_task(worker()) for _ in range(self.max_concurrent)]

        # Wait for all tasks in the queue to be processed
        await operation_queue.join()

        # Cancel workers
        for worker_task in workers:
            worker_task.cancel()

        return self.graph_of_operations.get_input_reasoning_states(self.graph_of_operations.end_node)
