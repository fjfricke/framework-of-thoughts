import asyncio
from llm_graph_optimizer.graph_of_operations.graph_of_operations import GraphOfOperations
from llm_graph_optimizer.operations.abstract_operation import AbstractOperation


class Controller:
    def __init__(self, graph_of_operations: GraphOfOperations, scheduler: callable[[GraphOfOperations], list[AbstractOperation]], max_concurrent: int = 3):
        self.graph_of_operations = graph_of_operations
        self.scheduler = scheduler
        # self.graph_over_time = []
        self.max_concurrent = max_concurrent

    async def execute(self, input: list[any]):
        """
        Execute operations in the graph using the scheduler and an async queue.
        :param input: Initial input for the graph.
        """
        # self.graph_over_time.append(self.graph_of_operations)

        # Initialize the graph with the input
        self.graph_of_operations.initialize_input(input)

        # Create an async queue for operations
        operation_queue = asyncio.Queue()

        # Track completed operations
        completed_operations = set()

        async def worker():
            """
            Worker function to process operations from the queue.
            """
            while True:
                operation = await operation_queue.get()
                try:
                    await operation.execute(self.graph_of_operations)
                    completed_operations.add(operation)
                finally:
                    operation_queue.task_done()

                # Re-run the scheduler and enqueue the next operations
                next_operations = [
                    op for op in self.scheduler(self.graph_of_operations)
                    if op not in completed_operations and op not in operation_queue._queue
                ]
                for op in next_operations:
                    await operation_queue.put(op)

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
