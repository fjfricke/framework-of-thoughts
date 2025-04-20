import asyncio
import logging

from examples.sorting.programs.io import io_controller
from examples.sorting.programs.cot import cot_controller
from examples.sorting.programs.tot import tot_controller

logging.basicConfig(level=logging.DEBUG)

async def run_controllers(input_list, expected_output):
    for controller in [io_controller(), cot_controller(), tot_controller()]:
    # for controller in [cot_controller()]:
    # for controller in [tot_controller(num_branches=3, improvement_levels=2)]:
        answer, process_measurement = await controller.execute(input={"input_list": input_list, "expected_output": expected_output})
        controller.graph_of_operations.snapshot.view(show_multiedges=False, show_values=True, show_keys=True, show_state=True)
        print(answer)
        print(process_measurement)

asyncio.run(run_controllers([5, 1, 0, 1, 2, 0, 4, 8, 1, 9, 5, 1, 3, 3, 9, 7], [0, 0, 1, 1, 1, 1, 2, 3, 3, 4, 5, 5, 7, 8, 9, 9]))