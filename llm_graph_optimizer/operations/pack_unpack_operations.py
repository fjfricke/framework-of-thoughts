from llm_graph_optimizer.graph_of_operations.graph_partitions import GraphPartitions
from llm_graph_optimizer.operations.abstract_operation import AbstractOperation


class PackOperation(AbstractOperation):
    def __init__(self, input_types: dict[str | int, type], output_key: str | int, params: dict = None, name: str = None):
        output_types = {output_key: dict[str | int, any]}
        self.output_key = output_key
        super().__init__(input_types, output_types, params, name)

    async def _execute(self, partitions: GraphPartitions, input_reasoning_states: dict[str | int, any]) -> dict[str | int, any]:
        return {self.output_key: input_reasoning_states}
    
    @property
    def corresponding_unpack_operation(self):
        return UnpackOperation(self.output_key, self.input_types, self.params, self.name)
    
class UnpackOperation(AbstractOperation):
    def __init__(self, input_key: str | int, output_types: dict[str | int, type], params: dict = None, name: str = None):
        input_types = {input_key: dict[str | int, any]}
        self.input_key = input_key
        super().__init__(input_types, output_types, params, name)

    async def _execute(self, partitions: GraphPartitions, input_reasoning_states: dict[str | int, any]) -> dict[str | int, any]:
        return input_reasoning_states[self.input_key]
    
    @property
    def corresponding_pack_operation(self):
        return PackOperation(self.output_types, self.input_key, self.params, self.name)
