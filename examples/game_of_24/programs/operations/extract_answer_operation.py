from llm_graph_optimizer.graph_of_operations.graph_partitions import GraphPartitions
from llm_graph_optimizer.graph_of_operations.types import ManyToOne, ReasoningState
from llm_graph_optimizer.measurement.measurement import Measurement
from llm_graph_optimizer.operations.abstract_operation import AbstractOperation
import re


class ExtractAnswerOperation(AbstractOperation):
    def __init__(self, params: dict = None, name: str = None):
        input_types = {"expressions": ManyToOne[str]}
        output_types = {"answer": str}
        super().__init__(input_types, output_types, params, name)

    async def _execute(self, partitions: GraphPartitions, input_reasoning_states: ReasoningState) -> tuple[ReasoningState, Measurement | None]: 
        expressions: list[str] = input_reasoning_states.get("expressions", [])

        def parse_expression(expr: str) -> tuple[str, str]:
            parts = expr.split("=", 1)
            lhs = parts[0].strip()
            rhs = parts[1].strip() if len(parts) > 1 else ""
            # Extract first numeric token from RHS (e.g., "24" from "24" or "24 (left: ...)")
            m = re.search(r"[-+]?\d+(?:\.\d+)?", rhs)
            rhs_num = m.group(0) if m else rhs
            return lhs, rhs_num

        if not expressions:
            return {"answer": ""}, None

        current_lhs, current_rhs = parse_expression(expressions[0])
        for expr in expressions[1:]:
            next_lhs, next_rhs = parse_expression(expr)
            # Replace the previous RHS value as a standalone number in the next LHS with the previous LHS wrapped in parentheses
            pattern = rf"(?<!\d){re.escape(current_rhs)}(?!\d)"
            composed_lhs = re.sub(pattern, f"({current_lhs})", next_lhs, count=1)
            current_lhs, current_rhs = composed_lhs, next_rhs

        answer = f"{current_lhs} = {current_rhs}"
        return {"answer": answer}, None

if __name__ == "__main__":
    import asyncio
    expressions = ["9 + 9 = 18", "8 * 18 = 144", "144 / 6 = 24"]
    print(asyncio.run(ExtractAnswerOperation()._execute(None, {"expressions": expressions})))