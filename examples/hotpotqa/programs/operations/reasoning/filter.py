from llm_graph_optimizer.graph_of_operations.types import StateSetFailed, StateSetFailedType


def filter_function(answers: list[str | StateSetFailedType], decomposition_scores: list[float | StateSetFailedType]) -> dict[str, any]:
    answers = [answer for answer in answers if answer is not StateSetFailed]
    decomposition_scores = [decomposition_score for decomposition_score in decomposition_scores if decomposition_score is not StateSetFailed]
    if not answers or not decomposition_scores:
        raise ValueError("No answers or decomposition scores")
    answer, decomposition_score = max(zip(answers, decomposition_scores), key=lambda x: x[1])
    return {"answer": answer, "decomposition_score": decomposition_score}