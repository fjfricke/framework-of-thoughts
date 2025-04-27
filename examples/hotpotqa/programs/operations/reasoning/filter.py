from llm_graph_optimizer.graph_of_operations.types import StateSetFailed, StateSetFailedType


def filter_function(answers: list[str | StateSetFailedType], decomposition_scores: list[float | StateSetFailedType]) -> dict[str, any]:
    answers = [answer for answer in answers if answer is not StateSetFailed]
    decomposition_scores = [decomposition_score for decomposition_score in decomposition_scores if decomposition_score is not StateSetFailed]
    if not answers or not decomposition_scores:
        raise ValueError("No answers or decomposition scores")
    answer, decomposition_score = max(zip(answers, decomposition_scores), key=lambda x: x[1])
    return {"answer": answer, "decomposition_score": decomposition_score}

def filter_function_with_scaling_and_shifting(answers: list[str | StateSetFailedType], decomposition_scores: list[float | StateSetFailedType], scaling_factors: list[float], shifting_factors: list[float]) -> dict[str, any]:
    # Filter out StateSetFailed entries and keep track of valid indices
    valid_indices = [i for i, (answer, score) in enumerate(zip(answers, decomposition_scores)) if answer is not StateSetFailed and score is not StateSetFailed]

    valid_answers = [answers[i] for i in valid_indices]
    valid_scores = [decomposition_scores[i] for i in valid_indices]
    valid_scaling_factors = [scaling_factors[i] for i in valid_indices]
    valid_shifting_factors = [shifting_factors[i] for i in valid_indices]

    # Apply scaling and shifting to decomposition scores
    adjusted_scores = [score * scale + shift for score, scale, shift in zip(valid_scores, valid_scaling_factors, valid_shifting_factors)]

    # Find the answer with the highest adjusted score
    best_answer, best_score = max(zip(valid_answers, adjusted_scores), key=lambda x: x[1])

    return {"answer": best_answer, "decomposition_score": best_score}
