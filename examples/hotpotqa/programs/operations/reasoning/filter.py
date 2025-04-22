def filter_function(answers: list[str], decomposition_scores: list[float]) -> dict[str, any]:
    answer, decomposition_score = max(zip(answers, decomposition_scores), key=lambda x: x[1])
    return {"answer": answer, "decomposition_score": decomposition_score}