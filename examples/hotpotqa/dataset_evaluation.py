from pathlib import Path

import numpy as np

from examples.hotpotqa.programs.dataloader import HotpotQADatasetLoader, Split
from examples.hotpotqa.programs.operations.utils import calculate_exact_match_score, calculate_f1_score
from examples.hotpotqa.programs.probtree import probtree_controller
from examples.openai_pricing import OPENAI_PRICING
from llm_graph_optimizer.graph_of_operations.types import ReasoningState
from llm_graph_optimizer.language_models.cache.cache import CacheContainer
from llm_graph_optimizer.language_models.helpers.language_model_config import Config
from llm_graph_optimizer.language_models.helpers.openai_rate_limiter import OpenAIRateLimiter
from llm_graph_optimizer.language_models.openai_chat_with_logprobs import OpenAIChatWithLogprobs
from llm_graph_optimizer.measurement.dataset_measurement import DatasetEvaluatorParameters, ScoreParameter
from llm_graph_optimizer.measurement.process_measurement import ProcessMeasurement
from llm_graph_optimizer.optimizer.dataset_evaluator import DatasetEvaluator

import logging
logging.getLogger().setLevel(logging.ERROR)

accuracy_score = ScoreParameter(
    name="accuracy",
)

f1_score = ScoreParameter(
    name="f1",
    confidence_interval_width=0.95,
)

precision_score = ScoreParameter(
    name="precision",
    confidence_interval_width=0.95,
)

recall_score = ScoreParameter(
    name="recall",
    confidence_interval_width=0.95,
)

parameters = DatasetEvaluatorParameters(
    min_runs=10,  # Should be > 10 when using the early stopping criterion from the acceptable_ci_width parameter
    # max_runs=950,  # Can be set for an additional early stopping criterion, used for validation in the paper
    score_parameters=[accuracy_score, f1_score, precision_score, recall_score]
)

def calculate_score(reasoning_state: ReasoningState | None, measurement: ProcessMeasurement, ground_truth: str) -> dict[ScoreParameter, float]:
    if reasoning_state is None:
        return {accuracy_score: 0.0, f1_score: 0.0, precision_score: 0.0, recall_score: 0.0}
    answer = reasoning_state["answer"]
    try:
        f1, precision, recall = calculate_f1_score(answer, ground_truth)
        accuracy = calculate_exact_match_score(answer, ground_truth)
    except Exception as e:
        logging.error(f"Error calculating exact match score: {e}")
        f1, precision, recall = 0.0, 0.0, 0.0
        accuracy = 0.0
    return {accuracy_score: accuracy, f1_score: f1, precision_score: precision, recall_score: recall}

def test_dataset_evaluation(dataset: str = "hotpotqa", hqdt_prompter: str = None):
    import asyncio

    if dataset == "hotpotqa":
        dataset_path = Path(__file__).parent / "dataset" / "HotpotQA" / "hotpot_dev_fullwiki_v1.json"
    elif dataset == "musique":
        dataset_path = Path(__file__).parent / "dataset" / "HotpotQA" / "musique_full_v1.0_dev.jsonl"
    else:
        raise ValueError(f"Dataset {dataset} not supported")
    dataloader_factory_with_split = lambda split: HotpotQADatasetLoader(execution_mode=split, dataset_path=dataset_path, split=0.5, seed=42, total_size=2000)  # Loads the dataset and sets training and test split.

    dataloader = lambda: dataloader_factory_with_split(Split.VALIDATION)
    cache = CacheContainer.from_persistent_cache_file(Path(__file__).parent / "output" / "cache.pkl", skip_on_file_not_found=True, load_as_virtual_persistent_cache=True)
    model = "gpt-4.1-mini"
    llm = OpenAIChatWithLogprobs(
        model=model,
        config=Config(temperature=0.0),
        request_price_per_token=OPENAI_PRICING[model]["request_price_per_token"],
        response_price_per_token=OPENAI_PRICING[model]["response_price_per_token"],
        cache=cache,
        openai_rate_limiter=OpenAIRateLimiter(
            rpm = OPENAI_PRICING[model]["RPM"]*2,
            tpm = OPENAI_PRICING[model]["TPM"]*2
        )
    )
    if hqdt_prompter is None:
        controller_factory = lambda: probtree_controller(llm, n_retrieved_docs=5, max_concurrent=1, dataset=dataset)
    else:
        controller_factory = lambda: probtree_controller(llm, n_retrieved_docs=5, max_concurrent=1, dataset=dataset, hqdt_prompter=hqdt_prompter)

    dataset_evaluator = DatasetEvaluator(
        controller_factory=controller_factory,
        calculate_score=calculate_score,
        dataloader_factory=dataloader,
        parameters=parameters,
        save_cache_on_completion_to=cache
    )
    scores = asyncio.run(dataset_evaluator.evaluate_dataset(max_concurrent=40))
    dataset_measurement = dataset_evaluator.dataset_measurement
    dataset_measurement.to_excel(Path(__file__).parent / "output" / "dataset_measurement.xlsx", maps_for_measurements={"mean": np.mean, "sum": np.sum})
    dataset_measurement.save(Path(__file__).parent / "output" / "dataset_measurement.pkl")
    cache.save_persistent_cache(Path(__file__).parent / "output" / "cache.pkl")
    print(scores)
if __name__ == "__main__":

    def optimized_hqdt_prompt(question: str) -> dict[str, str]:
        instructions = """"Given a multi-hop question that involves multiple reasoning steps, your task is to systematically decompose it into a complete hierarchical tree of simpler, atomic subquestions. Each node in the tree should be a subquestion that isolates a single reasoning step, ensuring that each one is independently answerable and collectively they help solve the original question via step-by-step reasoning. Represent this decomposition as a well-structured JSON object where each key is a complex or intermediate question and its corresponding value is a list of direct atomic subquestions needed to answer it. To show dependencies across subquestions, use consistent referencing (e.g., \"#1\" for the first subquestion) whenever one subquestion relies on an earlier one. The decomposition should continue recursively until every path ends in foundational, fact-based atomic questions that do not require further breakdown. Your resulting JSON tree must explicitly capture this layered structure prioritizing clarity, logical flow, and completeness for transparent multi-hop reasoning."""
        return [{"role": "system", "content": instructions}, {"role": "user", "content": question}]
    
    test_dataset_evaluation(dataset="hotpotqa", hqdt_prompter=optimized_hqdt_prompt)  # hotpotqa or musique