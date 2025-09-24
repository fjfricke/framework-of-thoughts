import asyncio
import logging
from pathlib import Path

import numpy as np

from examples.doc_merge.dataloader import DocMergeDataloader, Split
from examples.doc_merge.programs.got import got_controller
from examples.doc_merge.programs.prompter_parser import improve_prompt_dspy
from examples.openai_pricing import OPENAI_PRICING
from llm_graph_optimizer.graph_of_operations.types import ReasoningState
from llm_graph_optimizer.language_models.cache.cache import CacheContainer
from llm_graph_optimizer.language_models.helpers.language_model_config import Config
from llm_graph_optimizer.language_models.helpers.openai_rate_limiter import OpenAIRateLimiter
from llm_graph_optimizer.language_models.openai_chat import OpenAIChat
from llm_graph_optimizer.measurement.dataset_measurement import DatasetEvaluatorParameters, ScoreParameter
from llm_graph_optimizer.measurement.process_measurement import ProcessMeasurement
from llm_graph_optimizer.optimizer.dataset_evaluator import DatasetEvaluator

dataset_path = Path(__file__).parent / "dataset" / "documents.csv"

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

redundancy_score = ScoreParameter(
    name="redundancy",
    confidence_interval_width=0.95
)

retention_score = ScoreParameter(
    name="retention",
    confidence_interval_width=0.95
)

f1_score = ScoreParameter(
    name="f1_score",
    confidence_interval_width=0.95,
    # acceptable_ci_width=0.05
)

parameters = DatasetEvaluatorParameters(
    score_parameters=[redundancy_score, retention_score, f1_score],
    min_runs=10,
)

def calculate_score(reasoning_state: ReasoningState, measurement: ProcessMeasurement, ground_truth: None) -> float:
    return {
        redundancy_score: reasoning_state["redundancies"][0],
        retention_score: reasoning_state["retentions"][0],
        f1_score: reasoning_state["f1_scores"][0]
    }


def run_dataset_evaluation(dspy_or_optuna: str, original_or_optimized: str, split: Split):
    dataloader_factory = lambda: DocMergeDataloader(dataset_path=dataset_path, execution_mode=split, split=0.5, seed=42)
    cache = CacheContainer.from_persistent_cache_file(
        file_path=Path(__file__).parent / "output" / "cache.pkl",
        load_as_virtual_persistent_cache=True,
        skip_on_file_not_found=True
    )

    model = "gpt-3.5-turbo"
    openai_rate_limiter = OpenAIRateLimiter(
        rpm=OPENAI_PRICING[model]["RPM"],
        tpm=OPENAI_PRICING[model]["TPM"]
    )
    llm_generator = lambda temperature: OpenAIChat(
        model=model,
        config=Config(temperature=temperature),
        cache=cache,
        request_price_per_token=OPENAI_PRICING[model]["request_price_per_token"],
        response_price_per_token=OPENAI_PRICING[model]["response_price_per_token"],
        openai_rate_limiter=openai_rate_limiter
    )
    llm_gen = llm_generator(1)
    llm_score = llm_generator(0)

    num_merges = 4
    keep_best_merges = 3
    num_aggregations = 5
    num_improvements = 10
    optimized_improve_prompter = None
    if original_or_optimized == "optimized":
        if dspy_or_optuna == "optuna":
            num_merges = 4
            keep_best_merges = 3
            num_aggregations = 4
            num_improvements = 9
        elif dspy_or_optuna == "dspy":
            optimized_instruction = "Generate a succinct, informative, and harmonized summary of the provided non-disclosure agreements (NDAs) by meticulously blending insights from both the accompanying summaries and the original documents. Your output should encapsulate critical details while enhancing clarity and legibility, minimizing any excessive language. Highlight and comport essential legal terms appropriately, ensuring the intent and key clauses remain conspicuous. Finalize your summary formatted within the designated <Merged> and </Merged> tags aimed at promoting swift understanding and seamless usage of the key information presented."
            tag = "<Merged>"
            def improve_prompt_dspy_with_optimized_instruction(summaries, docs):
                return improve_prompt_dspy(summaries, docs, optimized_instruction, tag)
            optimized_improve_prompter = improve_prompt_dspy_with_optimized_instruction
        else:
            raise ValueError("Invalid dspy or optuna mode")
    elif original_or_optimized == "original":
        pass
    else:
        raise ValueError("Invalid original or optimized mode")
    
    controller_factory = lambda: got_controller(
        llm_gen=llm_gen,
        llm_score=llm_score,
        num_merges=num_merges,
        keep_best_merges=keep_best_merges,
        num_aggregations=num_aggregations,
        num_improvements=num_improvements,
        max_concurrent=1,
        optimized_improve_prompter=optimized_improve_prompter
    )
    dataset_evaluator = DatasetEvaluator(
        controller_factory=controller_factory,
        calculate_score=calculate_score,
        dataloader_factory=dataloader_factory,
        parameters=parameters,
        save_cache_on_completion_to=cache
    )
    scores = asyncio.run(dataset_evaluator.evaluate_dataset(max_concurrent=10))
    dataset_measurement = dataset_evaluator.dataset_measurement
    dataset_measurement.to_excel(Path(__file__).parent / "output" / "dataset_measurement.xlsx", maps_for_measurements={"mean": np.mean, "sum": np.sum})
    dataset_measurement.save(Path(__file__).parent / "output" / "dataset_measurement.pkl")
    print(scores)

if __name__ == "__main__":
    ORIGINAL_OR_OPTIMIZED = "optimized"
    DSPY_OR_OPTUNA = "dspy"
    SPLIT = Split.VALIDATION
    run_dataset_evaluation(DSPY_OR_OPTUNA, ORIGINAL_OR_OPTIMIZED, SPLIT)