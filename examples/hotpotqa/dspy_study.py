# study_run.py
import logging
from pathlib import Path
import dspy

from examples.hotpotqa.programs.probtree import probtree_controller
from examples.hotpotqa.dataset_evaluation import calculate_score, parameters, f1_score
from examples.hotpotqa.programs.dataloader import HotpotQADatasetLoader, Split
from examples.openai_pricing import OPENAI_PRICING
from llm_graph_optimizer.language_models.cache.cache import CacheContainer
from llm_graph_optimizer.language_models.helpers.language_model_config import Config
from llm_graph_optimizer.language_models.helpers.openai_rate_limiter import OpenAIRateLimiter
from llm_graph_optimizer.language_models.openai_chat_with_logprobs import OpenAIChatWithLogprobs
from llm_graph_optimizer.measurement.study_measurement import StudyMeasurement
from llm_graph_optimizer.optimizer.dataset_evaluator import DatasetEvaluator
from llm_graph_optimizer.optimizer.study_dspy import DSPyPromptStudy


logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

def dspy_study(dataset: str = "hotpotqa"):

    if dataset == "hotpotqa":
        dataset_path = Path(__file__).parent / "dataset" / "HotpotQA" / "hotpot_dev_fullwiki_v1.json"
    elif dataset == "musique":
        dataset_path = Path(__file__).parent / "dataset" / "HotpotQA" / "musique_full_v1.0_dev.jsonl"
    else:
        raise ValueError(f"Dataset {dataset} not supported")
    dataloader_factory_with_split = lambda split: HotpotQADatasetLoader(execution_mode=split, dataset_path=dataset_path, split=0.5, seed=42, total_size=2000)  # Loads the dataset and sets training and test split.


    cache = CacheContainer.from_persistent_cache_file(
        file_path=Path(__file__).parent / "output" / "cache.pkl",
        load_as_virtual_persistent_cache=True,
        skip_on_file_not_found=True
    )

    model = "gpt-4o-mini"

    openai_rate_limiter = OpenAIRateLimiter(
        rpm=OPENAI_PRICING[model]["RPM"]*0.8,
        tpm=OPENAI_PRICING[model]["TPM"]*0.8,
        max_estimated_response_tokens=1000
    )
    llm_generator = lambda model, temperature: OpenAIChatWithLogprobs(
        model=model,
        config=Config(temperature=temperature),
        cache=cache,
        request_price_per_token=OPENAI_PRICING[model]["request_price_per_token"],
        response_price_per_token=OPENAI_PRICING[model]["response_price_per_token"],
        openai_rate_limiter=openai_rate_limiter
    )
    llm = llm_generator(model, 0)

    controller_factory = lambda: probtree_controller(
        llm=llm,
        n_retrieved_docs=5,
        max_concurrent=1,
        use_dspy=True,
    )

    # dataloader factories
    train_dataloader = lambda: dataloader_factory_with_split(Split.TRAIN)

    dataset_evaluator = DatasetEvaluator(
        calculate_score=calculate_score,
        dataloader_factory=train_dataloader,
        parameters=parameters,
        save_cache_on_completion_to=cache,
        controller_factory=controller_factory
    )

    dspy_llm = dspy.LM('openai/gpt-4o-mini', temperature=1.6)

    study_measurement = StudyMeasurement(save_file_path=Path(__file__).parent / "output" / "dspy_study.pkl")

    study = DSPyPromptStudy(
        dataset_evaluator=dataset_evaluator,
        metrics=[f1_score],
        max_concurrent_eval=50,
        group_id="generate_hqdt",
        seed_instruction="",
        seed_prefix="",
        prompt_lm=dspy_llm,
        depth=8,
        keep_top=4,
        breadth=10,
        study_measurement=study_measurement,
        save_history_dir=Path(__file__).parent / "output" / "dspy_study"
    )

    program_star = study.run()  # noqa: F841
    study_measurement.save(Path(__file__).parent / "output" / "dspy_study_measurement.pkl")
    study_measurement.to_excel(Path(__file__).parent / "output" / "dspy_study_measurement.xlsx")

if __name__ == "__main__":
    dspy_study(dataset="musique")