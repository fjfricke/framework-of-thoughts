# study_run.py
import logging
from pathlib import Path
import dspy

from examples.doc_merge.dataloader import DocMergeDataloader, Split
from examples.doc_merge.programs.got import got_controller
from examples.doc_merge.got_dataset_evaluation import calculate_score, parameters, f1_score
from examples.openai_pricing import OPENAI_PRICING
from llm_graph_optimizer.language_models.cache.cache import CacheContainer
from llm_graph_optimizer.language_models.helpers.language_model_config import Config
from llm_graph_optimizer.language_models.helpers.openai_rate_limiter import OpenAIRateLimiter
from llm_graph_optimizer.language_models.openai_chat import OpenAIChat
from llm_graph_optimizer.measurement.study_measurement import StudyMeasurement
from llm_graph_optimizer.optimizer.dataset_evaluator import DatasetEvaluator
from llm_graph_optimizer.optimizer.study_dspy import DSPyPromptStudy


logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

cache = CacheContainer.from_persistent_cache_file(
    file_path=Path(__file__).parent / "output" / "cache.pkl",
    load_as_virtual_persistent_cache=True,
    skip_on_file_not_found=True
)

model_gen = "gpt-3.5-turbo"
model_score = "gpt-3.5-turbo"
openai_rate_limiter = OpenAIRateLimiter(
    rpm=OPENAI_PRICING[model_gen]["RPM"],
    tpm=OPENAI_PRICING[model_gen]["TPM"],
    max_estimated_response_tokens=1000
)
llm_generator = lambda model, temperature: OpenAIChat(
    model=model,
    config=Config(temperature=temperature),
    cache=cache,
    request_price_per_token=OPENAI_PRICING[model]["request_price_per_token"],
    response_price_per_token=OPENAI_PRICING[model]["response_price_per_token"],
    openai_rate_limiter=openai_rate_limiter
)
llm_gen = llm_generator(model_gen, 1)
llm_score = llm_generator(model_score, 0)

controller_factory = lambda: got_controller(
    llm_gen=llm_gen,
    llm_score=llm_score,
    num_merges=3,
    keep_best_merges=2,
    num_aggregations=4,
    num_improvements=5,
    max_concurrent=10,
    use_dspy=True,
    save_to_cache_after_execution=cache
)

# dataloader factories
train_dataloader = lambda: DocMergeDataloader(
    execution_mode=Split.TRAIN,
    dataset_path=Path(__file__).parent / "dataset" / "documents.csv",
    split=0.5,
    seed=42,
)
eval_dataloader = lambda: DocMergeDataloader(
    execution_mode=Split.TEST,
    dataset_path=Path(__file__).parent / "dataset" / "documents.csv",
    split=0.5,
    seed=42,
)

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
    max_concurrent_eval=10,
    group_id="improve",
    seed_instruction="",
    seed_prefix="",
    prompt_lm=dspy_llm,
    depth=6,
    keep_top=8,
    breadth=8,
    study_measurement=study_measurement,
    save_history_dir=Path(__file__).parent / "output" / "dspy_study"
)

program_star = study.run()
study_measurement.save(Path(__file__).parent / "output" / "dspy_study_measurement.pkl")
study_measurement.to_excel(Path(__file__).parent / "output" / "dspy_study_measurement.xlsx")