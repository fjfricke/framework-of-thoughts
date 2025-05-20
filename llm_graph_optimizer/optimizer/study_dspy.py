# study_dspy.py
from __future__ import annotations
import asyncio
import nest_asyncio
import time
import json
import gzip
import pathlib
from typing import Sequence, Tuple, List, Dict

import dspy
from dspy import Predict
from dspy.teleprompt.copro_optimizer import (
    BasicGenerateInstruction,
    GenerateInstructionGivenAttempts,
)

from llm_graph_optimizer.measurement.dataset_measurement import (
    ScoreParameter,
)
from llm_graph_optimizer.measurement.study_measurement import (
    StudyMeasurement,
)
from llm_graph_optimizer.optimizer.dataset_evaluator import DatasetEvaluator
from llm_graph_optimizer.operations.llm_operations.dspy.shared_prompt_llm_operation import (
    SharedPromptLLMOperation as SP,
)


# ----------------------------------------------------------------------
# helper – run async in sync code (Jupyter-safe)
def _run(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError as e:
        if "event loop" in str(e):
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(coro)
        raise


# ----------------------------------------------------------------------
class DSPyPromptStudy:
    """
    Search for better *instruction / prefix* prompts with DSPy
    (CoPro-style) but evaluate each candidate with your DatasetEvaluator.

    One group_id  ➜  one prompt to optimise.
    """

    # ------------------------------------------------------------------
    def __init__(
        self,
        *,
        # evaluation side ------------------------------------------------
        dataset_evaluator: DatasetEvaluator,
        metrics: Sequence[ScoreParameter],
        max_concurrent_eval: int = 4,
        # prompt-generation side ----------------------------------------
        group_id: str,
        seed_instruction: str = '',
        seed_prefix: str = '',
        prompt_lm,                           # e.g. dspy.OpenAI(...)
        breadth: int = 8,                    # prompts per round
        depth: int = 4,                      # how many rounds
        keep_top: int = 10,                  # history size
        temperature: float = 1.4,
        # measurement logging -------------------------------------------
        study_measurement: StudyMeasurement | None = None,
        save_history_dir: pathlib.Path | None = None,
    ):
        self.dataset_evaluator = dataset_evaluator
        self.metrics = list(metrics)
        self.max_concurrent_eval = max_concurrent_eval

        self.group_id = group_id
        self.seed_instruction = seed_instruction
        self.seed_prefix = seed_prefix

        self.prompt_lm = prompt_lm
        self.breadth = breadth
        self.depth = depth
        self.keep_top = keep_top
        self.temperature = temperature

        self.history: List[Tuple[str, str, float]] = []  # (instr, prefix, score)
        self.best_prompt: Tuple[str, str, float] | None = None

        self.study_measurement = study_measurement
        self.save_history_dir = (
            pathlib.Path(save_history_dir) if save_history_dir else None
        )
        if self.save_history_dir:
            self.save_history_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_evaluator.controller_factory()  # register SharedPromptLLMOperation

    # ------------------------------------------------------------------
    def _generate_candidates(self) -> List[Tuple[str, str]]:
        """Use DSPy to propose a batch of new (instruction,prefix) pairs."""
        if self.seed_instruction == '' and self.seed_prefix == '':
            pred = SP._registry[self.group_id]
            adapter = dspy.ChatAdapter()
            format_field_description = adapter.format_field_description(pred.signature)
            format_field_structure = adapter.format_field_structure(pred.signature)
            format_task_description = adapter.format_task_description(pred.signature)
            self.seed_instruction = format_field_description + format_field_structure + format_task_description

        with dspy.settings.context(lm=self.prompt_lm):
            if not self.history:  # first round
                gen = Predict(
                    BasicGenerateInstruction,
                    n=self.breadth,
                    temperature=self.temperature,
                )(basic_instruction=self.seed_instruction)
                instrs = gen.completions.proposed_instruction
                prefs = gen.completions.proposed_prefix_for_output_field
            else:  # subsequent rounds
                # turn history into CoPro-style “attempted_instructions”
                attempts = []
                # best→worst
                for i, (ins, pref, score) in enumerate(
                    sorted(self.history, key=lambda x: x[2], reverse=True), 1
                ):
                    attempts.extend(
                        [
                            f"Instruction #{i}: {ins}",
                            f"Prefix #{i}: {pref}",
                            f"Resulting Score #{i}: {score}",
                        ]
                    )

                gen = Predict(
                    GenerateInstructionGivenAttempts,
                    n=self.breadth,
                    temperature=self.temperature,
                )(attempted_instructions=attempts)
                instrs = gen.completions.proposed_instruction
                prefs = gen.completions.proposed_prefix_for_output_field

        return list(zip(instrs, prefs))

    # ------------------------------------------------------------------
    def _evaluate_prompt(
        self, instruction: str, prefix: str
    ) -> Tuple[Dict[ScoreParameter, float], float]:
        """Evaluate a single candidate on the whole dataset."""
        # --- patch registry *only for this evaluation* ---------------
        pred = SP._registry[self.group_id]
        sig = pred.signature.with_instructions(instruction)
        last_key = list(sig.fields.keys())[-1]
        sig = sig.with_updated_fields(last_key, prefix=prefix)
        pred.signature = sig

        scores = _run(
            self.dataset_evaluator.evaluate_dataset(
                max_concurrent=self.max_concurrent_eval
            )
        )
        # DatasetEvaluator returns {ScoreParameter: float}
        # choose optimisation target – here the *first* metric
        obj_score = scores[self.metrics[0]]

        if self.study_measurement:
            self.study_measurement.add_dataset_measurement(
                self.dataset_evaluator.dataset_measurement
            )

        return scores, obj_score

    # ------------------------------------------------------------------
    def run(self):
        for round_idx in range(self.depth):
            print(f"\n◎ round {round_idx+1}/{self.depth}")
            candidates = self._generate_candidates()

            round_results = []
            for instr, pref in candidates:
                _, sc = self._evaluate_prompt(instr, pref)
                round_results.append((instr, pref, sc))
                print(f"  {sc:7.4f}  |  {instr[:70]}")

            # update history & best
            self.history.extend(round_results)
            self.history.sort(key=lambda x: x[2], reverse=True)
            self.history = self.history[: self.keep_top]
            self.best_prompt = self.history[0]

            if self.save_history_dir:
                stamp = time.strftime("%Y%m%d_%H%M%S")
                with gzip.open(
                    self.save_history_dir / f"round_{round_idx+1}_{stamp}.json.gz",
                    "wt",
                    encoding="utf-8",
                ) as f:
                    json.dump(round_results, f, indent=2)

            if self.study_measurement:
                self.study_measurement.save()

        # finally replace global registry entry so future controllers use it
        best_instr, best_pref, _ = self.best_prompt
        pred = SP._registry[self.group_id]
        sig = pred.signature.with_instructions(best_instr)
        last = list(sig.fields.keys())[-1]
        sig = sig.with_updated_fields(last, prefix=best_pref)
        pred.signature = sig
        print("\n★ BEST PROMPT:", best_instr, best_pref)

        return self.best_prompt