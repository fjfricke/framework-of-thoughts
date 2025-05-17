# dspy_study.py
from __future__ import annotations

import asyncio
import nest_asyncio
from pathlib import Path
from typing import Callable, Iterable, List, Dict, Any, Optional, Tuple

import dspy
from dspy import Example

from llm_graph_optimizer.controller.controller import Controller
from llm_graph_optimizer.graph_of_operations.types import ReasoningState
from llm_graph_optimizer.measurement.process_measurement import ProcessMeasurement

def safe_asyncio_run(coro):
    """
    • If no event-loop is running  →  use asyncio.run(coro)
    • If we’re already inside a loop (Jupyter, DSPy threads)  →
      apply nest_asyncio & run_until_complete.
    """
    try:
        return asyncio.run(coro)
    except RuntimeError as e:
        if "event loop is running" in str(e):
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(coro)
        raise


class DSPyPromptStudy:
    """
    Tune all DSPy-visible prompts in a controller, then evaluate.

    Parameters
    ----------
    controller_factory
        Callable returning a **fresh Controller** each call.
    train_loader_factory, eval_loader_factory
        Factories that yield `(inputs_dict, ground_truth)` tuples.
    input_keys
        Keys inside `inputs_dict` that go into the controller (and thus into
        DSPy Examples).  Everything else in inputs_dict is ignored.
    calculate_score
        Callable that converts `(reasoning_state, measurement, ground_truth)`
        into a scalar to maximise.
    save_path
        Optional path to store/load the tuned prompts JSON.
    compile_kwargs
        Extra args for the DSPy optimiser (e.g. `auto="light"`).
    """

    def __init__(
        self,
        *,
        controller_factory: Callable[[], Controller],
        train_loader_factory: Callable[[], Iterable[Tuple[dict, Any]]],
        eval_loader_factory: Callable[[], Iterable[Tuple[dict, Any]]] | None = None,
        optimizer_factory: Callable[[], dspy.Optimizer],
        input_keys: List[str],
        calculate_score: Callable[
            [ReasoningState, ProcessMeasurement | None, Any], float
        ],
        save_path: Optional[Path] = None,
        compile_kwargs: Dict[str, Any] | None = None,
        eval_kwargs: Dict[str, Any] | None = None,
    ):
        self.controller_factory = controller_factory
        self.train_loader_factory = train_loader_factory
        self.eval_loader_factory = eval_loader_factory
        self.optimizer_factory = optimizer_factory
        self.input_keys = input_keys
        self.calculate_score = calculate_score
        self.compile_kwargs = compile_kwargs or {"auto": "light", "num_threads": 8}
        self.eval_kwargs = eval_kwargs or {}
        self.save_path = Path(save_path) if save_path else None

        self._compiled: dspy.Module | None = None
        self._eval_scores: List[float] | None = None

    # -----------------------------------------------------------------
    def run(self) -> dspy.Module:
        self._compile()
        if self.eval_loader_factory:
            self._evaluate()
        return self._compiled

    @property
    def eval_scores(self):
        return self._eval_scores

    # -----------------------------------------------------------------
    # internal
    # -----------------------------------------------------------------
    def _compile(self):
        dev_set = self._examples_from_loader(self.train_loader_factory())

        # ---------------- DSPy wrapper around the controller ----------
        study = self  # capture for inner class

        class _GraphModule(dspy.Module):
            def __init__(self):
                super().__init__()
                self._factory = study.controller_factory  # deepcopy-safe
                self._expose_inner_predictors()

            # expose each inner dspy.Module as an attribute
            def _expose_inner_predictors(self):
                ctrl = self._factory()
                for idx, node in enumerate(ctrl.graph_of_operations._graph.nodes):
                    if isinstance(node, dspy.Module):
                        setattr(self, f"sub_{idx}", node)

            def forward(self, **inputs):
                ctrl = self._build_ctrl_with_tuned_prompts()
                rs, meas = safe_asyncio_run(ctrl.execute(inputs))
                return dspy.Prediction(reasoning_state=rs, measurement=meas)

            async def aforward(self, **inputs):
                ctrl = self._build_ctrl_with_tuned_prompts()
                rs, meas = await ctrl.execute(inputs)
                return dspy.Prediction(reasoning_state=rs, measurement=meas)
            
            # ------------ critical: make sure controller sees MUTATED prompts
            def _build_ctrl_with_tuned_prompts(self) -> Controller:
                from llm_graph_optimizer.operations.llm_operations.dspy.shared_prompt_llm_operation import (
                        SharedPromptLLMOperation as SP,
                    )
                # copy tuned Predicts from exposed sub-modules into registry
                for attr in dir(self):
                    if attr.startswith("sub_"):
                        node = getattr(self, attr)
                        if hasattr(node, "predict"):
                            for gid in list(SP._registry.keys()):
                                if SP._registry[gid].signature == node.predict.signature:
                                    SP._registry[gid] = node.predict
                return self._factory()

        # ---------------- metric uses the user-provided callable -------
        def metric(gold, pred, trace=None):
            return study.calculate_score(
                pred.reasoning_state,
                pred.measurement,
                getattr(gold, "ground_truth", None),
            )

        optimiser: dspy.COPRO = self.optimizer_factory(metric=metric, **self.compile_kwargs)

        if self.save_path and self.save_path.exists():
            print("▶ Loading tuned prompts from", self.save_path)
            self._compiled = dspy.Module.load(self.save_path)
        else:
            print("▶ Compiling prompts with DSPy …")
            self._compiled = optimiser.compile(_GraphModule(), trainset=dev_set, eval_kwargs=self.eval_kwargs)
            if self.save_path:
                self.save_path.parent.mkdir(parents=True, exist_ok=True)
                self._compiled.save(self.save_path)
                print("✔ Tuned prompts saved →", self.save_path)

        self._patch_shared_prompt_registry()

    # copy the mutated Predict objects into SharedPromptLLMOperation’s registry
    def _patch_shared_prompt_registry(self):
        from llm_graph_optimizer.operations.llm_operations.dspy.shared_prompt_llm_operation import SharedPromptLLMOperation

        for _, pred in self._compiled.named_predictors():
            # assume each predictor lives under exactly one group_id in registry
            for gid, p in SharedPromptLLMOperation._registry.items():
                if p is pred:  # already patched
                    break
            else:
                # not in registry yet → add
                SharedPromptLLMOperation._registry["autopool_" + str(id(pred))] = pred

    # -----------------------------------------------------------------
    def _evaluate(self):
        assert self._compiled, "run compile first"
        async_prog = dspy.asyncify(self._compiled)

        async def _run():
            scores = []
            for inputs, gt in self.eval_loader_factory():
                mini = {k: inputs[k] for k in self.input_keys}
                pred = await async_prog(**mini)
                scores.append(
                    self.calculate_score(
                        pred.reasoning_state, pred.measurement, gt
                    )
                )
            return scores

        self._eval_scores = asyncio.run(_run())
        print(
            f"mean score on eval: {sum(self._eval_scores) / len(self._eval_scores):.4f}"
        )

    # -----------------------------------------------------------------
    def _examples_from_loader(self, loader):
        exs = []
        for inputs, gt in loader:
            inp_subset = {k: inputs[k] for k in self.input_keys}
            ex = Example(**inp_subset, ground_truth=gt).with_inputs(*self.input_keys)
            exs.append(ex)
        return exs