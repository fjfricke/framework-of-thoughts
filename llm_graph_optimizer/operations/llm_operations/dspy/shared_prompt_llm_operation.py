import copy
import uuid
import dspy
from llm_graph_optimizer.operations.llm_operations.base_llm_operation import (
    BaseLLMOperation,
)
from typing import Any, Dict, Optional
from llm_graph_optimizer.graph_of_operations.types import ReasoningStateType
from llm_graph_optimizer.operations.llm_operations.dspy.meta_bridge import OperationModuleMeta


class SharedPromptLLMOperation(dspy.Module, BaseLLMOperation, metaclass=OperationModuleMeta):
    """
    One `group_id`  ➔  one shared `dspy.Predict` prompt.

    * The **first** instance created for a given group_id is the *master*:
      • builds the dspy.Predict     → DSPy can mutate the template.  
      • exposes parameters to the optimiser.

    * Every *subsequent* instance with the same group_id is a *mirror*:
      • re-uses the same dspy.Predict object.  
      • is **frozen** (DSPy skips it).  
      • still behaves like an independent BaseLLMOperation node in the graph.
    """

    # registry: group_id  ->  shared dspy.Predict object
    _registry: Dict[str, dspy.Predict] = {}

    # ------------------------------------------------------------------
    # constructor
    # ------------------------------------------------------------------
    def __init__(
        self,
        *,
        group_id: str,
        llm,
        parser,
        signature: dspy.Signature | str,
        input_types: ReasoningStateType,
        output_types: ReasoningStateType,
        # --- BaseLLMOperation standard args ---------------------------
        name: Optional[str] = None,
        cache_seed=None,
        use_cache: bool = True,
        params: Optional[dict] = None,
        # --- extra kwargs for dspy.Predict (stop, examples, …) --------
        predict_kwargs: Optional[Dict[str, Any]] = None,
    ):
        predict_kwargs = predict_kwargs or {}

        # ---------- obtain / create the shared dspy.Predict -----------
        if group_id not in self._registry:
            # master copy – create Predict object DSPy can tune
            shared_predict = dspy.Predict(
                signature,
                **predict_kwargs,
            )
            self._is_master = True
            self._registry[group_id] = shared_predict
        else:
            # mirror copy – re-use existing Predict
            shared_predict = self._registry[group_id]
            self._is_master = False

        # make it an attribute *before* dspy.Module.__init__,
        # so DSPy registers it as a sub-module
        self.predict = shared_predict
        self._group_id = group_id
        def _render_prompt(**kv):
            """
            1. Prefer predict.config["template"]  (MiPro & friends).
               • str          → one user message
               • callable     → call(**kv)
               • dict{sys,user} or list[dict] → turn into chat text
            2. Fallback to signature.instructions + prefix  (CoPro).
            """
            tmpl = self.predict.config.get("template")

            # --- case 1: teleprompter rewrote template ----------------
            if tmpl:
                if isinstance(tmpl, dict):                         # {"system":..,"user":..}
                    sys_msg = tmpl.get("system", "")
                    user_msg = tmpl.get("user", "")
                    if callable(sys_msg):
                        sys_msg = sys_msg(**kv)
                    if callable(user_msg):
                        user_msg = user_msg(**kv)
                    return f"<SYS>\n{sys_msg}\n</SYS>\n<USER>\n{user_msg}"

                if isinstance(tmpl, (list, tuple)):                # list[chat messages]
                    return "\n".join(str(m) for m in tmpl)

                if callable(tmpl):                                 # function
                    return tmpl(**kv)

                return str(tmpl)                                   # plain string

            # --- case 2: CoPro path ----------------------------------
            sig = self.predict.signature
            adapter = dspy.ChatAdapter()

            # build the standard message list the ChatAdapter would send
            msgs = adapter.format(signature=sig,
                                  demos=self.predict.demos,
                                  inputs=kv)

            # turn list-of-dicts into plain text for your LLM
            return msgs



        # ------------ initialise BaseLLMOperation ---------------------
        BaseLLMOperation.__init__(
            self,
            llm=llm,
            prompter=_render_prompt,  # always current prompt
            parser=parser,
            cache_seed=cache_seed,
            use_cache=use_cache,
            params=params,
            input_types=input_types,
            output_types=output_types,
            name=name,
        )

        # ------------ initialise dspy.Module --------------------------
        dspy.Module.__init__(self)

    # ------------------------------------------------------------------
    # Freeze mirror copies so DSPy optimiser skips them
    # ------------------------------------------------------------------
    def parameters(self, recurse: bool = True):
        if not self._is_master:
            return iter(())        # empty iterator → no trainable params
        return dspy.Module.parameters(self, recurse)
    
    def __deepcopy__(self, memo):
        """
        • When the *first* node of a given `group_id` is copied
          ➜ make a fresh Predict, register it under a new registry id.
        • Later nodes of the *same* group reuse that fresh Predict.
        """
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new

        # shallow-copy everything except predict
        for k, v in self.__dict__.items():
            if k != "predict":
                setattr(new, k, v)

        # key used to share the new Predict among mirrors in this deepcopy
        share_key = ("_copied_predict", self._group_id)

        if share_key in memo:
            # --- mirror copy ----------------------------------------
            new.predict     = memo[share_key]          # reuse Predict
            new._is_master  = False
            new._group_id   = memo[("_copied_gid", self._group_id)]
        else:
            # --- first copy (local master) --------------------------
            new.predict     = copy.deepcopy(self.predict, memo)
            new._is_master  = True

            # generate a unique registry id just for **this** deepcopy
            new_gid = f"{self._group_id}__copy{uuid.uuid4().hex[:6]}"
            new._group_id = new_gid
            cls._registry[new_gid] = new.predict

            # remember for subsequent mirrors
            memo[share_key] = new.predict
            memo[("_copied_gid", self._group_id)] = new_gid

        return new