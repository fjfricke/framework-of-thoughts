from examples.hotpotqa.programs.operations.utils import find_dependencies, parse_branch_and_extract_logprob, replace_dependencies
from llm_graph_optimizer.graph_of_operations.types import ReasoningState
from llm_graph_optimizer.operations.llm_operations.llm_operation_with_logprobs import LLMOperationWithLogprobs


def prompter(question: str, dependency_answers: list[str]) -> str:
    dependencies_ids: list[int] = find_dependencies(question)
    replaced_question = replace_dependencies(question, {id: answer for id, answer in zip(dependencies_ids, dependency_answers)})
    return f"""You have to decide whether the question \"{replaced_question}\" still needs to be broken down into smaller sub‑questions.
If the question can be answered directly, answer **\"No\"**.
If it should be decomposed into subquestions whcih first should be answered, answer **\"Yes\"**.

Only answer with **\"No\"** (leaf) or **\"Yes\"** (not a leaf).

──────────────────  S‑H‑U‑F‑F‑L‑E‑D  E‑X‑A‑M‑P‑L‑E‑S  ──────────────────
Q: What nation provided the most legal immigrants to New York City in the Bahamas?  
A: No

Q: When did the capitol of Virginia move from Richmond to Williamsburg?  
A: No

Q: An actor in *Nowhere to Run* is a national of a European country. That country's King Albert I lived during a major war that Italy joined in what year?  
A: Yes

Q: What is the source of the most legal immigrants to the location of Gotham's filming from the region where Andy from *The Office* sailed to?  
A: Yes

Q: Albert I of Belgium lived during which war?  
A: No

Q: where did andy sail to in *the office*  
A: No

Q: When did Italy join World War I?  
A: No

Q: How many square miles is Dominican Republic?  
A: No

Q: When did the capitol of Virginia move from Robert Banks' birthplace to the city sharing a border with Laurel's county?  
A: Yes

Q: How many square miles is the source of the most legal immigrants to the location of Gotham's filming from the region where Andy from *The Office* sailed to?  
A: Yes

Q: What county is Laurel located in?  
A: No

Q: What is the city sharing a border with Laurel's county?  
A: Yes

Q: Tell me the country which has the actor in *Nowhere to Run*  
A: Yes

Q: *Nowhere to Run*'s cast member is whom?  
A: No

Q: where is the tv show *gotham* filmed at  
A: No

Q: Albert I of the country which has the actor in *Nowhere to Run* lived during which war?  
A: Yes

Q: What city shares a border with Henrico County?  
A: No

Q: Where is Robert Banks' birthplace?  
A: No

Q: What is the country of Jean‑Claude Van Damme?  
A: No
────────────────────────────────────────────────────────────────────────

Should the question be broken down into smaller sub‑questions?
Question: {replaced_question}
Answer (Yes or No):"""


def parser(output: list) -> ReasoningState:
    answer, logprob = parse_branch_and_extract_logprob(list(map(lambda x: x[0], output)), list(map(lambda x: x[1], output)))
    return {"should_decompose": answer, "decomposition_score": logprob}

class ShouldBranchClassifier(LLMOperationWithLogprobs):
    pass