from examples.hotpotqa.programs.operations.utils import find_dependencies, parse_branch_and_extract_logprob, replace_dependencies
from llm_graph_optimizer.graph_of_operations.types import ReasoningState
from llm_graph_optimizer.operations.llm_operations.llm_operation_with_logprobs import LLMOperationWithLogprobs


def prompter(question: str, dependency_answers: list[str]) -> str:
    dependencies_ids: list[int] = find_dependencies(question)
    replaced_question = replace_dependencies(question, {id: answer for id, answer in zip(dependencies_ids, dependency_answers)})
    return f"""You have to decide if the question \"{replaced_question}\" should be further decomposed into sub-questions. If you think it falls in the leaf nodes of the examples below, answer with \"No\". Otherwise, answer with \"Yes\".
Only answer with \"Yes\" (meaning should be decomposed) or \"No\" (meaning should not be decomposed).

Example decomposition trees:
Q: How many square miles is the source of the most legal immigrants to the location of Gotham's filming from the region where Andy from The Office sailed to?
A: {{"How many square miles is the source of the most legal immigrants to the location of Gotham's filming from the region where Andy from The Office sailed to?": ["What is the source of the most legal immigrants to the location of Gotham's filming from the region where Andy from The Office sailed to?", "How many square miles is #1?"], "What is the source of the most legal immigrants to the location of Gotham's filming from the region where Andy from The Office sailed to?": ["where is the tv show gotham filmed at", "where did andy sail to in the office", "What nation provided the most legal immigrants to #1 in the #2 ?"]}}.
Q: When did the capitol of Virginia move from Robert Banks' birthplace to the city sharing a border with Laurel's county?
A: {{"When did the capitol of Virginia move from Robert Banks' birthplace to the city sharing a border with Laurel's county?": ["What is the city sharing a border with Laurel's county?", "Where is Robert Banks' birthplace?", "When did the capitol of Virginia move from #1 to #2?"], "What is the city sharing a border with Laurel's county?": ["What county is Laurel located in?", "What city shares a border with #1?"]}}.
Q: An actor in Nowhere to Run is a national of a European country. That country's King Albert I lived during a major war that Italy joined in what year?
A: {{"An actor in Nowhere to Run is a national of a European country. That country's King Albert I lived during a major war that Italy joined in what year?": ["Albert I of the country which has the actor in Nowhere to Run lived during which war?", "When did Italy join #1?"], "Albert I of the country which has the actor in Nowhere to Run lived during which war?": ["Tell me the country which has the actor in Nowhere to Run", "Albert I of #1 lived during which war?"], "Tell me the country which has the actor in Nowhere to Run": ["Nowhere to Run's cast member is whom?", "What is the country of #1?"]}}.

Question: {replaced_question}
Answer (Yes or No):"""


def parser(output: list) -> ReasoningState:
    answer, logprob = parse_branch_and_extract_logprob(list(map(lambda x: x[0], output)), list(map(lambda x: x[1], output)))
    return {"should_decompose": answer, "decomposition_score": logprob}

class BranchOperation(LLMOperationWithLogprobs):
    pass