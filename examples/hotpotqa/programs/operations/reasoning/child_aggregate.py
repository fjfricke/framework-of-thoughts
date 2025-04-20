from types import NoneType
from examples.hotpotqa.programs.utils import find_dependencies, replace_dependencies

from llm_graph_optimizer.graph_of_operations.graph_partitions import GraphPartitions
from llm_graph_optimizer.graph_of_operations.types import ManyToOne, ReasoningState
from llm_graph_optimizer.language_models.abstract_language_model import AbstractLanguageModel
from llm_graph_optimizer.language_models.openai_chat import OpenAIChat
from llm_graph_optimizer.measurement.measurement import Measurement
from llm_graph_optimizer.operations.helpers.exceptions import OperationFailed
from llm_graph_optimizer.operations.llm_operations.llm_operation_with_logprobs import LLMOperationWithLogprobs


def prompter(question: str, question_decomposition_score: float, dependency_answers: list[str], subquestions: list[str], subquestion_answers: list[str], child_decomposition_scores: list[float]) -> str:
    exemplar = """
Given a qeustion and a context, answer the question and explain why. End with \"So the answer is: <answer>.\"

#
Context:
Which famous fashion show Stella Maxwell has been a model for? Victoria's Secret.
Since when Victoria's Secret? 1977.

Question:
Stella Maxwell has been a model for a famous fashion shown since when?

Answer:
Stella Maxwell has been a model for a famous fashion shown, Victoria's Secret since 2015. So the answer is: since 2015.
#
Context:
Who is the American retired professional basketball player who is current president of basketball operations for the Los Angeles Lakers? Devean George.
William Novac co-wrote the memoir of Devean George? no.

Question:
William Novac co-wrote the memoir of what American retired professional basketball player who is current president of basketball operations for the Los Angeles Lakers?

Answer:
William Novac co-wrote the memoir of Magic Johnson, an American retired professional basketball player who is current president of basketball operations for the Los Angeles Lakers. So the answer is: Magic Johnson.
#
Context:
Which athlete rode 400 miles across his country to bring attention to the plight of the disabled in the country? Emmanuel Ofosu Yeboah.
What is the title of the documentary narrated by Oprah Winfrey about Emmanuel Ofosu Yeboah? Emmanuel's Gift.

Question:
Oprah Winfrey narrated a documentary about this athlete who rode 400 miles across his country to bring attention to the plight of the disabled in the country?

Answer:
Oprah Winfrey narrated a documentary about the athelete Emmanuel Ofosu Yeboah, who rode 400 miles across his country to bring attention to the plight of the disabled in the country. So the answer is: Emmanuel Ofosu Yeboah.
#
"""
    dependencies_ids: list[int] = find_dependencies(question)
    replaced_question = replace_dependencies(question, {id: answer for id, answer in zip(dependencies_ids, dependency_answers)})
    full_prompt = exemplar + f"""
Context:
{''.join(subquestion + ' ' + answer + '\n' for subquestion, answer in zip(subquestions, subquestion_answers))}
Question:
{replaced_question}

Answer:
"""
    return full_prompt


def parser(data: list[tuple[str, float]], question_decomposition_score: float, child_decomposition_scores: list[float]) -> dict[str, any]:
    # Extract tokens and logprobs
    tokens = [token for token, _ in data]
    logprobs = [logprob for _, logprob in data]
    
    # Calculate the decomposition score (average of logprobs)
    output_decomposition_score = sum(logprobs) / len(logprobs)
    total_decomposition_score = (question_decomposition_score + output_decomposition_score + sum(child_decomposition_scores)) / (len(child_decomposition_scores) + 2)
    
    # Concatenate tokens into a single string
    full_text = "".join(tokens)
    
    # Extract the answer from the concatenated string and strip the last full stop if it exists
    answer = full_text.split(":")[-1].strip().rstrip(".")

    return {
        "answer": answer,
        "decomposition_score": total_decomposition_score
    }

class ChildAggregateReasoning(LLMOperationWithLogprobs):
    def __init__(self, llm: AbstractLanguageModel, use_cache: bool = True, params: dict = None, name: str = "ChildAggregateReasoning"):
        input_types = {"question": str, "question_decomposition_score": float, "dependency_answers": list[str] | NoneType, "subquestions": ManyToOne[str], "subquestion_answers": ManyToOne[str], "child_decomposition_scores": ManyToOne[float]}
        output_types = {"answer": str, "decomposition_score": float}
        super().__init__(llm, prompter, parser, use_cache, params, input_types, output_types, name)

    async def _execute(self, partitions: GraphPartitions, input_reasoning_states: ReasoningState) -> tuple[ReasoningState, Measurement]:
        try:
            # Unpack input_reasoning_states into named arguments for the prompter
            prompt = self.prompter(**input_reasoning_states)
            
            # Query the language model
            response, measurement = await self.llm.query(prompt=prompt, use_cache=self.use_cache)
            
            # Pass the response to the parser
            return self.parser(response, input_reasoning_states["question_decomposition_score"], input_reasoning_states["child_decomposition_scores"]), measurement
        except Exception as e:
            print(e)
            raise OperationFailed(e)


if __name__ == "__main__":
    import asyncio
    from llm_graph_optimizer.operations.llm_operations.llm_operation_with_logprobs import LLMOperationWithLogprobs

    llm = OpenAIChat(model="gpt-4o-mini")
    operation = ChildAggregateReasoning(llm)
    answer = asyncio.run(operation._execute(None, {"question": "What is the capital of #1?", "question_decomposition_score": -0.01, "dependency_answers": ["Germany"], "subquestions": ["What is the biggest city in Germany?"], "subquestion_answers": ["Berlin"], "child_decomposition_scores": [-0.2]}))
    print(answer)