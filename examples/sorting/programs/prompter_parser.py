# Initialize the LLM generate operation and node
import ast
import logging
from typing import TypedDict
from collections import Counter

from llm_graph_optimizer.graph_of_operations.types import ReasoningState


def generate_prompt(input_list: list[int]):
    return f"""<Instruction> Sort the following list of numbers in ascending order. Output only the sorted list of numbers, no additional text. </Instruction>

<Examples>
Input: [5, 1, 0, 1, 2, 0, 4, 8, 1, 9, 5, 1, 3, 3, 9, 7]
Output: [0, 0, 1, 1, 1, 1, 2, 3, 3, 4, 5, 5, 7, 8, 9, 9]

Input: [3, 7, 0, 2, 8, 1, 2, 2, 2, 4, 7, 8, 5, 5, 3, 9, 4, 3, 5, 6, 6, 4, 4, 5, 2, 0, 9, 3, 3, 9, 2, 1]
Output: [0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 9]

Input: [4, 4, 9, 7, 9, 7, 0, 0, 4, 9, 1, 7, 9, 5, 8, 7, 5, 6, 3, 8, 6, 7, 5, 8, 5, 0, 6, 3, 7, 0, 5, 3, 7, 5, 2, 4, 4, 9, 0, 7, 8, 2, 7, 7, 7, 2, 1, 3, 9, 9, 7, 9, 6, 6, 4, 5, 4, 2, 0, 8, 9, 0, 2, 2]
Output: [0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9]
</Examples>

Input: {input_list}"""

def generate_prompt_cot(input_list: list[int]):
    return f"""<Instruction> Sort the following list of numbers in ascending order. You can generate any intermediate lists, but the final output should be the sorted list of numbers, prefixed with "Output: ". </Instruction>

<Approach>
To sort the list of numbers follow these steps:
1. Split the list of numbers into two to four unsorted sublists, each containing an equal number of elements from the original list (make sure they don't overlap).
2. Sort each of the unsorted sublists.
3. Merge the sorted sublists into a single sorted list using the merging algorithm from merge sort.
</Approach>

<Examples>
Input: [4, 5, 3, 3, 7, 3, 0, 5, 0, 2, 8, 0, 2, 1, 6, 9]
Unsorted Subarrays:
[4, 5, 3, 3, 7, 3, 0, 5]
[0, 2, 8, 0, 2, 1, 6, 9]
Sorted Subarrays:
[0, 3, 3, 3, 4, 5, 5, 7]
[0, 0, 1, 2, 2, 6, 8, 9]
Output: [0, 0, 0, 1, 2, 2, 3, 3, 3, 4, 5, 5, 6, 7, 8, 9]

Input: [6, 4, 5, 7, 5, 6, 9, 7, 6, 9, 4, 6, 9, 8, 1, 9, 2, 4, 9, 0, 7, 6, 5, 6, 6, 2, 8, 3, 9, 5, 6, 1]
Unsorted Subarrays:
[6, 4, 5, 7, 5, 6, 9, 7, 6, 9, 4, 6, 9, 8, 1, 9]
[2, 4, 9, 0, 7, 6, 5, 6, 6, 2, 8, 3, 9, 5, 6, 1]
Sorted Subarrays:
[1, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 8, 9, 9, 9, 9]
[0, 1, 2, 2, 3, 4, 5, 5, 6, 6, 6, 6, 7, 8, 9, 9]
Output: [0, 1, 1, 2, 2, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 8, 8, 9, 9, 9, 9, 9, 9]

Input: [3, 7, 0, 2, 8, 1, 2, 2, 2, 4, 7, 8, 5, 5, 3, 9, 4, 3, 5, 6, 6, 4, 4, 5, 2, 0, 9, 3, 3, 9, 2, 1, 9, 3, 1, 8, 1, 8, 6, 0, 1, 6, 1, 7, 4, 4, 6, 3, 3, 7, 9, 3, 6, 0, 3, 4, 5, 6, 6, 9, 9, 9, 7, 3]
Unsorted Subarrays:
[3, 7, 0, 2, 8, 1, 2, 2, 2, 4, 7, 8, 5, 5, 3, 9]
[4, 3, 5, 6, 6, 4, 4, 5, 2, 0, 9, 3, 3, 9, 2, 1]
[9, 3, 1, 8, 1, 8, 6, 0, 1, 6, 1, 7, 4, 4, 6, 3]
[3, 7, 9, 3, 6, 0, 3, 4, 5, 6, 6, 9, 9, 9, 7, 3]
Sorted Subarrays:
[0, 1, 2, 2, 2, 2, 3, 3, 4, 5, 5, 7, 7, 8, 8, 9]
[0, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 9, 9]
[0, 1, 1, 1, 1, 3, 3, 4, 4, 6, 6, 6, 7, 8, 8, 9]
[0, 3, 3, 3, 3, 4, 5, 6, 6, 6, 7, 7, 9, 9, 9, 9]
Output: [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9]
</Examples>

Input: {input_list}"""

def tot_improve_prompt(input_list: list[int], incorrectly_sorted: list[int]):
    return f"""<Instruction> The following two lists represent an unsorted list of numbers and a sorted variant of that list. The sorted variant is not correct. Fix the sorted variant so that it is correct.
Make sure that the output list is sorted in ascending order, has the same number of elements as the input list ({len(input_list)}), and contains the same elements as the input list. </Instruction>

<Approach>
To fix the incorrectly sorted list follow these steps:
1. For each number from 0 to 9, compare the frequency of that number in the incorrectly sorted list to the frequency of that number in the input list.
2. Iterate through the incorrectly sorted list and add or remove numbers as needed to make the frequency of each number in the incorrectly sorted list match the frequency of that number in the input list.
</Approach>

<Examples>
Input: [3, 7, 0, 2, 8, 1, 2, 2, 2, 4, 7, 8, 5, 5, 3, 9]
Incorrectly Sorted: [0, 0, 0, 0, 0, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 7, 7, 8, 8, 9, 9, 9, 9]
Reason: The incorrectly sorted list contains four extra 0s, two extra 4s and three extra 9s and is missing two 2s.
Output: [0, 1, 2, 2, 2, 2, 3, 3, 4, 5, 5, 7, 7, 8, 8, 9]

Input: [6, 4, 5, 7, 5, 6, 9, 7, 6, 9, 4, 6, 9, 8, 1, 9, 2, 4, 9, 0, 7, 6, 5, 6, 6, 2, 8, 3, 9, 5, 6, 1]
Incorrectly Sorted: [0, 1, 1, 2, 2, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 8, 8, 9, 9, 9, 9, 9]
Reason: The incorrectly sorted list contains two extra 4s and is missing two 6s and one 9.
Output: [0, 1, 1, 2, 2, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 8, 8, 9, 9, 9, 9, 9, 9]

Input: [4, 4, 9, 7, 9, 7, 0, 0, 4, 9, 1, 7, 9, 5, 8, 7, 5, 6, 3, 8, 6, 7, 5, 8, 5, 0, 6, 3, 7, 0, 5, 3, 7, 5, 2, 4, 4, 9, 0, 7, 8, 2, 7, 7, 7, 2, 1, 3, 9, 9, 7, 9, 6, 6, 4, 5, 4, 2, 0, 8, 9, 0, 2, 2]
Incorrectly Sorted: [0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9]
Reason: The incorrectly sorted list contains one extra 8 and is missing two 2s, one 3, three 4s, two 5s, one 6, six 7s and one 9.
Output: [0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9]
</Examples>

Input: {input_list}
Incorrectly Sorted: {incorrectly_sorted}
"""

class ParserOutput(TypedDict):
    output: list[int]

def generate_parser(text: str) -> ParserOutput:
    answers = text.strip().split("\n")
    answers = [
        answer for answer in answers if "[" in answer and "]" in answer
    ]
    if any(["Output" in answer for answer in answers]):
        # cut elements until last output is found
        for answer in reversed(answers):
            if "Output" in answer:
                answers = answers[answers.index(answer) :]
                break

    answers = [
        answer[answer.index("[") : answer.index("]") + 1]
        for answer in answers
    ]
    if len(answers) == 0:
        logging.warning(
            f"Could not parse step answer: {text}. Returning empty list."
        )
        answer = []
    else:
        if len(answers) > 1:
            logging.warning(
                f"Multiple answers found for step answer: {text}. Using the first one."
            )
        try:
            answer = ast.literal_eval(answers[0])
        except (ValueError, SyntaxError) as e:
            logging.error(f"Failed to parse answer: {answers[0]}. Error: {e}")
            answer = []
    return {"output": answer}

def scoring_function(output: list[int], expected_output: list[int]) -> int:
    """
    Function to calculate the number of errors in the sorting process.

    :param output: The current state of the sorted list.
    :param expected_output: The expected sorted list.
    :return: Number of errors.
    :rtype: float
    """
    try:
        # Calculate errors
        num_errors = 0

        # Count errors in the frequency of each number (0-9)
        for i in range(10):
            num_errors += abs(
                sum([1 for num in output if num == i])
                - sum([1 for num in expected_output if num == i])
            )

        # Count ordering errors (numbers out of order)
        num_errors += sum(
            [1 for num1, num2 in zip(output, output[1:]) if num1 > num2]
        )

        return num_errors
    except Exception as e:
        logging.error(f"Error in scoring function: {e}")
        return 300  # Return a high error score in case of failure
    
def filter_function(outputs: list[list[int]], scores: list[int]) -> ReasoningState:
    # Flatten the outputs and count occurrences
    flattened_outputs = [tuple(output) for output in outputs]
    most_common_output, _ = Counter(flattened_outputs).most_common(1)[0]

    # Find the corresponding score for the most common output
    index = outputs.index(list(most_common_output))
    most_common_score = scores[index]

    return {"output": list(most_common_output), "score": most_common_score}