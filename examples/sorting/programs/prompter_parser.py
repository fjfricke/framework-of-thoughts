# The prompts are based on tge GoT paper: https://github.com/spcl/graph-of-thoughts
# and were adapted for this survey: https://github.com/simonmalberg/llm-graph-based-problem-solving

# Initialize the LLM generate operation and node
import ast
import logging
from typing import TypedDict

from typeguard import TypeCheckError, check_type

from llm_graph_optimizer.graph_of_operations.types import ReasoningState
from llm_graph_optimizer.operations.helpers.exceptions import OperationFailed


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

def got_split_prompt(input_list: list[int]):
    return f"""<Instruction> Split the following list of 128 numbers into 8 lists of 16 numbers each, the first list should contain the first 16 numbers, the second list the second 16 numbers, the third list the third 16 numbers, the fourth list the fourth 16 numbers, the fifth list the fifth 16 numbers and so on.
Only output the final 8 lists in the following format without any additional text or thoughts!:
{{
    "List 1": [3, 4, 3, 5, 7, 8, 1, ...],
    "List 2": [2, 9, 2, 4, 7, 1, 5, ...],
    "List 3": [6, 9, 8, 1, 9, 2, 4, ...],
    "List 4": [9, 0, 7, 6, 5, 6, 6, ...]
}} </Instruction>

<Example>
Input: [3, 1, 9, 3, 7, 5, 5, 4, 8, 1, 5, 3, 3, 2, 3, 0, 9, 7, 2, 2, 4, 4, 8, 5, 0, 8, 7, 3, 3, 8, 7, 0, 9, 5, 1, 6, 7, 6, 8, 9, 0, 3, 0, 6, 3, 4, 8, 0, 6, 9, 8, 4, 1, 2, 9, 0, 4, 8, 8, 9, 9, 8, 5, 9]
Output: 
{{
    "List 1": [3, 1, 9, 3, 7, 5, 5, 4, 8, 1, 5, 3, 3, 2, 3, 0],
    "List 2": [9, 7, 2, 2, 4, 4, 8, 5, 0, 8, 7, 3, 3, 8, 7, 0],
    "List 3": [9, 5, 1, 6, 7, 6, 8, 9, 0, 3, 0, 6, 3, 4, 8, 0],
    "List 4": [6, 9, 8, 4, 1, 2, 9, 0, 4, 8, 8, 9, 9, 8, 5, 9]
}}
</Example>

Input: {input_list}"""

class GotSplitOutput(TypedDict):
    output1: list[int]
    output2: list[int]
    output3: list[int]
    output4: list[int]

def got_split_parser(text: str) -> GotSplitOutput:
    try:
        start_index = text.index("{")
        end_index = text.rindex("}") + 1
        parsed_data = ast.literal_eval(text[start_index:end_index])
        lists = [values for key, values in parsed_data.items()]
        try:
            check_type(lists, list[list[int]])
            if len(lists) != 4:
                raise OperationFailed(f"Invalid number of lists: {len(lists)}")
            return {"output1": lists[0], "output2": lists[1], "output3": lists[2], "output4": lists[3]}
        except TypeCheckError:
            raise OperationFailed(f"Invalid output type: {type(lists)}")
    except (ValueError, SyntaxError) as e:
        raise OperationFailed(f"Error parsing text: {e}")
        

def got_aggregate_prompt(input1: list[int], input2: list[int]):
    len_input1 = len(input1)
    len_input2 = len(input2)
    if len_input1 == len_input2:
        length = len_input1
    elif len_input1 + len_input2 - 32 <= 16:
        length = 16
    elif len_input1 + len_input2 - 64 <= 32:
        length = 32
    else:
        length = 64
    length1 = length
    length2 = length * 2
    return f"""<Instruction> Merge the following 2 sorted lists of length {length1} each, into one sorted list of length {length2} using a merge sort style approach.
Only output the final merged list without any additional text or thoughts!:</Instruction>

<Approach>
To merge the two lists in a merge-sort style approach, follow these steps:
1. Compare the first element of both lists.
2. Append the smaller element to the merged list and move to the next element in the list from which the smaller element came.
3. Repeat steps 1 and 2 until one of the lists is empty.
4. Append the remaining elements of the non-empty list to the merged list.
</Approach>

Merge the following two lists into one sorted list:
1: {input1}
2: {input2}

Merged list:
"""

class ParserOutput(TypedDict):
    output: list[int]

def generate_parser(text: str) -> ParserOutput:
    answer = text.strip()
    if "Output" in answer:
        # cut elements until last output is found
        last_output_index = answer.rfind("Output")
        if last_output_index != -1:
            answer = answer[last_output_index + len("Output") :]
        else:
            logging.warning(
                f"Could not find 'Output' in the text: {text}. Returning empty list."
            )
            return {"output": []}

    if "[" in answer and "]" in answer:
        answer = answer[answer.rindex("[") : answer.rindex("]") + 1]
    else:
        logging.warning("Could not find '[' or ']' in the text. Returning empty list.")
        return {"output": []}
    try:
        answer = ast.literal_eval(answer)
    except (ValueError, SyntaxError) as e:
        logging.error(f"Failed to parse answer: {answer[0]}. Error: {e}")
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
    # Find the index of the smallest score
    min_score_index = scores.index(min(scores))

    # Get the output corresponding to the smallest score
    best_output = outputs[min_score_index]
    best_score = scores[min_score_index]

    return {"output": best_output, "score": best_score}