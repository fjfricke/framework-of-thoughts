# Code adapted from the ProbTree repository (https://github.com/THU-KEG/ProbTree)

import ast
import json
import logging
import re
import string
from typing import Counter

def find_dependencies(subquestion: str) -> list[int]:
    """
    Finds the dependencies of a subquestion.
    """
    return [int(match.group(1)) for part in subquestion.split() for match in re.finditer(r"#(\d+)", part)]

def replace_dependencies(subquestion: str, dependencies: dict[int, str]) -> str:
    """
    Replaces the dependencies in a subquestion with the given dependencies by their id.
    """
    for id, dependency in dependencies.items():
        subquestion = subquestion.replace(f"#{id}", dependency)
    return subquestion
        
def calculate_average_logprob(logprobs, start, end):
    """
    Calculates the average log probability for a range of tokens.

    Args:
        tokens (list): List of tokens.
        logprobs (list): List of log probabilities.
        start (int): Start index of the range.
        end (int): End index of the range.

    Returns:
        float: The average log probability for the range.
    """
    return sum(logprobs[start:end + 1]) / len(logprobs[start:end + 1])

# relevant for probtree

def parse_tree_and_extract_logprobs(tokens, logprobs):
    """
    Parses the LLM response to extract sub-question data and their associated log probabilities.

    Args:
        llm_response (ChatCompletion): The response object from the LLM, containing choices, tokens, and log probabilities.

    Returns:
        dict: A dictionary where each sub-question is mapped to a tuple of (parsed data, average log probability).
    """
    def parse_response_content(response_content: str):
        """
        Parses the JSON content from the LLM response.

        Args:
            response_content (str): The raw content of the LLM response.

        Returns:
            dict: Parsed JSON object or None if parsing fails.
        """
        try:
            # Extract the portion of the string starting at the first '{' and ending at the last '}'
            start_index = response_content.find('{')
            end_index = response_content.rfind('}')
            if start_index != -1 and end_index != -1:
                response_content = response_content[start_index:end_index + 1]
            else:
                logging.error("Failed to find JSON boundaries in the response content.")
                return None
            # Parse the JSON
            return json.loads(response_content)
        except json.JSONDecodeError:
            logging.error("Failed to parse JSON from LLM response content.")
            return None
        
    # Step 1: Parse the response content
    parsed_data = parse_response_content(''.join(tokens))
    if not parsed_data:
        return None
    
    # Step 2: Process each sub-question and calculate log probabilities
    sub_question_data = {}
    token_index = 0

    for sub_question, question_data in parsed_data.items():
        # Find the start and end indices for the sub-question tokens
        start_index, end_index = None, None

        while token_index < len(tokens):
            if "[" in tokens[token_index]:
                start_index = token_index
                break
            token_index += 1

        while token_index < len(tokens):
            if "]" in tokens[token_index]:
                end_index = token_index
                break
            token_index += 1

        if start_index is None or end_index is None:
            logging.error(f"Failed to find token range for sub-question: {sub_question}")
            continue

        # Calculate the average log probability for the sub-question
        avg_logprob = calculate_average_logprob(logprobs, start_index, end_index)

        # Handle cases where the sub-question data is invalid
        if any(sub_question == item for item in question_data):
            question_data, avg_logprob = [], None

        # Store the result
        sub_question_data[sub_question] = (question_data, avg_logprob)

    return sub_question_data

# relevant for dynamic probtree

def parse_list_and_extract_logprobs(tokens, logprobs):
    def parse_response_content(response_content: str):
        try:
            # Extract the portion of the string starting at the first '{' and ending at the last '}'
            start_index = response_content.find('[')
            end_index = response_content.rfind(']')
            if start_index != -1 and end_index != -1:
                response_content = response_content[start_index:end_index + 1]
            else:
                logging.error("Failed to find list boundaries in the response content.")
                return None
            # Parse the JSON
            parsed_content_list = ast.literal_eval(response_content)
            if not (isinstance(parsed_content_list, list) and all(isinstance(item, str) for item in parsed_content_list)):
                logging.error("Failed to parse list from LLM response content.")
                return None
            return parsed_content_list
        except (ValueError, SyntaxError) as e:
            logging.error(f"Failed to parse list from LLM response content: {e}")
            return None
        
    parsed_data = parse_response_content(''.join(tokens))
    if not parsed_data:
        return None
    token_index = 0
        
    while token_index < len(tokens):
        if "[" in tokens[token_index]:
            start_index = token_index
            break
        token_index += 1

    while token_index < len(tokens):
        if "]" in tokens[token_index]:
            end_index = token_index
            break
        token_index += 1

    if start_index is None or end_index is None:
        logging.error(f"Failed to find token range for list: {parsed_data}")
        return None, None

    # Calculate the average log probability for the sub-question
    avg_logprob = calculate_average_logprob(logprobs, start_index, end_index)
    return parsed_data, avg_logprob

def parse_branch_and_extract_logprob(tokens, logprobs):
    parsed_data = ''.join(tokens)
    token_index = 0
    answer = None
    while token_index < len(tokens):
        if "Yes" in tokens[token_index]:
            answer = True
            index = token_index
            break
        elif "No" in tokens[token_index]:
            answer = False
            index = token_index
            break
        token_index += 1
    if index is None:
        logging.error(f"Failed to find token range for branch: {parsed_data}")
        return None, None
    logprob = logprobs[index]
    return answer, logprob

# relevant for metrics

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def calculate_f1_score(prediction, ground_truth) -> tuple[float, float, float]:

    zero_metric = (0, 0, 0)
    
    if prediction is None:
        return zero_metric
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return zero_metric
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return zero_metric

    prediction_tokens: list[str] = normalized_prediction.split()
    ground_truth_tokens: list[str] = normalized_ground_truth.split()
    common = Counter[str](prediction_tokens) & Counter[str](ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return zero_metric
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def calculate_exact_match_score(prediction, ground_truth) -> bool:

    if prediction is None:
        return False
    return normalize_answer(prediction) == normalize_answer(ground_truth)