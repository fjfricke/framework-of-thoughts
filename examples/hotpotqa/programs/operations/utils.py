import json
import logging
import re
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