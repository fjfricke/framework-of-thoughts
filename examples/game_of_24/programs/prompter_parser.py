from llm_graph_optimizer.graph_of_operations.types import ReasoningState


def propose_prompt(input: list[int]) -> str:
    return f"""Input: 2 8 8 14
Possible next steps:
2 + 8 = 10 (left: 8 10 14)
8 / 2 = 4 (left: 4 8 14)
14 + 2 = 16 (left: 8 8 16)
2 * 8 = 16 (left: 8 14 16)
8 - 2 = 6 (left: 6 8 14)
14 - 8 = 6 (left: 2 6 8)
14 /  2 = 7 (left: 7 8 8)
14 - 2 = 12 (left: 8 8 12)
Input: {' '.join(map(str, input))}
Possible next steps:
"""

def propose_parser(output: str) -> ReasoningState:
    expressions = []
    remainings = []
    
    for line in output.split('\n'):
        if "(left:" in line:
            # Split the line into the expression and the remaining integers
            expression, remaining = line.split("(left:")
            
            # Extract the integers from the remaining part
            try:
                remainings.append([int(x) for x in remaining.strip(")").split()])
                expressions.append(expression[:-1])
            except ValueError:
                continue
    
    return {"expressions": expressions, "remainings": remainings}

def value_prompt(input: list[int]) -> str:
    return f"""Evaluate if given numbers can reach 24 (sure/likely/impossible)
10 14
10 + 14 = 24
sure
11 12
11 + 12 = 23
12 - 11 = 1
11 * 12 = 132
11 / 12 = 0.91
impossible
4 4 10
4 + 4 + 10 = 8 + 10 = 18
4 * 10 - 4 = 40 - 4 = 36
(10 - 4) * 4 = 6 * 4 = 24
sure
4 9 11
9 + 11 + 4 = 20 + 4 = 24
sure
5 7 8
5 + 7 + 8 = 12 + 8 = 20
(8 - 5) * 7 = 3 * 7 = 21
I cannot obtain 24 now, but numbers are within a reasonable range
likely
5 6 6
5 + 6 + 6 = 17
(6 - 5) * 6 = 1 * 6 = 6
I cannot obtain 24 now, but numbers are within a reasonable range
likely
10 10 11
10 + 10 + 11 = 31
(11 - 10) * 10 = 10
10 10 10 are all too big
impossible
1 3 3
1 * 3 * 3 = 9
(1 + 3) * 3 = 12
1 3 3 are all too small
impossible
Input: {' '.join(map(str, input))}
"""

def value_parser(output: str) -> ReasoningState:
    value_map = {'impossible': 0.001, 'likely': 1, 'sure': 20}
    
    # Iterate through the lines in reverse to find the last occurrence of a value name
    for line in reversed(output.split('\n')):
        for value_name, value in value_map.items():
            if value_name in line:
                return {"value": value}
    
    # If no value name is found, return None
    return {"value": None}

def value_last_step_prompt(input: list[int], answer: str) -> str:
    return f"""Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Given an input and an answer, give a judgement (sure/impossible) if the answer is correct, i.e. it uses each input exactly once and no other numbers, and reach 24.
Input: 4 4 6 8
Answer: (4 + 8) * (6 - 4) = 24
Judge: 
sure
Input: 2 9 10 12
Answer: 2 * 12 * (10 - 9) = 24
Judge: 
sure
Input: 4 9 10 13
Answer: (13 - 9) * (10 - 4) = 24
Judge: 
sure
Input: 4 4 6 8
Answer: (4 + 8) * (6 - 4) + 1 = 25
Judge: 
impossible
Input: 2 9 10 12
Answer: 2 * (12 - 10) = 24
Judge: 
impossible
Input: 4 9 10 13
Answer: (13 - 4) * (10 - 9) = 24
Judge: 
impossible
Input: {' '.join(map(str, input))}
Answer: {answer}
Judge:
"""

def value_last_step_parser(output: str) -> ReasoningState:
    if "answer" not in output.lower():
        return {"value": 0}
    else:
        return value_parser(output)

def cot_prompt(input: list[int], expressions: list[str], remainings: list[list[int]]) -> str:
    return f"""Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Each step, you are only allowed to choose two of the remaining numbers to obtain a new number.
Input: 4 4 6 8
Steps:
4 + 8 = 12 (left: 4 6 12)
6 - 4 = 2 (left: 2 12)
2 * 12 = 24 (left: 24)
Answer: (6 - 4) * (4 + 8) = 24
Input: 2 9 10 12
Steps:
12 * 2 = 24 (left: 9 10 24)
10 - 9 = 1 (left: 1 24)
24 * 1 = 24 (left: 24)
Answer: (12 * 2) * (10 - 9) = 24
Input: 4 9 10 13
Steps:
13 - 10 = 3 (left: 3 4 9)
9 - 3 = 6 (left: 4 6)
4 * 6 = 24 (left: 24)
Answer: 4 * (9 - (13 - 10)) = 24
Input: 1 4 8 8
Steps:
8 / 4 = 2 (left: 1 2 8)
1 + 2 = 3 (left: 3 8)
3 * 8 = 24 (left: 24)
Answer: (1 + 8 / 4) * 8 = 24
Input: 5 5 5 9
Steps:
5 + 5 = 10 (left: 5 9 10)
10 + 5 = 15 (left: 9 15)
15 + 9 = 24 (left: 24)
Answer: ((5 + 5) + 5) + 9 = 24
Input: {' '.join(map(str, input))}
Steps:
{'\n'.join([f"{e} (left: {', '.join(map(str, r))})" for e, r in zip(expressions, remainings)])}"""

def cot_parser(output: str) -> ReasoningState:
    ### returns the answer. do not delete whitespaces in the middle:
    return {"answer": output.split("Answer: ")[1]}
