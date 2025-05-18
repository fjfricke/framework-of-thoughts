from typing import TypedDict

# TypedDicts for structured outputs
class MergeOutput(TypedDict):
    merged: str

class ScoreOutput(TypedDict):
    redundancy: float
    retention: float
    f1_score: float

class ValidationOutput(TypedDict):
    valid: bool

# Helper functions
def strip_tags_helper(text: str, tag: str) -> str:
    """
    Helper function to remove specified tags from a text.

    :param text: The input text.
    :type text: str
    :param tag: The tag to be stripped.
    :type tag: str
    :return: The stripped text.
    :rtype: str
    """
    text = text.strip()
    start = text.find(f"<{tag}>")
    end = text.find(f"</{tag}>")
    if start != -1 and end != -1:
        return text[start + len(f"<{tag}>"):end].strip()
    else:
        start = text.find(f"[[ ## {tag.lower()} ## ]]")
        end = text.find("[[ ## completed ## ]]")
        if start != -1 and end != -1:
            return text[start + len(f"[[ ## {tag.lower()} ## ]]") : end].strip()
    return ""

# Prompt generation functions
def merge_prompt(docs: list[str]) -> str:
    return f"""
Merge the following {len(docs)} NDA documents <Doc1> - <Doc{len(docs)}> into a single NDA, maximizing retained information and minimizing redundancy. Output only the created NDA between the tags <Merged> and </Merged>, without any additional text.
Here are NDAs <Doc1> - <Doc{len(docs)}>

{'\n'.join([f'<Doc{i+1}>\n{doc}\n</Doc{i+1}>\n' for i, doc in enumerate(docs)])}
"""

def merge_parser(text: str) -> MergeOutput:
    result = strip_tags_helper(text, "Merged")
    return {"merged": result}

def score_prompt(summary: str, docs: list[str]) -> str:
    return f"""The following NDA <S> merges NDAs <Doc1> - <Doc{len(docs)}>.
Please score the merged NDA <S> in terms of how much redundant information is contained, independent of the original NDAs, as well as how much information is retained from the original NDAs.
A score of 10 for redundancy implies that absolutely no information is redundant, while a score of 0 implies that at least half of the information is redundant (so everything is at least mentioned twice).
A score of 10 for retained information implies that all information from the original NDAs is retained, while a score of 0 implies that no information is retained.
You may provide reasoning for your scoring, but the final score for redundancy should be between the tags <Redundancy> and </Redundancy>, and the final score for retained information should be between the tags <Retained> and </Retained>, without any additional text within any of those tags.

Here are NDAs <Doc1> - <Doc{len(docs)}>:

{'\n'.join([f'<Doc{i+1}>\n{doc}\n</Doc{i+1}>\n' for i, doc in enumerate(docs)])}

Here is the summary NDA <S>:
<S>
{summary}
</S>
"""

def score_parser(text: str) -> ScoreOutput:
    try:
        redundancy_score = float(strip_tags_helper(text, "Redundancy"))
        retention_score = float(strip_tags_helper(text, "Retained"))
        f1_score = (2 * redundancy_score * retention_score) / (redundancy_score + retention_score) if redundancy_score + retention_score > 0 else 0.0
        return {"redundancy": redundancy_score, "retention": retention_score, "f1_score": f1_score}
    except ValueError:
        return {"redundancy": 0.0, "retention": 0.0, "f1_score": 0.0}

def aggregate_prompt(summaries: list[str], docs: list[str]) -> str:
    num_ndas_summaries = len(summaries)
    return f"""
The following NDAs <S1> - <S{num_ndas_summaries}> each merge the initial NDAs <Doc1> - <Doc{len(docs)}>.
Combine the merged NDAs <S1> - <S{num_ndas_summaries}> into a new one, maximizing their advantages and overall information retention, while minimizing redundancy.
Output only the new NDA between the tags <Merged> and </Merged>, without any additional text.

Here are the original NDAs <Doc1> - <Doc{len(docs)}>:

{'\n'.join([f'<Doc{i+1}>\n{doc}\n</Doc{i+1}>\n' for i, doc in enumerate(docs)])}

Here are the summary NDAs <S1> - <S{num_ndas_summaries}>:

{'\n'.join([f'<S{i+1}>\n{summary}\n</S{i+1}>\n' for i, summary in enumerate(summaries)])}
"""

def improve_prompt(summaries: list[str], docs: list[str]) -> str:
    summary = summaries[0]
    return f"""
The following NDA <S> merges initial NDAs <Doc1> - <Doc{len(docs)}>.
Please improve the summary NDA <S> by adding more information and removing redundancy. Output only the improved NDA, placed between the two tags <Merged> and </Merged>, without any additional text.

Here are NDAs <Doc1> - <Doc{len(docs)}>:

{'\n'.join([f'<Doc{i+1}>\n{doc}\n</Doc{i+1}>\n' for i, doc in enumerate(docs)])}

Here is the summary NDA <S>:
<S>
{summary}
</S>
"""

def validation_parser(text: str) -> ValidationOutput:
    valid = text.lower().strip() in ["true", "yes", "valid"]
    return {"valid": valid}

def aggregate_parser(text: str) -> MergeOutput:
    result = strip_tags_helper(text, "Merged")
    return {"merged": result}

def improve_parser(text: str) -> MergeOutput:
    result = strip_tags_helper(text, "Merged")
    return {"merged": result}

if __name__ == "__main__":
    print(merge_prompt(["doc1", "doc2", "doc3", "doc4"]))
    print(score_prompt("s", ["doc1", "doc2", "doc3", "doc4"]))
    print(aggregate_prompt(["s1", "s2", "s3", "s4"], ["doc1", "doc2", "doc3", "doc4"]))
    print(improve_prompt("s", ["doc1", "doc2", "doc3", "doc4"]))