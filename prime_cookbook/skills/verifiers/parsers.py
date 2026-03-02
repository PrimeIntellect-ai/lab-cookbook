"""Output parsers for structured model responses.

Parsers extract the relevant answer part from model completions.
Used by reward functions and environment stop conditions.

Two patterns:
    XML tags:   <reasoning>...</reasoning><answer>Paris</answer>
    Last line:  (model puts final answer on its own line at the end)
"""
from typing import Optional


def make_xml_parser(tags: list) -> object:
    """Create a verifiers XMLParser for structured outputs.

    The model is expected to wrap its response in XML-style tags.
    Commonly used tags: ["reasoning", "answer"] or ["thinking", "answer"].

    Example:
        Model output:
            <reasoning>France is in Western Europe...</reasoning>
            <answer>Paris</answer>

        parser = make_xml_parser(["reasoning", "answer"])
        rubric = vf.Rubric(funcs=[xml_parser_reward], parser=parser)

        # parsed.answer == "Paris"
        # parsed.reasoning == "France is in Western Europe..."
    """
    import verifiers as vf
    return vf.XMLParser(tags)


async def xml_parser_reward(
    completion: str,
    answer: str,
    parser: object,
    **kwargs,
) -> float:
    """Exact match on the parsed 'answer' XML tag.

    Requires:
        - parser = vf.XMLParser([..., "answer"]) set on the rubric
        - Model outputs an <answer>...</answer> tag

    Returns 1.0 if the extracted answer matches reference (case-insensitive,
    stripped), 0.0 otherwise. Also returns 0.0 if no <answer> tag is found.

    Example:
        parser = make_xml_parser(["reasoning", "answer"])
        rubric = vf.Rubric(funcs=[xml_parser_reward], parser=parser)
    """
    if parser is None:
        return 0.0

    parsed = parser.parse(completion)
    extracted = getattr(parsed, "answer", None)

    if extracted is None:
        return 0.0

    return 1.0 if extracted.strip().lower() == answer.strip().lower() else 0.0


async def last_line_reward(
    completion: str,
    answer: str,
    **kwargs,
) -> float:
    """Exact match on the last non-empty line of completion.

    Useful when the model is instructed to put the final answer
    on the last line without any XML formatting.

    System prompt example:
        "Think through the problem, then put your final answer
        on the last line of your response with nothing else."

    Returns 1.0 if the last line matches reference (case-insensitive,
    stripped), 0.0 otherwise or if completion is empty.

    Example:
        rubric = vf.Rubric(funcs=[last_line_reward])
        # Dataset: {"answer": "Paris"}
        # Model output ends with: "...therefore the capital is Paris\nParis"
        # → extracts "Paris" → matches
    """
    if not completion:
        return 0.0

    lines = [line.strip() for line in completion.strip().split("\n") if line.strip()]
    if not lines:
        return 0.0

    return 1.0 if lines[-1].lower() == answer.strip().lower() else 0.0
