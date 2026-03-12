import asyncio
import json
import os
import random
import re
import subprocess
from pathlib import Path
from dataclasses import dataclass

import chatan


def setup_repo() -> Path:
    repo = Path("./vscode")
    if not repo.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/microsoft/vscode.git"],
            check=True,
        )
    return repo


@dataclass
class CodePattern:
    """A code pattern found via grep with its context."""
    pattern_type: str
    name: str
    file_path: str
    context: str


def grep_patterns(repo: Path, pattern: str, max_results: int = 100) -> list[tuple[str, str]]:
    """Run grep and return list of (file_path, matching_line) tuples."""
    try:
        result = subprocess.run(
            ["grep", "-r", "-n", "--include=*.ts", "-E", pattern, str(repo)],
            capture_output=True,
            text=True,
            timeout=30
        )
        matches = []
        for line in result.stdout.strip().split('\n')[:max_results]:
            if ':' in line:
                parts = line.split(':', 2)
                if len(parts) >= 2:
                    file_path = parts[0]
                    if 'node_modules' not in file_path and 'test' not in file_path.lower():
                        matches.append((file_path, line))
        return matches
    except Exception:
        return []


def get_file_context(file_path: str, line_num: int, context_lines: int = 30) -> str:
    """Get context around a specific line in a file."""
    try:
        with open(file_path, 'r', errors='ignore') as f:
            lines = f.readlines()
        start = max(0, line_num - context_lines // 2)
        end = min(len(lines), line_num + context_lines // 2)
        return ''.join(lines[start:end])
    except Exception:
        return ""


def find_all_patterns(repo: Path) -> list[CodePattern]:
    """Find all code patterns (classes, interfaces, functions)."""
    patterns = []

    # Classes
    for file_path, line in grep_patterns(repo, r"^export class [A-Z][a-zA-Z0-9]+ "):
        match = re.search(r"export class ([A-Z][a-zA-Z0-9]+)", line)
        if match:
            name = match.group(1)
            parts = line.split(':')
            line_num = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
            context = get_file_context(file_path, line_num)
            if context and len(context) > 200:
                patterns.append(CodePattern('class', name, file_path, context))

    # Interfaces
    for file_path, line in grep_patterns(repo, r"^export interface I[A-Z][a-zA-Z0-9]+ "):
        match = re.search(r"export interface (I[A-Z][a-zA-Z0-9]+)", line)
        if match:
            name = match.group(1)
            parts = line.split(':')
            line_num = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
            context = get_file_context(file_path, line_num)
            if context and len(context) > 200:
                patterns.append(CodePattern('interface', name, file_path, context))

    # Functions
    for file_path, line in grep_patterns(repo, r"^export function [a-z][a-zA-Z0-9]+\("):
        match = re.search(r"export function ([a-z][a-zA-Z0-9]+)", line)
        if match:
            name = match.group(1)
            parts = line.split(':')
            line_num = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
            context = get_file_context(file_path, line_num)
            if context and len(context) > 200:
                patterns.append(CodePattern('function', name, file_path, context))

    return patterns


def find_related_file(repo: Path, pattern: CodePattern) -> tuple[str, str] | None:
    """Find a file that imports/uses the given pattern."""
    import_pattern = f"import.*{pattern.name}"
    matches = grep_patterns(repo, import_pattern, max_results=50)

    for file_path, line in matches:
        if file_path == pattern.file_path:
            continue
        if 'test' in file_path.lower():
            continue

        parts = line.split(':')
        line_num = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
        context = get_file_context(file_path, line_num, context_lines=50)

        if pattern.name in context and len(context) > 300:
            return file_path, context

    return None


# Global cache built once at module load
_all_patterns: list[CodePattern] = []
_patterns_loaded = False


def _ensure_patterns_loaded():
    global _all_patterns, _patterns_loaded
    if not _patterns_loaded:
        repo = setup_repo()
        print("Loading patterns from VS Code repo...")
        _all_patterns = find_all_patterns(repo)
        random.shuffle(_all_patterns)
        print(f"Loaded {len(_all_patterns)} patterns")
        _patterns_loaded = True


def get_example_data(row: dict) -> str:
    """
    Get a complete example as a JSON string.
    Returns JSON with all fields needed for the example.
    Chatan passes row dict to all functions.
    """
    _ensure_patterns_loaded()

    if not _all_patterns:
        raise RuntimeError("No patterns available")

    # Pick a random pattern
    pattern = random.choice(_all_patterns)
    repo = setup_repo()

    # Try to find a related file
    related = find_related_file(repo, pattern)

    if related:
        related_path, related_context = related
        context_instruction = "You are given code from TWO related files. Explain how they work together."
        file_context_2_section = f"\n\nRelated code (showing where {pattern.name} is used):\n{related_context}"
    else:
        related_path = ""
        context_instruction = "You are given code from a single file."
        file_context_2_section = ""

    data = {
        "file_path": pattern.file_path,
        "file_path_2": related_path,
        "pattern_type": pattern.pattern_type,
        "pattern_name": pattern.name,
        "file_context": pattern.context,
        "context_instruction": context_instruction,
        "file_context_2_section": file_context_2_section,
    }

    return json.dumps(data)


# Chatan passes the row dict to all functions
# Extract functions get example_data from row["example_data"]

def extract_file_path(row: dict) -> str:
    """Extract file_path from example_data JSON."""
    return json.loads(row["example_data"])["file_path"]


def extract_file_path_2(row: dict) -> str:
    """Extract file_path_2 from example_data JSON."""
    return json.loads(row["example_data"])["file_path_2"]


def extract_pattern_type(row: dict) -> str:
    """Extract pattern_type from example_data JSON."""
    return json.loads(row["example_data"])["pattern_type"]


def extract_pattern_name(row: dict) -> str:
    """Extract pattern_name from example_data JSON."""
    return json.loads(row["example_data"])["pattern_name"]


def extract_file_context(row: dict) -> str:
    """Extract file_context from example_data JSON."""
    return json.loads(row["example_data"])["file_context"]


def extract_context_instruction(row: dict) -> str:
    """Extract context_instruction from example_data JSON."""
    return json.loads(row["example_data"])["context_instruction"]


def extract_file_context_2_section(row: dict) -> str:
    """Extract file_context_2_section from example_data JSON."""
    return json.loads(row["example_data"])["file_context_2_section"]


GROUND_TRUTH_PROMPT = """You are analyzing TypeScript code from the VS Code codebase.

Based on the code context below, write a factual technical answer about the {pattern_type} named `{pattern_name}`.

{context_instruction}

Code context:
{file_context}
{file_context_2_section}

Write a clear, technical answer that:
1. Explains what `{pattern_name}` does or represents
2. Describes key methods, properties, or behaviors visible in the code
3. If there are two code sections, explain how they relate to each other
4. DO NOT mention "File 1", "File 2", or any file names/paths
5. Write as if explaining to a developer who will search the codebase to verify

Provide only the factual answer, no preamble.
"""


USER_QUERY_PROMPT = """You are role-playing as a developer working on VS Code who has a question.

Based on this technical answer about the codebase:
{ground_truth}

Write a natural question that a developer might ask, where this answer would be the correct response.

Rules:
1. Do NOT mention specific file names or paths
2. Ask about concepts, functionality, or implementation details
3. The question should be answerable by searching the codebase with grep
4. Write only the question, nothing else

Example good questions:
- "How does the editor handle undo/redo operations?"
- "What service is responsible for managing workspace configurations?"
- "How are keyboard shortcuts registered and processed?"
"""


CHECK_GROUND_TRUTH_PROMPT = """Evaluate if this question-answer pair is suitable for training a code search agent.

Question: {user_query}

Answer: {ground_truth}

Criteria:
1. The question should be specific enough to search for in a codebase
2. The answer should contain factual, verifiable information about code
3. The answer should NOT reference "File 1", "File 2", or mention being "defined in" a specific file
4. The question and answer should logically match

Return "Yes" if the pair meets all criteria.
Return "No" if any criterion is not met.
"""


async def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    gen = chatan.generator("openai", api_key)

    # Pre-load patterns
    _ensure_patterns_loaded()

    ds = chatan.dataset(
        {
            # First generate the example data (no dependencies)
            "example_data": get_example_data,
            # Extract individual fields - chatan matches param name 'example_data' to column
            "file_path": extract_file_path,
            "file_path_2": extract_file_path_2,
            "pattern_type": extract_pattern_type,
            "pattern_name": extract_pattern_name,
            "file_context": extract_file_context,
            "context_instruction": extract_context_instruction,
            "file_context_2_section": extract_file_context_2_section,
            # Generate LLM outputs - uses template variables matching column names
            "ground_truth": gen(GROUND_TRUTH_PROMPT),
            "user_query": gen(USER_QUERY_PROMPT),
            "check": gen(CHECK_GROUND_TRUTH_PROMPT),
        }
    )

    await ds.generate(n=1000, max_concurrent_rows=50)

    hf_ds = ds.to_huggingface()

    # Remove intermediate columns before upload
    cols_to_remove = ["example_data", "file_context", "pattern_type", "pattern_name",
                      "context_instruction", "file_context_2_section"]
    for col in cols_to_remove:
        if col in hf_ds.column_names:
            hf_ds = hf_ds.remove_columns([col])

    hf_ds.push_to_hub("cdreetz/swe-grep-v2", token=os.getenv("HF_TOKEN"))
    return hf_ds


if __name__ == "__main__":
    d = asyncio.run(main())
