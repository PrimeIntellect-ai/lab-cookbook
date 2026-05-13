from __future__ import annotations

import os
import shlex
from collections.abc import Iterable, Mapping
from typing import Any

import verifiers.v1 as vf
from datasets import Dataset, load_dataset
from openai import AsyncOpenAI
from verifiers import Parser, ensure_keys

DATASET_NAME = "cdreetz/swe-grep-v2"

SYSTEM_PROMPT = """You are a helpful assistant that can answer questions and help with tasks.
Use the provided tools to search through the codebase to best answer user queries.
You will only have 2 turns to complete the task, but can make up to 8 tool calls per turn.
This means you MUST make multiple parallel tool calls to find information efficiently.

IMPORTANT: Questions may require finding information from MULTIPLE files. When providing your final answer, you MUST:
1. Include ALL file paths where you found relevant information
2. Provide the complete answer to the question

Format your response like this:
Files:
- <path/to/file1>
- <path/to/file2>
Answer: <your answer here>

With only 2 turns, you must parallelize your searches to find all relevant files.
"""

JUDGE_PROMPT = """Given a ground truth answer and a response, determine if the answer is correct.

Question:
{question}

Ground truth answer:
{answer}

Response:
{response}

Respond either 'yes' or 'no' only.
"""


class SweGrepTasksetConfig(vf.TasksetConfig):
    dataset_name: str = DATASET_NAME
    train_ratio: float = 0.9
    max_turns: int = 2
    system_prompt: str | None = SYSTEM_PROMPT
    judge_model: str = "gpt-4.1-mini"
    judge_base_url: str = "https://api.openai.com/v1"
    judge_api_key_var: str = "OPENAI_API_KEY"
    sandbox_image: str = "python:3.11-slim"
    sandbox_timeout_minutes: int = 60
    sandbox_command_timeout_seconds: int = 60
    sandbox_setup_timeout_seconds: int = 600


parser = Parser()


def convert_dataset(train_ratio: float, dataset_name: str) -> tuple[Dataset, Dataset]:
    dataset = load_dataset(dataset_name, split="train")
    dataset = dataset.filter(lambda row: row["check"] == "Yes")
    dataset = dataset.rename_columns({"user_query": "question", "ground_truth": "answer"})
    cols_to_remove = [name for name in ("file_chunk", "check") if name in dataset.column_names]
    if cols_to_remove:
        dataset = dataset.remove_columns(cols_to_remove)
    split = dataset.train_test_split(test_size=1 - train_ratio, seed=42)
    return split["train"], split["test"]


def rows(dataset: Iterable[Mapping[str, object]], max_turns: int):
    for index, row in enumerate(dataset):
        question = str(row["question"])
        yield {
            **dict(row),
            "example_id": index,
            "prompt": [{"role": "user", "content": question}],
            "max_turns": max_turns,
        }


async def run_sandbox_command(sandbox: Any, command: str) -> tuple[bool, str]:
    result = await sandbox.execute(command)
    stdout = str(getattr(result, "stdout", "") or "")
    stderr = str(getattr(result, "stderr", "") or "")
    if int(getattr(result, "exit_code", 0) or 0):
        return False, stderr or stdout
    return True, stdout


async def grep_tool(
    pattern: str,
    sandbox: Any,
    path: str = "vscode",
    file_pattern: str = "",
    context_lines: int = 2,
    case_insensitive: bool = False,
) -> str:
    """Search for a pattern in files using ripgrep.

    Args:
        pattern: Text or regex to search for inside files.
        path: Directory to search in.
        file_pattern: Only search files matching this glob, e.g. *.ts or *.py.
        context_lines: Lines of context around each match.
        case_insensitive: Ignore case when matching.
    """
    max_lines = 50
    flags = ["-n", "--max-filesize", "100K"]
    if context_lines > 0:
        flags.extend(["-C", str(min(context_lines, 5))])
    if case_insensitive:
        flags.append("-i")
    if file_pattern:
        if file_pattern.startswith(".") and not file_pattern.startswith("*"):
            file_pattern = "*" + file_pattern
        flags.extend(["-g", shlex.quote(file_pattern)])
    command = (
        f"rg {' '.join(flags)} {shlex.quote(pattern)} {shlex.quote(path)} "
        f"2>&1 | head -{max_lines + 1}"
    )
    success, output = await run_sandbox_command(sandbox, command)
    if not success:
        return f"Error: {output[:100]}"
    if not output.strip():
        return "No matches found"
    lines = [line[:300] + "..." if len(line) > 300 else line for line in output.split("\n")]
    if len(lines) > max_lines:
        output = "\n".join(lines[:max_lines])
        return (
            f"{output}\n\n[TRUNCATED - results exceed {max_lines} lines. "
            "Narrow your search with a more specific pattern or file_pattern]"
        )
    return output


async def list_files(path: str, sandbox: Any) -> str:
    """List files and directories at a path.

    Args:
        path: Directory path to list contents of.
    """
    success, output = await run_sandbox_command(sandbox, f"ls -la {shlex.quote(path)}")
    if not success:
        return f"Error: {output[:100]}"
    return output.strip() or "Empty directory"


async def read_file(
    file_path: str,
    sandbox: Any,
    start_line: int = 1,
    num_lines: int = 100,
) -> str:
    """Read lines from a file.

    Args:
        file_path: Path to the file.
        start_line: Line number to start from, 1-indexed.
        num_lines: Number of lines to read, max 50.
    """
    num_lines = min(num_lines, 50)
    end_line = start_line + num_lines - 1
    command = f"sed -n '{start_line},{end_line + 1}p' {shlex.quote(file_path)}"
    success, output = await run_sandbox_command(sandbox, command)
    if not success:
        return f"Error: {output[:100]}"
    if not output.strip():
        return f"No content at lines {start_line}-{end_line}"
    lines = output.split("\n")
    if len(lines) > num_lines:
        output = "\n".join(lines[:num_lines])
        return (
            f"Lines {start_line}-{end_line} of {file_path}:\n{output}\n\n"
            f"[MORE CONTENT BELOW - use start_line={end_line + 1} to continue]"
        )
    return f"Lines {start_line}-{end_line} of {file_path}:\n{output}"


def judge_reward_factory(
    judge_model: str,
    judge_base_url: str,
    judge_api_key_var: str,
):
    ensure_keys([judge_api_key_var])
    judge_client = AsyncOpenAI(
        api_key=os.environ[judge_api_key_var],
        base_url=judge_base_url,
    )

    async def yes_no_judge(question: str, answer: str, response: str) -> bool:
        judge_response = await judge_client.chat.completions.create(
            model=judge_model,
            messages=[
                {
                    "role": "user",
                    "content": JUDGE_PROMPT.format(
                        question=question,
                        answer=answer,
                        response=response,
                    ),
                }
            ],
        )
        text = judge_response.choices[0].message.content or ""
        return "yes" in text.lower()

    @vf.reward(weight=0.4)
    async def correct_answer(task: vf.Task, state: vf.State) -> float:
        response = parser.parse_answer(state["completion"]) or ""
        is_correct = await yes_no_judge(
            str(task["question"]), str(task["answer"]), response
        )
        state["_is_correct"] = is_correct
        return 1.0 if is_correct else 0.0

    @vf.reward(weight=0.4)
    async def correct_file_paths(task: vf.Task, state: vf.State) -> float:
        file_path_1 = task.get("file_path")
        file_path_2 = task.get("file_path_2")
        if not file_path_1:
            return 0.0
        response = parser.parse_answer(state["completion"]) or ""
        found_1 = await yes_no_judge(
            "Does the response reference this file path?", str(file_path_1), response
        )
        state["_found_file_1"] = found_1
        if not file_path_2:
            state["_found_file_2"] = True
            state["_is_single_file"] = True
            return 1.0 if found_1 else 0.0
        found_2 = await yes_no_judge(
            "Does the response reference this file path?", str(file_path_2), response
        )
        state["_found_file_2"] = found_2
        state["_is_single_file"] = False
        if found_1 and found_2:
            return 1.0
        if found_1 or found_2:
            return 0.3
        return 0.0

    return [correct_answer, correct_file_paths]


@vf.reward(weight=0.2)
async def parallel_tool_calls(task: vf.Task, state: vf.State) -> float:
    del task
    completion = state["completion"]
    if not isinstance(completion, list):
        return 0.0
    counts = [
        len(msg["tool_calls"])
        for msg in completion
        if isinstance(msg, dict)
        and msg.get("role") == "assistant"
        and isinstance(msg.get("tool_calls"), list)
        and msg["tool_calls"]
    ]
    if not counts:
        return 0.0
    return min((sum(counts) / len(counts)) / 8.0, 1.0)


@vf.reward(weight=0.0, stage="group")
async def efficiency_bonus_for_correct(
    tasks: list[vf.Task],
    states: list[vf.State],
) -> list[float]:
    del tasks
    rewards = [0.0] * len(states)
    correct_indices = [
        index
        for index, state in enumerate(states)
        if state.get("_found_file_1", False) and state.get("_found_file_2", False)
    ]
    if not correct_indices:
        return rewards
    turn_counts = [len(state.get("trajectory", [])) for state in states]
    min_turns = min(turn_counts[index] for index in correct_indices)
    for index in correct_indices:
        rewards[index] = min_turns / max(turn_counts[index], 1)
    return rewards


def load_toolset(
    config: vf.ToolsetConfig | Mapping[str, object] | None = None,
    sandbox_image: str = "python:3.11-slim",
    sandbox_timeout_minutes: int = 60,
    sandbox_command_timeout_seconds: int = 60,
    sandbox_setup_timeout_seconds: int = 600,
) -> vf.Toolset:
    return vf.Toolset(
        tools=[grep_tool, list_files, read_file],
        sandbox={
            "image": sandbox_image,
            "scope": "rollout",
            "network_access": True,
            "timeout_minutes": sandbox_timeout_minutes,
            "command_timeout": sandbox_command_timeout_seconds,
            "setup_timeout": sandbox_setup_timeout_seconds,
            "setup_commands": [
                "apt-get update && apt-get install -y git ripgrep",
                "git clone --depth 1 https://github.com/microsoft/vscode.git",
                "test -d vscode",
            ],
        },
        config=config,
    )


def load_taskset(
    config: vf.TasksetConfig | Mapping[str, object] | None = None,
    dataset_name: str | None = None,
    train_ratio: float | None = None,
    max_turns: int | None = None,
    system_prompt: str | None = None,
    judge_model: str | None = None,
    judge_base_url: str | None = None,
    judge_api_key_var: str | None = None,
    sandbox_image: str | None = None,
    sandbox_timeout_minutes: int | None = None,
    sandbox_command_timeout_seconds: int | None = None,
    sandbox_setup_timeout_seconds: int | None = None,
) -> vf.Taskset:
    taskset_config = SweGrepTasksetConfig.from_config(
        config,
        dataset_name=dataset_name,
        train_ratio=train_ratio,
        max_turns=max_turns,
        system_prompt=system_prompt,
        judge_model=judge_model,
        judge_base_url=judge_base_url,
        judge_api_key_var=judge_api_key_var,
        sandbox_image=sandbox_image,
        sandbox_timeout_minutes=sandbox_timeout_minutes,
        sandbox_command_timeout_seconds=sandbox_command_timeout_seconds,
        sandbox_setup_timeout_seconds=sandbox_setup_timeout_seconds,
    )
    dataset_cache: dict[str, Dataset] = {}

    def split(name: str) -> Dataset:
        if not dataset_cache:
            train, test = convert_dataset(
                taskset_config.train_ratio,
                taskset_config.dataset_name,
            )
            dataset_cache["train"] = train
            dataset_cache["test"] = test
        return dataset_cache[name]

    return vf.Taskset(
        source=lambda: rows(split("train"), taskset_config.max_turns),
        eval_source=lambda: rows(split("test"), taskset_config.max_turns),
        system_prompt=taskset_config.system_prompt,
        toolsets=[
            load_toolset(
                sandbox_image=taskset_config.sandbox_image,
                sandbox_timeout_minutes=taskset_config.sandbox_timeout_minutes,
                sandbox_command_timeout_seconds=taskset_config.sandbox_command_timeout_seconds,
                sandbox_setup_timeout_seconds=taskset_config.sandbox_setup_timeout_seconds,
            )
        ],
        rewards=[
            *judge_reward_factory(
                taskset_config.judge_model,
                taskset_config.judge_base_url,
                taskset_config.judge_api_key_var,
            ),
            parallel_tool_calls,
        ],
        config=taskset_config,
    )


def load_harness(
    config: vf.HarnessConfig | Mapping[str, object] | None = None,
) -> vf.Harness:
    return vf.Harness(config=config)


def load_environment(
    config: vf.EnvConfig | Mapping[str, object] | None = None,
    dataset_name: str | None = None,
    train_ratio: float | None = None,
    max_turns: int | None = None,
    system_prompt: str | None = None,
    judge_model: str | None = None,
    judge_base_url: str | None = None,
    judge_api_key_var: str | None = None,
    sandbox_image: str | None = None,
    sandbox_timeout_minutes: int | None = None,
    sandbox_command_timeout_seconds: int | None = None,
    sandbox_setup_timeout_seconds: int | None = None,
) -> vf.Env:
    config = vf.EnvConfig.from_config(
        config,
        taskset=SweGrepTasksetConfig.from_config(
            dataset_name=dataset_name,
            train_ratio=train_ratio,
            max_turns=max_turns,
            system_prompt=system_prompt,
            judge_model=judge_model,
            judge_base_url=judge_base_url,
            judge_api_key_var=judge_api_key_var,
            sandbox_image=sandbox_image,
            sandbox_timeout_minutes=sandbox_timeout_minutes,
            sandbox_command_timeout_seconds=sandbox_command_timeout_seconds,
            sandbox_setup_timeout_seconds=sandbox_setup_timeout_seconds,
        ),
    )
    return vf.Env(
        taskset=load_taskset(config=config.taskset),
        harness=load_harness(config=config.harness),
    )
