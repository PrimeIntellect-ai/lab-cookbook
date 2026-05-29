import functools
import os
import shlex

import verifiers as vf
from datasets import Dataset, load_dataset
from openai import AsyncOpenAI

JUDGE_PROMPT = """Given a ground truth answer and a response, determine if the answer is correct.

Question:
{question}

Ground truth answer:
{answer}

Response:
{response}

Respond either 'yes' or 'no' only.
"""

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


class SweGrepTasksetConfig(vf.TasksetConfig):
    dataset_name: str = "cdreetz/swe-grep-v2"
    train_ratio: float = 0.9
    max_turns: int = 2
    repo_url: str = "https://github.com/microsoft/vscode.git"
    repo_path: str = "vscode"
    system_prompt: vf.PromptInput | vf.SystemPromptConfig | None = SYSTEM_PROMPT


def assistant_text(state: vf.State) -> str:
    messages = vf.get_messages(state.get("completion") or [], role="assistant")
    if not messages:
        return ""
    return str(messages[-1].content or "")


@functools.lru_cache(maxsize=8)
def load_splits(dataset_name: str, train_ratio: float) -> tuple[Dataset, Dataset]:
    dataset = load_dataset(dataset_name, split="train")
    dataset = dataset.filter(lambda row: row["check"] == "Yes")
    dataset = dataset.rename_columns({"user_query": "question", "ground_truth": "answer"})
    drop = [name for name in ("file_chunk", "check") if name in dataset.column_names]
    if drop:
        dataset = dataset.remove_columns(drop)
    split = dataset.train_test_split(test_size=1 - train_ratio, seed=42)
    return split["train"], split["test"]


class SweGrepTaskset(vf.Taskset[SweGrepTasksetConfig]):
    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        train, test = load_splits(self.config.dataset_name, self.config.train_ratio)
        dataset = train if split == "train" else test
        for index, raw_row in enumerate(dataset):
            if not isinstance(raw_row, dict):
                raise TypeError("Dataset rows must be dicts.")
            question = str(raw_row["question"])
            yield {
                **raw_row,
                "example_id": index,
                "prompt": [{"role": "user", "content": question}],
                "max_turns": self.config.max_turns,
            }

    def load_toolsets(self, config: SweGrepTasksetConfig) -> vf.Toolsets:
        _ = config
        repo_path = self.config.repo_path

        async def grep_tool(
            pattern: str,
            sandbox,
            state,
            *,
            path: str = repo_path,
            file_pattern: str = "",
            context_lines: int = 2,
            case_insensitive: bool = False,
        ) -> str:
            """Search for a pattern in files using ripgrep."""
            _ = state
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
            result = await sandbox.execute(command)
            if result.exit_code:
                error = (result.stderr or result.stdout or "")[:100]
                return f"Error: {error}"
            output = result.stdout or ""
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

        async def list_files(path: str, sandbox, state) -> str:
            """List files and directories at a path."""
            _ = state
            result = await sandbox.execute(f"ls -la {shlex.quote(path)}")
            if result.exit_code:
                error = (result.stderr or result.stdout or "")[:100]
                return f"Error: {error}"
            output = result.stdout or ""
            return output.strip() or "Empty directory"

        async def read_file(
            file_path: str,
            sandbox,
            state,
            *,
            start_line: int = 1,
            num_lines: int = 100,
        ) -> str:
            """Read lines from a file."""
            _ = state
            num_lines = min(num_lines, 50)
            end_line = start_line + num_lines - 1
            command = f"sed -n '{start_line},{end_line + 1}p' {shlex.quote(file_path)}"
            result = await sandbox.execute(command)
            if result.exit_code:
                error = (result.stderr or result.stdout or "")[:100]
                return f"Error: {error}"
            output = result.stdout or ""
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

        return {
            "swe_grep": vf.Toolset(
                tools=[grep_tool, list_files, read_file],
                sandbox=vf.SandboxConfig(
                    image="python:3.11-slim",
                    scope="rollout",
                    network_access=True,
                    timeout_minutes=60,
                    command_timeout=60,
                    setup_timeout=600,
                    setup_commands=[
                        "apt-get update && apt-get install -y git ripgrep",
                        f"git clone --depth 1 {self.config.repo_url} {self.config.repo_path}",
                        f"test -d {self.config.repo_path}",
                    ],
                ),
            )
        }

    @vf.update
    async def score_with_judge(self, task: vf.Task, state: vf.State) -> None:
        endpoint = state.get_endpoint_config(api="chat")
        judge = AsyncOpenAI(
            api_key=os.getenv(endpoint.api_key_var, ""),
            base_url=endpoint.base_url,
        )
        response = await judge.chat.completions.create(
            model=endpoint.model,
            messages=[
                {
                    "role": "user",
                    "content": JUDGE_PROMPT.format(
                        question=task["question"],
                        answer=task["answer"],
                        response=assistant_text(state),
                    ),
                }
            ],
        )
        text = response.choices[0].message.content or ""
        state["judge_score"] = 1.0 if "yes" in text.lower() else 0.0

    @vf.reward(weight=0.4)
    async def correct_answer(self, task: vf.Task, state: vf.State) -> float:
        _ = task
        return float(state.get("judge_score", 0.0))

    @vf.reward(weight=0.4)
    async def correct_file_paths(self, task: vf.Task, state: vf.State) -> float:
        file_1 = task.get("file_path")
        file_2 = task.get("file_path_2")
        if not file_1:
            return 0.0
        response = assistant_text(state)
        found_1 = str(file_1) in response
        state["found_file_1"] = found_1
        if not file_2:
            state["found_file_2"] = True
            return 1.0 if found_1 else 0.0
        found_2 = str(file_2) in response
        state["found_file_2"] = found_2
        if found_1 and found_2:
            return 1.0
        if found_1 or found_2:
            return 0.3
        return 0.0

    @vf.reward(weight=0.2)
    async def parallel_tool_calls(self, task: vf.Task, state: vf.State) -> float:
        _ = task
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
        self, tasks: list[vf.Task], states: list[vf.State]
    ) -> list[float]:
        _ = tasks
        rewards = [0.0] * len(states)
        correct_indices = [
            index
            for index, state in enumerate(states)
            if state.get("found_file_1", False) and state.get("found_file_2", False)
        ]
        if not correct_indices:
            return rewards
        turn_counts = [len(state.get("trajectory", [])) for state in states]
        min_turns = min(turn_counts[index] for index in correct_indices)
        for index in correct_indices:
            rewards[index] = min_turns / max(turn_counts[index], 1)
        return rewards


def load_taskset(config: SweGrepTasksetConfig) -> SweGrepTaskset:
    return SweGrepTaskset(config=config)


def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(
        taskset=vf.load_taskset(config=config.taskset),
        harness=vf.Harness(config=config.harness),
    )
