import asyncio
import hashlib
import os
import shlex
from pathlib import Path

import verifiers.v1 as vf
from datasets import Dataset, load_dataset
from verifiers.v1.dialects import ChatDialect

JUDGE_PROMPT = """Given a ground truth answer and a response, determine if the answer is correct.

Question:
{question}

Ground truth answer:
{answer}

Response:
{response}

Respond either "yes" or "no" only.
"""

SYSTEM = """You are a helpful assistant that can answer questions about a codebase.
Use the provided grep, list_files, and read_file tools to search efficiently.

When providing your final answer, include all file paths where you found relevant information.
"""


class SweGrepTask(vf.Task):
    question: str
    answer: str
    file_path: str
    file_path_2: str | None = None


class SweGrepToolConfig(vf.ToolsetConfig):
    repo_url: str = "https://github.com/microsoft/vscode.git"
    repo_path: str = "vscode"
    setup_timeout_seconds: int = 600
    command_timeout_seconds: int = 60
    shared: bool = True


class JudgeConfig(vf.BaseClientConfig):
    model: str = "openai/gpt-4.1-mini"


class SweGrepConfig(vf.TasksetConfig):
    dataset_name: str = "cdreetz/swe-grep-v2"
    train_ratio: float = 0.9
    judge: JudgeConfig = JudgeConfig()
    tools: SweGrepToolConfig = SweGrepToolConfig()


async def run_shell(
    command: str, cwd: str | None = None, timeout: int = 60
) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        "sh",
        "-lc",
        command,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    return (
        proc.returncode or 0,
        stdout.decode(errors="replace"),
        stderr.decode(errors="replace"),
    )


class SweGrepToolset(vf.Toolset[SweGrepToolConfig]):
    TOOL_PREFIX = "swe_grep"

    async def setup(self) -> None:
        digest = hashlib.sha256(self.config.repo_url.encode()).hexdigest()[:12]
        root = Path(os.environ.get("VF_SWE_GREP_CACHE", "/tmp/vf_swe_grep")) / digest
        root.mkdir(parents=True, exist_ok=True)
        self.repo_dir = root / self.config.repo_path
        if not self.repo_dir.exists():
            code, stdout, stderr = await run_shell(
                f"git clone --depth 1 {shlex.quote(self.config.repo_url)} {shlex.quote(str(self.repo_dir))}",
                timeout=self.config.setup_timeout_seconds,
            )
            if code:
                raise RuntimeError(stderr or stdout)

    @vf.tool
    async def grep_tool(
        self,
        pattern: str,
        path: str = "",
        file_pattern: str = "",
        context_lines: int = 2,
        case_insensitive: bool = False,
    ) -> str:
        """Search for a pattern in files using ripgrep."""
        search_path = path or str(self.repo_dir)
        flags = ["-n", "--max-filesize", "100K"]
        if context_lines > 0:
            flags.extend(["-C", str(min(context_lines, 5))])
        if case_insensitive:
            flags.append("-i")
        if file_pattern:
            if file_pattern.startswith(".") and not file_pattern.startswith("*"):
                file_pattern = "*" + file_pattern
            flags.extend(["-g", file_pattern])
        command = (
            f"rg {' '.join(shlex.quote(flag) for flag in flags)} "
            f"{shlex.quote(pattern)} {shlex.quote(search_path)} 2>&1 | head -51"
        )
        code, stdout, stderr = await run_shell(command, timeout=self.config.command_timeout_seconds)
        if code:
            return f"Error: {(stderr or stdout)[:100]}"
        if not stdout.strip():
            return "No matches found"
        lines = [line[:300] + "..." if len(line) > 300 else line for line in stdout.splitlines()]
        return "\n".join(lines[:50])

    @vf.tool
    async def list_files(self, path: str) -> str:
        """List files and directories at a path."""
        target = path or str(self.repo_dir)
        code, stdout, stderr = await run_shell(
            f"ls -la {shlex.quote(target)}",
            timeout=self.config.command_timeout_seconds,
        )
        if code:
            return f"Error: {(stderr or stdout)[:100]}"
        return stdout.strip() or "Empty directory"

    @vf.tool
    async def read_file(self, file_path: str, start_line: int = 1, num_lines: int = 100) -> str:
        """Read lines from a file."""
        num_lines = min(num_lines, 50)
        end_line = start_line + num_lines - 1
        code, stdout, stderr = await run_shell(
            f"sed -n '{start_line},{end_line + 1}p' {shlex.quote(file_path)}",
            timeout=self.config.command_timeout_seconds,
        )
        if code:
            return f"Error: {(stderr or stdout)[:100]}"
        if not stdout.strip():
            return f"No content at lines {start_line}-{end_line}"
        lines = stdout.splitlines()
        more = (
            f"\n\n[MORE CONTENT BELOW - use start_line={end_line + 1} to continue]"
            if len(lines) > num_lines
            else ""
        )
        return (
            f"Lines {start_line}-{end_line} of {file_path}:\n" + "\n".join(lines[:num_lines]) + more
        )


def load_splits(dataset_name: str, train_ratio: float) -> tuple[Dataset, Dataset]:
    dataset = load_dataset(dataset_name, split="train")
    dataset = dataset.filter(lambda row: row["check"] == "Yes")
    dataset = dataset.rename_columns({"user_query": "question", "ground_truth": "answer"})
    drop = [name for name in ("file_chunk", "check") if name in dataset.column_names]
    if drop:
        dataset = dataset.remove_columns(drop)
    split = dataset.train_test_split(test_size=1 - train_ratio, seed=42)
    return split["train"], split["test"]


class SweGrepTaskset(vf.Taskset[SweGrepTask, SweGrepConfig]):
    def load_tasks(self) -> list[SweGrepTask]:
        train, _ = load_splits(self.config.dataset_name, self.config.train_ratio)
        return [
            SweGrepTask(
                idx=i,
                prompt=f"{SYSTEM}\n\n{row['question']}",
                question=str(row["question"]),
                answer=str(row["answer"]),
                file_path=str(row.get("file_path") or ""),
                file_path_2=(str(row["file_path_2"]) if row.get("file_path_2") else None),
            )
            for i, row in enumerate(train)
        ]

    def tools(self, task: SweGrepTask) -> list[vf.Toolset]:
        _ = task
        return [SweGrepToolset(self.config.tools)]

    def assistant_text(self, trace: vf.Trace) -> str:
        return trace.assistant_messages[-1].content or "" if trace.assistant_messages else ""

    @vf.reward(weight=0.4)
    async def correct_answer(self, task: SweGrepTask, trace: vf.Trace) -> float:
        prompt = JUDGE_PROMPT.format(
            question=task.question,
            answer=task.answer,
            response=self.assistant_text(trace),
        )
        client = vf.resolve_client(self.config.judge)
        try:
            verdict = await client.get_response(
                ChatDialect(),
                {"messages": [{"role": "user", "content": prompt}]},
                self.config.judge.model,
                vf.SamplingConfig(),
            )
        finally:
            await client.close()
        return float("yes" in (verdict.message.content or "").lower())

    @vf.reward(weight=0.4)
    async def correct_file_paths(self, task: SweGrepTask, trace: vf.Trace) -> float:
        response = self.assistant_text(trace)
        found_1 = bool(task.file_path and task.file_path in response)
        trace.info["found_file_1"] = found_1
        if task.file_path_2 is None:
            trace.info["found_file_2"] = True
            return float(found_1)
        found_2 = task.file_path_2 in response
        trace.info["found_file_2"] = found_2
        if found_1 and found_2:
            return 1.0
        return 0.3 if found_1 or found_2 else 0.0

    @vf.reward(weight=0.2)
    async def parallel_tool_calls(self, trace: vf.Trace) -> float:
        counts = [len(message.tool_calls or []) for message in trace.assistant_messages]
        counts = [count for count in counts if count]
        if not counts:
            return 0.0
        return min((sum(counts) / len(counts)) / 8.0, 1.0)

    @vf.group_reward(weight=0.1)
    async def efficiency_bonus_for_correct(self, traces: list[vf.Trace]) -> list[float]:
        rewards = [0.0] * len(traces)
        correct = [
            i
            for i, trace in enumerate(traces)
            if trace.info.get("found_file_1") and trace.info.get("found_file_2")
        ]
        if not correct:
            return rewards
        min_turns = min(traces[i].num_turns for i in correct)
        for i in correct:
            rewards[i] = min_turns / max(traces[i].num_turns, 1)
        return rewards


if __name__ == "__main__":
    SweGrepToolset.run()


__all__ = ["SweGrepTaskset"]
