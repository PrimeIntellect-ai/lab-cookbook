import asyncio
import json
import sys
import textwrap
from typing import cast

import verifiers.v1 as vf
from pydantic import Field

SYSTEM = (
    "Use the python_execute tool for calculations when useful. Give your final answer "
    "inside \\boxed{}."
)


def extract_boxed(text: str) -> str:
    start = text.rfind("\\boxed{")
    if start < 0:
        return ""
    start += len("\\boxed{")
    end = text.find("}", start)
    return text[start:end].strip() if end >= 0 else ""


class MathPythonTask(vf.Task):
    answer: str


class PythonState(vf.State):
    executions: int = 0
    restarts: int = 0
    last_error: str = ""
    history: list[str] = Field(default_factory=list)


class PythonToolConfig(vf.ToolsetConfig):
    timeout_seconds: int = 60
    runtime: vf.RuntimeConfig = vf.DockerConfig(image="python:3.11-slim", workdir="/tmp")


class MathPythonConfig(vf.TasksetConfig):
    dataset_name: str = "PrimeIntellect/Hendrycks-Math"
    dataset_subset: str = "default"
    dataset_split: str = "train"
    question_key: str = "question"
    answer_key: str = "answer"
    num_tasks: int = -1
    tools: PythonToolConfig = PythonToolConfig()


WORKER_SOURCE = r"""
import contextlib
import io
import json
import sys
import traceback

namespace = {}


def run(code):
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            try:
                compiled = compile(code, "<python_execute>", "eval")
            except SyntaxError:
                compiled = compile(code, "<python_execute>", "exec")
                exec(compiled, namespace)
            else:
                value = eval(compiled, namespace)
                if value is not None:
                    print(repr(value))
        except Exception:
            traceback.print_exc(file=stderr)
            return {"ok": False, "stdout": stdout.getvalue(), "stderr": stderr.getvalue()}
    return {"ok": True, "stdout": stdout.getvalue(), "stderr": stderr.getvalue()}


for line in sys.stdin:
    try:
        request = json.loads(line)
        response = run(str(request["code"]))
    except Exception:
        response = {"ok": False, "stdout": "", "stderr": traceback.format_exc()}
    print(json.dumps(response), flush=True)
"""


class PythonToolset(vf.Toolset[PythonToolConfig, PythonState]):
    TOOL_PREFIX = "python"

    async def setup_task(self, task: MathPythonTask) -> None:
        _ = task
        await self._start_worker()

    async def _start_worker(self) -> None:
        self._lock = asyncio.Lock()
        self._proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-u",
            "-c",
            WORKER_SOURCE,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    async def _restart_worker(self) -> None:
        proc = getattr(self, "_proc", None)
        if proc is not None and proc.returncode is None:
            proc.kill()
            await proc.communicate()
        self.state.restarts += 1
        await self._start_worker()

    @vf.tool
    async def execute(self, code: str) -> str:
        """Execute Python code in the rollout's persistent interpreter."""
        proc = getattr(self, "_proc", None)
        if proc is None or proc.returncode is not None:
            await self._restart_worker()
            proc = self._proc
        async with self._lock:
            assert proc.stdin is not None
            assert proc.stdout is not None
            payload = json.dumps({"code": code}).encode() + b"\n"
            proc.stdin.write(payload)
            await proc.stdin.drain()
            try:
                line = await asyncio.wait_for(
                    proc.stdout.readline(),
                    timeout=self.config.timeout_seconds,
                )
            except TimeoutError:
                await self._restart_worker()
                self.state.last_error = f"timed out after {self.config.timeout_seconds} seconds"
                return f"Error: {self.state.last_error}. The Python session was restarted."
        if not line:
            await self._restart_worker()
            self.state.last_error = "Python worker exited without a response"
            return f"Error: {self.state.last_error}. The Python session was restarted."
        response = json.loads(line)
        self.state.executions += 1
        self.state.history.append(code[:500])
        self.state.history = self.state.history[-20:]
        stdout = str(response.get("stdout") or "").strip()
        stderr = str(response.get("stderr") or "").strip()
        if not response.get("ok"):
            self.state.last_error = stderr[-1000:] or "execution failed"
            return "Error: " + self.state.last_error
        self.state.last_error = ""
        if stderr:
            return (
                textwrap.shorten(stderr, width=1000, placeholder="...")
                + "\n"
                + (stdout or "(no output)")
            )
        return stdout or "(no output)"


class MathPythonTaskset(vf.Taskset[MathPythonTask, MathPythonConfig, PythonState]):
    def load_tasks(self) -> list[MathPythonTask]:
        from datasets import load_dataset

        rows = load_dataset(
            self.config.dataset_name,
            self.config.dataset_subset,
            split=self.config.dataset_split,
        )
        if self.config.num_tasks > 0:
            rows = rows.select(range(min(self.config.num_tasks, len(rows))))
        return [
            MathPythonTask(
                idx=i,
                prompt=str(row[self.config.question_key]),
                system_prompt=SYSTEM,
                answer=str(row[self.config.answer_key]),
            )
            for i, row in enumerate(rows)
        ]

    def tools(self, task: MathPythonTask) -> list[vf.Toolset]:
        _ = task
        return [cast(vf.Toolset, PythonToolset(self.config.tools))]

    @vf.reward(weight=1.0)
    async def correct_answer(self, task: MathPythonTask, trace: vf.Trace) -> float:
        completion = trace.assistant_messages[-1].content if trace.assistant_messages else ""
        response = extract_boxed(completion or "")
        if not response:
            return 0.0
        try:
            from math_verify import parse, verify

            parsed_answer = parse(rf"\boxed{{{task.answer}}}", parsing_timeout=5)
            parsed_response = parse(rf"\boxed{{{response}}}", parsing_timeout=5)
            return float(verify(parsed_answer, parsed_response, timeout_seconds=5))
        except Exception:
            return float(response == task.answer)


if __name__ == "__main__":
    PythonToolset.run()


__all__ = ["MathPythonTaskset"]
