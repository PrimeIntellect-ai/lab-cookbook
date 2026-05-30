import hashlib
import json
import shlex
from typing import Protocol

import verifiers as vf
from math_verify import parse, verify
from verifiers.errors import SandboxError
from verifiers.utils.data_utils import extract_boxed_answer, load_example_dataset


class SandboxResult(Protocol):
    stdout: object
    stderr: object
    exit_code: int


class PythonSandbox(Protocol):
    async def upload_bytes(self, path: str, data: bytes) -> object: ...

    async def execute(self, command: str, timeout: int = 0) -> SandboxResult: ...


class MathPythonTasksetConfig(vf.TasksetConfig):
    dataset_name: str = "math"
    num_examples: int = -1
    system_prompt: vf.SystemPrompt = None


class MathPythonTaskset(vf.Taskset[MathPythonTasksetConfig]):
    python_packages: tuple[str, ...] = (
        "ipython",
        "ipykernel",
        "jupyter-client",
        "numpy",
        "sympy",
        "scipy",
    )
    sandbox_root: str = "/tmp/vf_math_python"
    executor_path: str = "/tmp/vf_math_python/execute_cell.py"
    kernel_start_timeout_seconds: int = 30
    python_timeout_seconds: int = 60

    def install_command(self) -> str:
        packages = shlex.join(self.python_packages)
        return f"python -m pip install --disable-pip-version-check -q {packages}"

    def kernel_id(self, state: vf.State) -> str:
        trajectory_id = str(state["trajectory_id"])
        return hashlib.sha256(trajectory_id.encode()).hexdigest()[:16]

    def kernel_paths(self, state: vf.State) -> tuple[str, str, str, str]:
        kernel_id = self.kernel_id(state)
        root = self.sandbox_root
        return (
            f"{root}/{kernel_id}.json",
            f"{root}/{kernel_id}.pid",
            f"{root}/{kernel_id}.log",
            f"{root}/{kernel_id}.input.json",
        )

    def executor_source(self) -> str:
        return """
import json
import sys

from jupyter_client import BlockingKernelClient


def execute_cell(connection_file: str, code: str, timeout: int) -> tuple[bool, str]:
    client = BlockingKernelClient(connection_file=connection_file)
    client.load_connection_file()
    client.start_channels()
    outputs = []
    errored = False
    try:
        msg_id = client.execute(code, store_history=True, allow_stdin=False)
        while True:
            msg = client.get_iopub_msg(timeout=timeout)
            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            msg_type = msg["header"]["msg_type"]
            content = msg["content"]
            if msg_type == "stream":
                outputs.append(content.get("text", ""))
            elif msg_type in {"execute_result", "display_data"}:
                text = content.get("data", {}).get("text/plain")
                if text:
                    outputs.append(str(text) + "\\n")
            elif msg_type == "error":
                errored = True
                outputs.append("\\n".join(content.get("traceback", [])) + "\\n")
            elif msg_type == "status" and content.get("execution_state") == "idle":
                break
        client.get_shell_msg(timeout=timeout)
    finally:
        client.stop_channels()
    return errored, "".join(outputs).strip()


def main() -> None:
    with open(sys.argv[1]) as f:
        payload = json.load(f)
    errored, output = execute_cell(
        payload["connection_file"],
        payload["code"],
        int(payload["timeout"]),
    )
    if output:
        print(output)
    if errored:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
"""

    async def upload_executor(self, sandbox: PythonSandbox) -> None:
        await sandbox.execute(f"mkdir -p {shlex.quote(self.sandbox_root)}")
        await sandbox.upload_bytes(self.executor_path, self.executor_source().encode())

    async def start_kernel(self, sandbox: PythonSandbox, state: vf.State) -> str:
        connection_file, pid_file, log_file, _ = self.kernel_paths(state)
        command = f"""
set -e
mkdir -p {shlex.quote(self.sandbox_root)}
if [ -s {shlex.quote(connection_file)} ] && [ -s {shlex.quote(pid_file)} ] && kill -0 "$(cat {shlex.quote(pid_file)})" 2>/dev/null; then
  exit 0
fi
rm -f {shlex.quote(connection_file)}
if [ ! -s {shlex.quote(connection_file)} ]; then
  nohup python -m ipykernel_launcher -f {shlex.quote(connection_file)} > {shlex.quote(log_file)} 2>&1 &
  echo $! > {shlex.quote(pid_file)}
fi
for _ in $(seq 1 {self.kernel_start_timeout_seconds * 10}); do
  [ -s {shlex.quote(connection_file)} ] && exit 0
  sleep 0.1
done
cat {shlex.quote(log_file)} >&2 || true
exit 1
"""
        result = await sandbox.execute(command, timeout=self.kernel_start_timeout_seconds)
        if result.exit_code:
            raise SandboxError(f"Python kernel failed to start: {result.stderr or ''}")
        return connection_file

    async def python(self, code: str, sandbox: PythonSandbox, state: vf.State) -> str:
        """Execute Python code in the rollout sandbox."""
        await self.upload_executor(sandbox)
        connection_file = await self.start_kernel(sandbox, state)
        _, _, _, input_path = self.kernel_paths(state)
        payload = {
            "connection_file": connection_file,
            "code": code,
            "timeout": self.python_timeout_seconds,
        }
        await sandbox.upload_bytes(input_path, json.dumps(payload).encode())
        result = await sandbox.execute(
            f"python {shlex.quote(self.executor_path)} {shlex.quote(input_path)}",
            timeout=self.python_timeout_seconds,
        )
        stdout = str(result.stdout or "")
        stderr = str(result.stderr or "")
        if result.exit_code:
            raise SandboxError(f"Python command failed: {stdout or stderr}")
        return stdout.strip() or "(no output)"

    def load_system_prompt(self, config: MathPythonTasksetConfig) -> vf.SystemPrompt:
        if config.system_prompt is not None:
            return config.system_prompt
        return (
            "Use Python for all calculations. Give your answer inside \\boxed{}."
            "\n\n"
            "The python tool runs an IPython kernel in a python:3.11-slim sandbox. "
            "NumPy, SymPy, and SciPy are installed."
        )

    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        source_split = "test" if split in ("eval", "test") else "train"
        return load_example_dataset(
            self.config.dataset_name,
            source_split,
            n=self.config.num_examples,
        )

    def load_toolsets(self, config: MathPythonTasksetConfig) -> vf.Toolsets:
        return {
            "python": vf.Toolset(
                tools=[self.python],
                write=True,
                sandbox=vf.SandboxConfig(
                    image="python:3.11-slim",
                    scope="group",
                    setup_commands=[self.install_command()],
                    setup_timeout=300,
                ),
            )
        }

    @vf.cleanup(priority=10)
    async def collect_python_commands(self, state: vf.State) -> None:
        state["commands"] = list(state.get("sandbox_commands", []))
        state.pop("sandbox_commands", None)

    @vf.reward(weight=1.0)
    async def correct_answer(self, task: vf.Task, state: vf.State) -> float:
        completion = state.get("completion") or []
        messages = vf.get_messages(completion, role="assistant")
        response_text = str(messages[-1].content or "") if messages else ""
        response = extract_boxed_answer(response_text)
        answer = str(task["answer"])
        if not response or len(response) > 50_000:
            return 0.0

        try:
            parsed_answer = parse(rf"\boxed{{{answer}}}", parsing_timeout=5)
            parsed_response = parse(rf"\boxed{{{response}}}", parsing_timeout=5)
            return float(verify(parsed_answer, parsed_response, timeout_seconds=5))
        except Exception:
            return 0.0


def load_taskset(config: MathPythonTasksetConfig) -> MathPythonTaskset:
    return MathPythonTaskset(config=config)


def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(
        taskset=vf.load_taskset(config=config.taskset),
        harness=vf.load_harness(config=config.harness),
    )
