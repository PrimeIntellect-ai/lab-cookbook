# Coding Agents and Sandboxes

Train and evaluate coding agents in isolated runtimes.

Sandboxed execution environments let the model inspect and write code, run commands safely in a container, collect logs, and score edits programmatically and scalably.

This guide considers two examples of coding agent environments:
- [prime/math-python](https://app.primeintellect.ai/dashboard/environments/prime/math-python) -- a lightweight Python REPL environment for solving math problems with code
- [prime/opencode-harbor](https://app.primeintellect.ai/dashboard/environments/prime/opencode-harbor) -- a full CLI agent environment that runs OpenCode on Harbor tasks

## Sandboxes as Tools

`math-python` asks math questions that are easier to solve with code than by mental arithmetic. The model gets a Python tool backed by a sandbox, uses it for calculations, and returns a final boxed answer.

Run a small eval:

```bash
prime eval run prime/math-python \
  -m openai/gpt-5.4-nano \
  -n 5 \
  -r 2 \
  -t 1024
```

Or run with a config file:

```toml
# [configs/10/math-python-eval.toml](../../configs/10/math-python-eval.toml)
model = "openai/gpt-5.4-nano"
save_results = true

[[eval]]
env_id = "prime/math-python"
num_examples = 5
rollouts_per_example = 2

[eval.sampling]
max_tokens = 1024
```

```bash
prime eval run configs/10/math-python-eval.toml
```

This is the smallest useful sandbox pattern: one task, one Python toolset, one isolated runtime, and a deterministic reward.

### How Math-Python Is Built

[environments/math_python/math_python.py](../../environments/math_python/math_python.py) keeps everything on `MathPythonTaskset` — dataset, prompt, reward, the `python` tool, and the sandbox lifecycle helpers that tool needs.

**Config.** Two timeouts and a dataset selector, all serializable:

```python
class MathPythonTasksetConfig(vf.TasksetConfig):
    dataset_name: str = "math"
    num_examples: int = -1
    system_prompt: vf.SystemPrompt = None
    kernel_start_timeout_seconds: int = 30
    python_timeout_seconds: int = 60
```

**Sandbox contract on the toolset.** The toolset *owns* the sandbox spec, not the harness. That's the rule: when a tool needs a runtime, the `vf.Toolset` exposing it declares the image and setup:

```python
class MathPythonTaskset(vf.Taskset[MathPythonTasksetConfig]):
    python_packages: ClassVar[tuple[str, ...]] = (
        "ipython", "ipykernel", "jupyter-client", "numpy", "sympy", "scipy",
    )

    def load_sandbox(self) -> vf.SandboxConfig:
        return vf.SandboxConfig(
            image="python:3.11-slim",
            scope="group",
            setup_commands=[self.install_command()],
            setup_timeout=300,
        )

    def load_toolsets(self, config: MathPythonTasksetConfig) -> vf.Toolsets:
        _ = config
        return {
            "python": vf.Toolset(
                tools=[self.python],
                write=True,
                sandbox=self.load_sandbox(),
            )
        }
```

`scope="group"` reuses one sandbox across all rollouts of the same task group; `setup_commands` runs `pip install ...` once when the sandbox boots, so the model sees the same `numpy/sympy/scipy` everywhere. Standalone tasksets use an owned `vf.SandboxConfig(...)` like this; sharing a program/CLI sandbox uses `vf.SandboxConfig(prefer="program", ...)`; only declare `sandbox="program"` when the toolset truly cannot run without the harness-owned program sandbox.

**The Python tool.** One async method that talks to a per-rollout IPython kernel:

```python
async def python(self, code: str, sandbox: PythonSandbox, state: vf.State) -> str:
    """Execute Python code in the rollout sandbox."""
    await self.upload_executor(sandbox)
    connection_file = await self.start_kernel(sandbox, state)
    _, _, _, input_path = self.kernel_paths(state)
    payload = {
        "connection_file": connection_file,
        "code": code,
        "timeout": self.config.python_timeout_seconds,
    }
    await sandbox.upload_bytes(input_path, json.dumps(payload).encode())
    result = await sandbox.execute(
        f"python {shlex.quote(self.executor_path)} {shlex.quote(input_path)}",
        timeout=self.config.python_timeout_seconds,
    )
    ...
    return stdout.strip() or "(no output)"
```

The signature is the contract:

- `code: str` is the only model-visible parameter — that's what shows up in the tool schema.
- `sandbox: PythonSandbox` is injected by the framework because the toolset declared a sandbox. It's stripped from the schema, so the model never sees it.
- `state: vf.State` is also injected by name; the tool uses it to derive a stable per-rollout kernel id (`hashlib.sha256(str(state["trajectory_id"]).encode()).hexdigest()[:16]`) so each rollout gets its own kernel.

The `upload_executor` / `start_kernel` / `kernel_paths` / `executor_source` helpers all live as methods on the Taskset — they're real sandbox lifecycle, not abstraction overhead. They earn their place because they handle: uploading a tiny `execute_cell.py` script that talks to `jupyter_client`, starting an `ipykernel_launcher` in the sandbox and waiting for its connection file, and reading streamed output back.

**Per-rollout state via `@vf.cleanup`.** The sandbox records every command it ran. A cleanup hook lifts that record into a top-level state key for the saved rollout:

```python
@vf.cleanup(priority=10)
async def collect_python_commands(self, state: vf.State) -> None:
    state["commands"] = list(state.get("sandbox_commands", []))
    state.pop("sandbox_commands", None)
```

`@vf.cleanup` is the first lifecycle hook these guides use, and it attaches to the **end** of the [rollout loop from guide 04](../04-prompt-optimization/README.md#how-a-multi-turn-rollout-runs) — step 4, after the conversation is rendered. It's where you release per-rollout resources (close a session, flush a record). Its sibling `@vf.teardown` runs once when the whole environment shuts down, not per rollout — use it for resources shared across rollouts (a connection pool, a cached client). Both must be idempotent: a cancelled rollout can hit `@vf.cleanup` with partial state, so read keys with `.get(...)` defaults and tolerate missing ones. (Guide 11 covers the hooks at the *start* and *exit* of the loop.)

**Reward.** Symbolic equivalence via the `math-verify` library, with both the gold answer and the model's response parsed as boxed expressions:

```python
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
```

The 50,000-char cap is a guard against pathological parser timeouts on adversarial inputs. `parsing_timeout` and `timeout_seconds` are explicit because `math-verify` can be slow on malformed input — a reward that takes 60s per rollout will dominate any RL run.

## Move to a CLI Agent

`opencode-harbor` runs a real coding agent inside a sandbox. Each task comes from Harbor: an instruction, files or setup scripts, a Docker image, and tests that determine reward.

Run a small eval:

```bash
prime eval run prime/opencode-harbor -m openai/gpt-5.4-mini
```

Or run with a config file:

```toml
# [configs/10/opencode-harbor.toml](../../configs/10/opencode-harbor.toml)
model = "openai/gpt-5.4-mini"
save_results = true

[[eval]]
env_id = "prime/opencode-harbor"

[eval.taskset]
task_names = ["regex-log"]

[eval.harness]
max_turns = 4

[eval.harness.program]
disabled_tools = ["webfetch", "question"]
```

```bash
prime eval run configs/10/opencode-harbor.toml
```

A broader baseline run without `task_names` (5 examples × 3 rollouts, the env's defaults from `pyproject.toml`) cost roughly **$3.04** end-to-end against `gpt-5.4-mini`. Expect a longer eval than the text-only examples.

The reward comes from the task tests, not from judging the final message. That makes coding-agent environments useful for training, but it also means broken tests, missing dependencies, or unrealistic timeouts can dominate results.

### How Opencode-Harbor Is Built

[environments/opencode_harbor/opencode_harbor.py](../../environments/opencode_harbor/opencode_harbor.py) is twenty-two lines because both the taskset and the harness already exist as reusable, third-party packages — the env just composes them:

```python
import verifiers as vf
from harnesses import OpenCode, OpenCodeConfig
from tasksets import HarborTaskset, HarborTasksetConfig


class OpenCodeHarborTasksetConfig(HarborTasksetConfig):
    bundle_package: str | None = __name__


def load_taskset(config: OpenCodeHarborTasksetConfig) -> HarborTaskset:
    return HarborTaskset(config=config)


def load_harness(config: OpenCodeConfig) -> OpenCode:
    return OpenCode(config=config)


def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(
        taskset=vf.load_taskset(config=config.taskset),
        harness=vf.load_harness(config=config.harness),
    )
```

Three things to copy when adapting another reusable harness/taskset pair:

- **A typed child loader is the config contract.** `load_taskset(config: OpenCodeHarborTasksetConfig)` tells the framework which Pydantic class to coerce `config.taskset` into; nothing else needs to know about the type.
- **`load_harness` exists only because the package owns reusable harness behavior.** Wordle and reverse-text don't define `load_harness` — they use the framework default. Opencode-harbor does, because the `OpenCode` harness is the agent runtime.
- **Override only when you must.** `OpenCodeHarborTasksetConfig` exists for one reason: to set `bundle_package = __name__` so Harbor can find this env's bundled assets. Every other field rides through from `HarborTasksetConfig` defaults. There is no inline `vf.Taskset(...)` construction, no subclass narrowing of `vf.Env`, no extra root-loader arguments — the loader trio is unembellished.

The same `[eval.taskset]` / `[eval.harness]` / `[eval.harness.program]` split shown in the config above mirrors the type tree: taskset config fields go to Harbor, harness config fields go to OpenCode, program config fields go to the OpenCode CLI invocation. The split is the rule for any third-party-harness env: task data and difficulty on the taskset, rollout execution and program adapters on the harness, command flags under `harness.program`.

## When to Reach for Each Pattern

Start with the smallest sandbox that proves the scoring loop. `math-python`-shape envs — one tool, one sandbox image, deterministic reward — are the right baseline for any sandbox-backed task because every failure is obviously local to the toolset. `opencode-harbor`-shape envs — full agent runtime inside the sandbox, reward driven by external tests — only earn the extra moving parts when the task genuinely requires an agent the framework default harness can't drive.

When the baseline eval runs cleanly, train against the same environment ID as in [Training with RL](../03-training-with-rl/README.md).

## Next

In [Synthetic Agent Environments](../11-synthetic-agent-environments/README.md), you will simulate a small world in memory and have an agent interact with it through tools.
