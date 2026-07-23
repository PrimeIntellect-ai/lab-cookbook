# Build Your Own Coding-Agent Harness

A model's score on an agentic benchmark is never just the model's score — it's the model *plus the harness it runs in*: the agent loop, the tools it's given, the way context is managed. Swap the harness and the number moves, sometimes a lot. In verifiers v1 the harness is configurable rather than baked into the benchmark, which means your own scaffold can be a first-class citizen: selectable by id, runnable on any taskset, comparable against Codex or mini-SWE-agent on equal terms.

This recipe does both halves. First you build a complete CLI-agent harness from scratch — a ~60-line bash-loop agent — and run it on a real task. Then you sweep it against the built-in harnesses on a SWE taskset and read the comparison honestly.

**You need:** [Build Your First Environment](../tutorials/5_build_first_environment.md) and [Coding Agent Environments](../tutorials/11_coding_agents.md); Docker running locally. The finished harness ships in this repo at `environments/mini_loop/`.

## What a harness is

A harness owns *how the agent acts*: it installs an agent program into the rollout's runtime, launches it against the task, and gets out of the way. It does **not** own scoring — the taskset does — and it does not call the model provider directly. Every model call flows through the **interception endpoint** the framework hands you, which is what makes token counts, transcripts, and turn caps comparable across wildly different scaffolds.

The contract is three methods and a few capability flags:


| Piece                                                     | Job                                                                    |
| --------------------------------------------------------- | ---------------------------------------------------------------------- |
| `setup(runtime)`                                          | Install the agent into the runtime, once per rollout.                  |
| `launch(ctx, trace, runtime, endpoint, secret, mcp_urls)` | Run the agent program to completion; return its `ProgramResult`.       |
| `SUPPORTS_MCP`, `APPENDS_SYSTEM_PROMPT`, ...              | Advertise what the harness can do, so incompatible tasksets fail fast. |




## The agent program

The agent itself is deliberately minimal — a uv script (`environments/mini_loop/mini_loop/program.py`) with inline dependencies, so the runtime needs nothing preinstalled:

```python
# /// script
# dependencies = ["openai>=1.0"]
# ///
client = OpenAI(base_url=os.environ["OPENAI_BASE_URL"], api_key=os.environ["OPENAI_API_KEY"])
messages = [{"role": "user", "content": args.task + INSTRUCTIONS}]

for _ in range(args.max_steps):
    reply = client.chat.completions.create(model=args.model, messages=messages).choices[0].message.content or ""
    messages.append({"role": "assistant", "content": reply})
    match = BASH_BLOCK.search(reply)
    if match is None:
        break  # DONE, or the model stopped producing commands
    result = subprocess.run(["bash", "-lc", match.group(1)], capture_output=True, text=True, timeout=args.command_timeout)
    messages.append({"role": "user", "content": f"exit={result.returncode}\n{(result.stdout + result.stderr)[-4000:]}"})
```

One completion per step; the model answers with a single ```bash block; the program runs it *inside the task container* and feeds exit code + output back. That's the whole agent — mini-SWE-agent's core idea in its most legible form. Note what it doesn't do: no tool-calling API, no context compaction, no retries. Those absences are measurable, and the comparison below will price them.

## The harness class

The harness (`environments/mini_loop/mini_loop/harness.py`) wires that program into the v1 contract:

```python
class MiniLoopHarnessConfig(vf.HarnessConfig):
    max_steps: int = 20
    command_timeout_seconds: float = 120.0


class MiniLoopHarness(vf.Harness[MiniLoopHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = False
    SUPPORTS_MCP = False

    async def setup(self, runtime: vf.Runtime) -> None:
        await runtime.prepare_uv_script(PROGRAM_SOURCE, self.config.resolved_env)

    async def launch(self, ctx, trace, runtime, endpoint, secret, mcp_urls) -> vf.ProgramResult:
        _, prompt = self.resolve_prompt(trace.task.data)
        program = await runtime.prepare_uv_script(PROGRAM_SOURCE, self.config.resolved_env)
        args = ["--model", ctx.model, "--task", prompt, "--max-steps", str(self.config.max_steps)]
        env = {**self.config.resolved_env, "OPENAI_BASE_URL": endpoint, "OPENAI_API_KEY": secret}
        return await runtime.run_program([*program, *args], env)
```

The load-bearing details:

- `prepare_uv_script` stages the program in the runtime and resolves its inline dependencies there — `setup` warms it, `launch` reuses it. This is how the built-in `mini_swe_agent` harness installs itself too.
- **The endpoint is the model.** The program never sees a provider key; it gets the interception URL and a per-rollout secret. Turn caps, token caps, and trace recording all happen at that boundary, for any harness.
- **Config is the public surface.** `max_steps` and `command_timeout_seconds` become `[harness]` TOML fields and `--harness.`* flags automatically, exactly like taskset config.
- **Fail fast on what you don't support.** The harness raises on MCP toolsets and non-string prompts rather than silently degrading — the same manners as the built-in harnesses.



## Making it selectable

A harness is a package exporting its `Harness` subclass via `__all__`:

```python
from mini_loop.harness import MiniLoopHarness

__all__ = ["MiniLoopHarness"]
```

Any installed package that does this resolves by id — `[harness] id = "mini-loop"` — the same lookup that finds the built-ins (`verifiers.v1.harnesses.<id>`) or a Hub environment. This repo installs `environments/mini_loop/` as a workspace member, so it's already available after `uv sync`.

## Smoke it

Run it on the Harbor `hello-world` task from [Coding Agent Environments](../tutorials/11_coding_agents.md) — same taskset, same scoring, only the harness swapped:

```bash
uv run eval @ configs/recipes/mini-loop-smoke.toml
```

```toml
model = "openai/gpt-5.4-mini"
num_tasks = 1
num_rollouts = 1

[sampling]
max_tokens = 4096

[taskset]
id = "opencode-harbor"
tasks = ["hello-world"]

[harness]
id = "mini-loop"
runtime = { type = "docker" }
```

Open the trace: you'll see the loop verbatim — the model's bash block as an assistant turn, the exit-code feedback as a user turn, and the Harbor verifier's reward computed by the *taskset*, untouched by your harness. That separation is the whole design.

## Now compare it against the field

Your harness runs; the next question is whether it's any *good*. verifiers ships several coding-agent harnesses, resolved by id:


| Harness id       | What it is                                                                          |
| ---------------- | ----------------------------------------------------------------------------------- |
| `codex`          | OpenAI's Codex CLI agent.                                                           |
| `mini_swe_agent` | mini-SWE-agent — a deliberately minimal bash-loop agent, ~100 lines of scaffolding. |
| `mini-loop`      | Yours, from this recipe.                                                            |


(Also available: `terminus_2`, `kimi_code`, `rlm`, and `default`.)

The taskset is `terminal-bench-2-v1`, a Harbor taskset bundled with this cookbook (`environments/terminal_bench_2_v1`): real terminal/SWE tasks, each a container the agent works in plus tests that score the outcome.

For a faithful comparison, vary **one** variable. Everything below stays identical across runs — model, sampling, tasks, turn caps, runtime — and only `harness.id` changes. The shared part lives once in `configs/compare/base.toml`:

```toml
model = "openai/gpt-5.4-mini"
num_tasks = 10          # start small; scale once the sweep works
num_rollouts = 2        # agentic runs are high-variance — 1 rollout lies to you
max_turns = 40

[sampling]
max_tokens = 4096

[taskset]
id = "terminal-bench-2-v1"
use_prime_registry = true   # pull task images from the Prime registry (avoids Docker Hub rate limits)

[harness]
runtime = { type = "docker" }
```

Then the sweep is three one-liners — the config file carries the constants, the flag carries the variable:

```bash
uv run eval @ configs/compare/base.toml --harness.id mini-loop
uv run eval @ configs/compare/base.toml --harness.id mini_swe_agent
uv run eval @ configs/compare/base.toml --harness.id codex
```

Each run lands in its own `outputs/terminal-bench-2-v1--openai--gpt-5.4-mini--<harness>/...` folder with its resolved `config.toml`, so every number has provenance.

Since harnesses are real, evolving software, two details keep the comparison faithful:

- **Pin harness versions.** `[harness] id = "codex"` + `version = "0.116.0"` in the TOML makes the run reproducible next month. (Your own harness's pin is its package version.)
- **Every model call is intercepted.** Whatever the harness, its model traffic flows through the same interception server — which is why token counts and transcripts are comparable across scaffolds at all.



## Reading the result

The mean reward per harness is the headline — but it's the *least* interesting part. Pull up the runs side by side (`prime eval view`) and compare, per harness:

- **Reward** — the solve rate. Note it with its sample size; at `-n 10`, differences under ~15 points are noise.
- **Tokens and turns per rollout** — from the traces' usage fields. Scaffolds differ wildly in cost: a harness that scores 5 points higher while spending 3× the tokens is a different trade-off, not a strict win.
- **Errors vs. legitimate failures** — a rollout that died with a `HarnessError` (install failed, agent crashed) is not evidence about the model. Count these separately before comparing means; a harness with a 20% crash rate has an infrastructure problem, not a capability deficit.

Then read a few transcripts for the same task across harnesses — this is where the comparison gets real. You'll see personality differences immediately: one agent greps and reads surgically, another cats entire files into context; one runs the tests after every edit, another edits blind and hopes. Expect `mini-loop` to lose on tasks that need file editing finesse (heredocs over a proper edit tool) and to hold its own on short investigative tasks — and now you know *which behavior* each reward point is made of.

## Things to try

- **Improve your harness where it lost.** Give the program a real file-edit convention (e.g. a `cat > file << 'EOF'` instruction in `INSTRUCTIONS`), re-run the sweep, and watch which task categories move. Iterating on a scaffold against a fixed taskset is exactly how the serious harnesses got good.
- **Tool ablations:** most built-in harnesses accept a `disabled_tools` list (`[harness] disabled_tools = ["websearch"]` — names are harness-specific). Re-run the winner with a tool removed and see how much of its edge that tool was carrying.
- **Model × harness grid:** repeat the sweep with a second model. Harness rankings are *not* stable across models — small models often do better in minimal scaffolds that don't demand sophisticated tool orchestration. Two models × three harnesses is six runs and one genuinely publishable chart.
- **Your own repo:** the same sweep works on a taskset of *your* tasks — package your repo's issues as a Harbor taskset ([Coding Agent Environments](../tutorials/11_coding_agents.md)) and rank scaffolds where it actually matters to you.
- **Scale honestly:** once the pipeline works at `-n 10`, drop the `-n` cap, run the full taskset, and quote *those* numbers.



## Recap

A harness is a package with two methods and honest capability flags: install in `setup`, run against the interception endpoint in `launch`, never touch scoring. A uv script with inline dependencies makes the agent program portable to any runtime. And once your scaffold resolves by id, the comparison discipline is the same as any experiment: one variable, pinned versions, reward read alongside cost and crash rate, transcripts read before conclusions are drawn.