---
name: review-v1-environments
description: Review v1 Taskset/Harness environments for code-style, typing, and framework-usage issues. Use when asked to audit a Lab v1 environment for defensive-code smells, helper-function bloat, weak typing, registry-routing antipatterns, system-prompt wiring, or pre-publish readiness.
---

# Review v1 Environments

Audit a v1 Taskset/Harness (TH) environment for correctness, framework misuse, and the code-style anti-patterns that make environments brittle. This skill is the standing reference for how a v1 environment should be shaped; the review is the act of comparing a given environment against it.

## The Objects You Are Reviewing

A v1 environment is a small package that exposes `load_environment(config) -> vf.Env`. Everything else hangs off these objects. You cannot review an environment without knowing what each one owns.

- **`vf.Env`** — the adapter that binds one taskset and one harness so eval and training workers can run rollouts. Authors never subclass it; they assemble it in `load_environment`.
- **`vf.Taskset`** — owns *what is being attempted*: task data (`load_tasks`), the system prompt, answers, task metadata, task-defining tools, rewards, metrics, and any per-rollout state those need. Most environment logic lives here.
- **`vf.Harness`** — owns *how it is attempted*: the rollout loop, the program that drives it, model/client defaults, and execution-level system prompt. Most environments use the framework default harness and never define one.
- **`vf.User`** — the simulated counterpart that produces the environment's reply *between model turns*. A rollout alternates: model responds, then the user responds, then the model again, until a stop condition fires. `User.get_response(task, state, messages) -> list[vf.UserMessage]` returns the next user turn; returning `[]` means "no reply", which typically ends the rollout. The base `User` returns `[]` (single-turn). Subclass `vf.User` for games, negotiations, or wrappers like TextArena where the environment talks back. A user lives on **either** the taskset config **or** the harness config, never both — the runtime raises if both define one.
- **`vf.Toolset`** — a frozen dataclass that exposes tools (and optionally a sandbox) to the model. Tools are plain methods; the toolset is the container, not a subclass point.
- **`vf.Task`** — one immutable, serializable unit of work: `prompt`, optional `system_prompt`, `answer`, `info`, and per-row overrides like `max_turns`. Produced by `load_tasks`, frozen for the rollout.
- **`vf.State`** — the mutable, serializable record of one rollout: `prompt`, `completion`, `trajectory`, metrics, reward, plus any author keys set in `@vf.setup` and mutated by tools. Live runtime handles (clients, sandboxes) are reached *through* state helpers, never stored on it.
- **Config classes** — `MyTasksetConfig(vf.TasksetConfig)` and `MyHarnessConfig(vf.HarnessConfig)` are Pydantic models holding every tunable. They must be serializable (they appear in TOML), so no `Path` objects and no live handles. The config is the *only* surface a user tunes from `[eval.taskset]` / `[env.taskset]` / `[eval.harness]` blocks.

The loaders connect them:

```python
def load_taskset(config: MyTasksetConfig) -> MyTaskset:
    return MyTaskset(config=config)


def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(
        taskset=vf.load_taskset(config=config.taskset),
        harness=vf.load_harness(config=config.harness),
    )
```

The framework reads the *annotations* on the child loaders (`MyTasksetConfig`) to coerce the raw `config.taskset` blob into a typed config. That is why `load_environment` is always typed `vf.EnvConfig` and never customized, and why the child loaders must carry concrete annotations.

## Review Workflow

The order is: confirm the code is *good* before you confirm it *runs*. A clean eval on defensive, mistyped code is still a failing review.

1. **Static read for anti-patterns.** Read the module top to bottom and hunt for the smells in the criteria below. The high-salience ones, in rough order of how often they appear:
   - Defensive helpers and branches: a module-level or `@staticmethod` function whose only job is to coerce message content to `str` or walk a `str | list | None` union; `if isinstance(...)` fallbacks for shapes this environment never produces.
   - Type-checker appeasement: `cast(...)`, `# pyright: ignore`, `# type: ignore`, `assert isinstance(...)` on values that came out of `vf`.
   - Loose types declared by the env: a config field typed `str | dict | Path | None` where one concrete type (or a `vf.*` type) belongs.
   - Empty subclasses that exist only to drive a registry (see *Users*).
   - Loader-shape deviations (see *Loaders*).
2. **Confirm typing is clean at the source.** The environment must pass the type checker and linter with no suppressions. If it only passes because of an ignore comment, that is a finding, not a pass.
3. **Then run it.** Standard runs save automatically; do not add `--skip-upload` unless asked.
   ```bash
   prime eval run <env> -m openai/gpt-4.1-mini -n 5
   ```
4. **Trace one rollout end to end.** What does `load_tasks` return? What does `@vf.setup` add to state? What do tools mutate? What do rewards read? Confirm the strictest available type is used at every boundary.

## Typing & Defensive Code

The contract is strict typing with no defensive padding. The environment operates under clear, known assumptions about the types it handles; code that exists only to handle cases the environment will never see is an anti-pattern, not safety.

**No defensive helpers for message content.** Reading text off a message is one line. A text-only environment knows `message.content` is a string, so the read is an explicit transformation at the call site:

```python
text = str(message.content or "")
```

A helper like `def content_text(content): ...` that branches over `str | list[ContentPart] | None` is the smell to remove — it hides the type contract and implies the environment handles multimodal content it never produces. Flag every such helper; the fix is to inline the coercion.

**Asserts only at external boundaries.** `assert isinstance(x, T)` is acceptable when `x` crosses in from outside the framework — a raw `datasets` row, a third-party SDK response, `json.loads` output. It is wrong on values that came out of `vf` (`task["answer"]`, `message.content`, `state["..."]`); those already carry a type, and an assert there signals the author distrusts the framework instead of narrowing correctly. Replace with an explicit transformation.

**Explicit transformation, not `cast`.** The correct way out of a wider type is a real coercion that produces a value of the target type: `str(x or "")`, `int(value)`, `MyModel.model_validate(payload)`. `cast(T, x)` only silences the checker — the runtime value is unchanged and may not actually be a `T`. Flag every `cast` and require a transformation or a genuine narrowing branch.

**No suppressions.** `# pyright: ignore`, `# type: ignore`, and `SkipValidation` hide the real problem. A type error is evidence the code is wrong; the fix is to restructure until the checker accepts it honestly. Flag every suppression.

**Strictest type, earliest.** Use `vf.*` types as soon as a value enters the environment: `vf.Task`, `vf.State`, `vf.Message`, `vf.AssistantMessage`, `vf.UserMessage`, `vf.JsonData`, `vf.SystemPrompt`, `vf.Toolset`, `vf.SandboxConfig`. Do not pass raw dicts where a typed model exists, and do not declare loose unions to defer narrowing.

## Content vs Logic

Before flagging or rewriting anything, classify the artifact: **content** (prompts, fixed instructions, message templates, fixtures) is represented as visible literals; **logic** is represented as functions. The two get different representations, and the most common review miss is letting content get coded as logic.

A prompt is content. The correct shape is a module-level string constant, or a single `.format()` / f-string template whose only interpolations are task data (`{question}`, `{argument}`). The anti-patterns to flag:

- **Prompts built by helper-function composition.** A `def critic_message(...)` that returns a prompt by calling `turn_contract(...)` and splicing fragments. The reader can no longer see what the model receives without chasing a call graph. Flag it; the fix is one literal per prompt.
- **Shared sentences factored out of prompts to deduplicate.** A `turn_contract()` extracted so three prompts don't repeat one sentence. DRY is a code heuristic and does not apply to content — repeat the sentence so each prompt reads whole. Flag the extraction.
- **Near-static strings dressed as functions.** Three almost-identical prompt builders that differ by a word are three constants, not three `def`s.

The discriminator when reviewing: "it repeats" and "it has a varying part" justify a function only for behavior. For content they justify a template with a data placeholder. If a function's entire body is returning an f-string, it is almost certainly content wearing a function's clothes.

## System Prompts

The system prompt has more valid configurations than any other field, which is why it is the most common source of confusion. Trace it deliberately.

**Where a system prompt can come from.** There are two *sides* and a per-task override:

- **Taskset side** — `TasksetConfig.system_prompt`, or a per-row `task["system_prompt"]` that overrides it for that one task. This is task policy: the instructions intrinsic to the task.
- **Harness side** — `HarnessConfig.system_prompt`. This is execution policy: instructions about how the agent should operate, independent of any specific task.

A task-row `system_prompt` wins over the taskset config for that row. Within a side, the resolved text is one or the other, never concatenated.

**What types are allowed.** The field type is `vf.SystemPrompt = PromptInput | SystemPromptConfig | None`:

- a plain `str` — the literal system prompt text;
- a list of system messages — for multi-message system prompts;
- `vf.SystemPromptConfig(messages=[...])` — inline messages as a config object;
- `vf.SystemPromptConfig(path="prompts/system_prompt.txt")` — load the text from a file in the environment package (path resolves relative to the environment module);
- `None` — this side contributes nothing.

`SystemPromptConfig` requires exactly one of `path` or `messages`.

**How the two sides resolve.** `HarnessConfig.system_prompt_strategy` (default `"HT"`) decides how the harness side `H` and taskset side `T` combine:

- `HT` — harness messages, then taskset messages (default).
- `TH` — taskset first, then harness.
- `H` / `T` — use only that side.
- `H_OR_T` / `T_OR_H` — use the first side that is non-empty.
- `REJECT` — error if both sides are set; forces the author to choose.

The author never assembles the final system message. The harness resolves the strategy and injects the result during rollout setup. Reviewing this means checking that task policy is on the taskset side, execution policy is on the harness side, and the strategy matches the intent — not that someone hand-built a system message.

**Computed prompts.** When the prompt is not static, override `load_system_prompt(config)` on the taskset (or harness) and return a `vf.SystemPrompt`. The common shape is "use the config value if set, else fall back to a packaged file":

```python
def load_system_prompt(self, config: MyTasksetConfig) -> vf.SystemPrompt:
    if config.system_prompt is not None:
        return config.system_prompt
    return vf.SystemPromptConfig(path="prompts/system_prompt.txt")
```

Do not read files by hand inside `load_system_prompt`; `SystemPromptConfig(path=...)` does the resolution and the not-found/empty handling.

**Where prompts live in the package, and prompt optimization.** A file-backed prompt (`prompts/system_prompt.txt` shipped in the package) is the form that prompt optimizers can rewrite. GEPA — the prompt-optimization workflow (`prime gepa run`) that searches for a better system prompt against environment reward — writes its result back to that file when run with `save_to_environment = true`. Because `load_system_prompt` reads the file, the optimized prompt becomes the environment default on the next eval with no code change. So: inline `str` defaults are fine for prompts you do not intend to optimize; a packaged `SystemPromptConfig(path=...)` default is the right choice when the prompt is meant to be tuned. Confirm the `[tool.hatch.build] include` ships the `prompts/` directory if the env uses one.

## Users

(See the object definition above for what a user *is*.) Review focuses on three things.

**Placement.** The user is configured on exactly one side — `TasksetConfig.user` or `HarnessConfig.user`. A user that is intrinsic to the task (a game opponent, the feedback channel of a wrapped benchmark) belongs on the taskset; a user that is part of a reusable execution protocol belongs on the harness. Both sides defining a user is an error the runtime rejects.

**Routing without empty subclasses.** A custom `vf.User` is selected for a config via a registry keyed by config type. The anti-pattern is creating an empty `MyUserConfig(SomeUserConfig): pass` purely so the registry routes back to `MyUser`. Replace it with a typed loader override on the owning taskset/harness:

```python
def load_user(self, config: vf.UserConfig) -> MyUser:
    return MyUser(config=config)
```

The loader override is the explicit routing contract; the empty subclass is load-bearing only as a registry side effect, which is exactly what to avoid.

**`get_response` shape.** The signature is `async def get_response(self, task, state, messages) -> list[vf.UserMessage]`. The parameter for the rendered conversation is `messages`, never `transcript`. Returning `[]` ends or advances the rollout per the harness stop logic. Reading the latest model turn out of `messages` is inline message manipulation — `str(messages[-1].content or "")`, not a helper.

## Loaders

The loader trio is fixed shape. Deviations are findings.

- `load_environment(config: vf.EnvConfig) -> vf.Env` returns exactly `vf.Env(taskset=vf.load_taskset(config=config.taskset), harness=vf.load_harness(config=config.harness))`. No extra root-loader kwargs (`difficulty=...`), no `Optional[EnvConfig]`, no fallback config synthesis, no inline `vf.Env(...)` wiring elsewhere.
- `load_taskset(config: MyTasksetConfig) -> MyTaskset` whenever the package has a custom taskset config.
- `load_harness` only when the package owns reusable execution behavior. Most environments use the framework default and omit it — adding it "for symmetry" is a smell.
- No subclassing `vf.Env` or `vf.EnvConfig`. The child-loader annotations are how the framework types the config; subclassing `EnvConfig` to narrow `taskset` is wrong.

## Tasks & Data

- `load_tasks(split)` returns `vf.Tasks` — a `datasets.Dataset`, an iterable of serializable dicts, or `vf.Task` objects. Prefer returning a `Dataset` whose columns already match the contract (`question`/`answer`); transform rows only to meet it.
- No system messages inside `task["prompt"]` — system prompts are config-owned.
- No framework-managed IDs (`id`, `trajectory_id`) injected into task rows — those belong on state.
- Per-row fields (`max_turns`, `sandbox`, `program`) appear only when they genuinely vary by example. Do not copy a config default into every row.

## Rewards & Metrics

- Reward and metric functions are `@vf.reward` / `@vf.metric` methods on the owner (usually the taskset). They are direct decorated methods, never returned from a factory function.
- They request rollout data by parameter name (`task`, `state`, `answer`, `info`, `prompt`, `completion`). Request exactly what the body reads — an unused `task` parameter is a finding.
- Weights make correctness dominate and judges nudge. A `judge_reward(weight=1.0)` next to `correct_answer(weight=0.2)` is inverted and teaches the model to please the judge.
- Monitor-only signals use `@vf.metric`; `@vf.reward(weight=0.0)` is a code smell. A reward is part of the scoring surface even at zero weight; a metric is observability. Flag every `weight=0.0` reward and convert it to `@vf.metric`.

## Lifecycle Hooks

- `@vf.setup` initializes per-rollout state at the start of a rollout. Per-rollout bookkeeping (budgets, counters, submission slots) belongs here, not lazily inside a tool's first call.
- `@vf.stop(priority=...)` defines stop conditions; higher priority runs first.
- `@vf.cleanup` / `@vf.teardown` release resources. Cleanup must be idempotent — a cancelled rollout can hit it with partial state, so read keys with `.get(...)` defaults.

## Tools

- Task tools are methods on the taskset, returned from `load_toolsets(config)` inside a plain `vf.Toolset`. Do not subclass `vf.Toolset` (frozen dataclass).
- A tool's model-visible schema is its signature minus injected args. `task: vf.Task`, `state: vf.State`, `sandbox`, and `runtime` are injected by parameter name and stripped from the schema; the model never passes them.
- One `Toolset` instance is shared across all concurrent rollouts. Per-rollout data comes from injected `task`/`state`, never from mutable attributes on the toolset or taskset. A process-wide `asyncio.Semaphore` for rate-limiting a shared backend is fine (it is not per-rollout state) and should be created once in `load_toolsets`.

## Sandboxes

- The sandbox contract lives on the `vf.Toolset` that exposes the sandboxed tools. Use an owned `vf.SandboxConfig(...)` for a standalone taskset; `vf.SandboxConfig(prefer="program", ...)` to share a harness/program sandbox when one exists; `sandbox="program"` only when the toolset cannot run without the harness-owned sandbox.
- The framework boots and tears down the sandbox. `setup_commands` / `setup_timeout` on the config are fine; manual `await sandbox.start()` in a tool is not.

## Async Hygiene

- No sync blocking on the rollout hot path (tools, rewards, `env_response`): no `time.sleep`, `requests`, sync `OpenAI`, `deepcopy`/`json` of large payloads. Use `asyncio.sleep`, `httpx.AsyncClient`, `AsyncOpenAI`, or `asyncio.to_thread(...)` for unavoidable sync work.
- Bound concurrency against shared external services with a process-wide semaphore.
- Close resources: per-call clients in a `finally`; long-lived clients in `@vf.teardown`.

## Secrets & Config

- `vf.ensure_keys([...])` at the component that owns the dependency — the reward method for a direct judge call, `load_toolsets` for embedding-backed tools, the harness load step for sandboxed agents.
- Config holds env-var *names*, not secret values (`judge_api_key_var: str = "PRIME_API_KEY"`).
- No `Path` objects in config; path strings round-trip through TOML, `Path` does not.

## Packaging

- `pyproject.toml` pins the minimum `verifiers` version the env actually relies on — not unpinned when v1 APIs are used, not exact-pinned to a dev build.
- `[tool.hatch.build] include` lists exactly what installs (source modules, `prompts/` if used); no stray files, no bytecode.
- `[tool.verifiers.eval]` defaults produce a fast smoke run.
- No tracked bytecode, coverage files, local eval outputs, or `.chroma_db/` blobs; add to `.gitignore` if found.

## Flagging

Report findings severity-first:

1. **`P0` correctness / framework misuse** — empty registry-routing subclasses; broken loader signatures; both sides defining a user; inverted reward weights; sync blocking in tools.
2. **`P1` typing & defensive code** — content-extraction helpers; prompts built from helper-function composition or with shared sentences factored out; asserts on internal `vf` types; `cast(...)`; suppression comments; loose union config fields.
3. **`P2` style & maintainability** — restated config defaults; verbose docstrings; unused parameters; mismatched async patterns.

Each finding gives the file and line, the principle it violates in plain terms, and the exact code that should replace it.

## If No Findings

State that no violations were found, then list residual risk — places where the contract is correct but fragile under future change — so the next iteration knows what to watch.
