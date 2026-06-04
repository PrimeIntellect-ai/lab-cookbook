# The Golden Path: Authoring a v1 Taskset/Harness (TH) Environment

> STATUS: DRAFT / INCOMPLETE. Many sections are stubs and many rules are open
> questions. Nothing here is final.

---

## 0. About This Document

### 0.1 Purpose

This document is the authoritative, step-by-step walkthrough for building a
verifiers **v1 Taskset/Harness (TH)** environment. The goal is that every design
decision an author faces is answered here with an explicit rule and a rationale,
alongside a precise exposition of the behavior the framework provides
automatically.

### 0.2 Status and trust level

- v1 TH is an **alpha pattern under active development**. Breaking changes are
  expected; rules track current behavior and move with it.
- Treat every rule here as provisional until confirmed against current behavior.
  Unless a rule is marked `CONFIRMED`, it is a proposal to confirm, not a settled
  fact.

### 0.3 Markers and conventions

- `RULE n.m` — a numbered, atomic rule. Each carries a status:
  - `CONFIRMED` — verified against current code/behavior.
  - `IMPLICIT` — implied by current code or docs, but not directly verified.
  - `PROPOSED` — a convention recommended by this guide, possibly not yet
    enforced or supported by the framework.
- `OPEN QUESTION` — an unresolved design decision, including whether the
  framework itself should change.
- `AUTOMATIC` — framework behavior applied on the author's behalf, not written
  in environment code, that the author must understand.
- `STUB` — placeholder; content not yet written.

### 0.4 Core objects

- `Taskset` — what is being attempted: the data, the tools that define the
  action space, success conditions, and rewards.
- `Harness` — how it is attempted: rollout execution, model and client
  defaults, agent protocols.
- `Env` — the adapter that binds one taskset and one harness for the eval
  and training workers.

> OPEN QUESTION 0.a: Is "TH" the name we want to standardize on in the cookbook,
> or do we prefer "v1 environment" / "Taskset/Harness environment"? Pick one and
> use it everywhere.

---

## 1. Mental Model

`RULE 1.1 IMPLICIT` — v1 is data-first.
- `Task` is immutable, serializable input data.
- `State` is mutable, serializable rollout output.
- Live runtime handles (model clients, sandboxes, MCP sessions, tool backends)
  are process-local and reached through `state` helpers while a rollout is
  active. They are never stored on task data or config, and never serialized.

`RULE 1.2 IMPLICIT` — Ownership split.
- Tasksets own the domain: task data and loading, prompts, answers, task
  metadata, task-defining tools, users, and task-specific lifecycle, metrics,
  and rewards.
- Harnesses own execution: the rollout loop, programs, model and client
  defaults, endpoint interception, primary sandbox placement, framework
  adapters.

`RULE 1.3 IMPLICIT` — The deciding test for where something goes. If a tool
defines the task's action space, observations, or success condition, it belongs
to the taskset. If code only describes how an arbitrary task is attempted, it
belongs to the harness.

`RULE 1.4 PROPOSED` — Defensive code is not allowed. The environment operates
under clear assumptions about what types it works with. Code that exists only to
handle hypothetical cases the env will never actually see — multimodal content
in a text-only task, unexpected `None`s in fields the framework guarantees,
fallback branches for "what if" — is forbidden. If you find yourself reaching
for it, the type is wrong upstream and the fix is to narrow the type, not to
spread guards downstream.

`RULE 1.5 PROPOSED` — Strict typing. Use the strictest type appropriate at the
boundary. Do not abuse custom types or union types to defer narrowing. Adopt
native `verifiers` types (`vf.Task`, `vf.State`, `vf.Message`,
`vf.AssistantMessage`, `vf.UserMessage`, `vf.TextContentPart`, `vf.JsonData`,
`vf.SystemPrompt`, `vf.Toolset`, `vf.SandboxConfig`, ...) as early in the chain
as possible. Pass typed objects through, not raw dicts.

`RULE 1.6 PROPOSED` — Backdoors out of a wider type are, in order of preference:

1. **Explicit transformation**, in-line. `str(x or "")`, `int(value)`,
   `dict(payload)`, `vf.UserMessage(content=text)`. The transformation itself is
   the narrowing; the result type is concrete.
2. **`assert isinstance(x, T)`**, in-line — only allowed when `x` originates
   outside the framework (raw `datasets` rows, third-party SDK responses,
   `json.loads` output). Asserts on values that came out of `vf` are forbidden —
   if `vf` returns a wider type than you need, the assumption should be
   expressed by the caller through transformation or by tightening the
   framework type.

`cast(T, x)` is almost always wrong. It silences the type checker without
producing a real value of `T`; the runtime can still hold something else. Reach
for it only when there is no transformation that preserves identity and the
type checker is provably wrong about a specific call site.

`RULE 1.7 PROPOSED` — Linter errors are evidence the code is wrong, not noise
to suppress. Do not add `# pyright: ignore`, `# type: ignore`, or equivalent
suppressions. If the type checker rejects what you wrote, restructure until it
accepts (typically by following 1.4–1.6).

`RULE 1.8 PROPOSED` — Basic message manipulation lives in-line at the call
site. Reading text off `message.content`, filtering `vf.get_messages(...)` by
role, extracting tagged spans with `re.search` — these are one or two lines and
should sit inside the reward or user method that needs them. Wrapping them in a
module-level helper or a `@staticmethod` "for reuse" is a code smell unless the
same logic is genuinely shared across owners that cannot reach it through the
class hierarchy. The cost of a small amount of repetition is lower than the
cost of a vague helper that hides the type contract.

`RULE 1.9 PROPOSED` — Classify the artifact before choosing a representation.
**Content** (prompts, fixed instructions, message templates, fixtures) is
represented as visible literals: a module-level string constant, or a single
`.format()` / f-string template whose only interpolations are *task data*
(`{question}`, `{argument}`). **Logic / behavior** is represented as functions.

- Do not build a prompt by calling helper functions that return prompt
  fragments. A reader must see exactly what the model receives by reading one
  literal, without chasing a call graph.
- Do not factor a shared sentence out of two prompts to deduplicate it. Repeat
  the sentence. For content, legibility outranks deduplication; DRY is a code
  heuristic and does not apply.
- "It repeats" and "it has a varying part" justify a function only for
  behavior. For content they justify a template with a data placeholder, not a
  builder function.

The failure this prevents: pattern-matching on surface structure (repeated text
+ a varying slot) and reaching for `def`, instead of first asking whether the
thing is content or logic. Ask that question first; it selects the
representation.

> OPEN QUESTION 1.a: Where is the boundary for tools that are execution
> mechanics but task-flavored (e.g. a `submit_answer` tool)? `submit_window` in
> calendar-scheduling defines the success condition and is currently treated as a
> taskset tool. Confirm this matches RULE 1.3.

> AUTOMATIC 1.b (STUB): Task -> Harness -> State data flow diagram, with each arrow
> described.

---

## 2. Repository & Package Layout

`RULE 2.1 PROPOSED` — Each environment is a self-contained package under
`environments/<env_name>/` containing at minimum the implementation module,
`pyproject.toml`, and `README.md`.

`RULE 2.2 IMPLICIT` — Create environments with `prime env init <env-name>` rather
than hand-rolling the scaffold.

> OPEN QUESTION 2.a: What is the rule for splitting an environment across
> multiple modules? calendar-scheduling uses `calendar_scheduling.py`,
> `calendar_problem.py`, and a separate TUI. Reconcile with the rule against
> detached helper functions at the bottom of an environment file (11.c): does it
> push logic into sibling modules, or only bar single-use clutter?

> STUB 2.b: `pyproject.toml` contract for v1 (build backend, included files,
> `[tool.verifiers.eval]` defaults, dependency pinning, `verifiers` min version).

---

## 3. The Loaders (Golden Loader Shape)

`RULE 3.1 IMPLICIT` — Every environment module exposes exactly one root loader
with this exact signature, returning a `vf.Env`:

```python
def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(
        taskset=vf.load_taskset(config=config.taskset),
        harness=vf.load_harness(config=config.harness),
    )
```

`RULE 3.2 IMPLICIT` — `load_environment` is typed as `vf.EnvConfig` and is not
customized. The config surface is expressed entirely through the child loaders'
annotations.

`RULE 3.3 IMPLICIT` — When the environment has a custom taskset config, expose a
typed child loader. The annotation is the config contract:

```python
def load_taskset(config: MyTasksetConfig) -> MyTaskset:
    return MyTaskset(config=config)
```

`RULE 3.4 IMPLICIT` — Expose `load_harness(config: MyHarnessConfig)` only when
the package owns reusable execution behavior. Otherwise use the framework's
`vf.load_harness(config=config.harness)` with the base harness.

`RULE 3.5 IMPLICIT` — Prohibited in loaders:
- accepting root-loader kwargs for taskset/harness fields (no
  `load_environment(difficulty=...)`-style args),
- subclassing `vf.Env`,
- subclassing `vf.EnvConfig` to narrow child types,
- synthesizing fallback configs, accepting `None`, or mutating configs.

> AUTOMATIC 3.a IMPLICIT: The framework reads the child loader parameter
> annotations (`MyTasksetConfig`, `MyHarnessConfig`) and coerces the raw
> `config.taskset` and `config.harness` blobs into typed config objects. The
> child loaders must therefore be annotated, and `EnvConfig` must not be
> subclassed.

> OPEN QUESTION 3.b: What is the exact resolution order when both a package
> `load_taskset` and a packaged/`id`-selected taskset exist? (TOML can select
> `[eval.taskset].id = "tasksets.harbor"`.) Document precedence.

---

## 4. Config Objects

`RULE 4.1 IMPLICIT` — Structured settings use Pydantic config classes:
`MyTasksetConfig(vf.TasksetConfig)` and `MyHarnessConfig(vf.HarnessConfig)`.

`RULE 4.2 IMPLICIT` — Config values must be serializable and stable enough to
appear in TOML. Pass callables as import-ref strings, not function objects.
Do not pass `Path` objects.

`RULE 4.3 IMPLICIT` — Put task-policy fields on the taskset config; put
execution-policy fields on the harness config.

`RULE 4.4 IMPLICIT` — Do not add a generic `split` or `dataset_split` config
field that duplicates `load_tasks(split=...)`. Use named fields like
`train_split` and `eval_split` only to map v1 split names to upstream source
split names.

`RULE 4.5 IMPLICIT` — `Taskset.__init__`, `Harness.__init__`, and `User.__init__`
are final. Customize via config, public load methods, lifecycle decorators, and
program config. Do not override `__init__`.

`RULE 4.6 PROPOSED` — No empty config subclasses for registry routing. If your
env needs a custom `vf.User` subclass paired with a reusable taskset (e.g.
`TextArenaUser` / `TextArenaUserConfig`), do *not* create
`MyUserConfig(TextArenaUserConfig): pass` solely so the user-type registry maps
back to your subclass. Override the typed loader instead — `load_user`,
`load_taskset`, `load_harness`, `load_toolsets`, `load_objects`,
`load_artifacts` — and instantiate the desired class directly:

```python
class MyTaskset(TextArenaTaskset[MyTasksetConfig]):
    def load_user(self, config: vf.UserConfig) -> MyUser:
        return MyUser(config=config)
```

The loader hook is the routing contract. Empty config subclasses are
load-bearing only because some other system is being routed through
registration metadata; replacing them with an explicit `load_*` override moves
the routing into typed code where it can be reviewed.

> OPEN QUESTION 4.a: `Taskset`, `Harness`, `User`, and `Toolset` all have `@final`
> constructors (`vf.Toolset` is additionally a frozen dataclass; see 11.5). Is
> final `__init__` across these owner types the intended permanent design, or a
> current implementation detail? This underlies the Toolset questions in Section
> 11 (see 11.a).

> STUB 4.b: Enumerate the base fields on `vf.TasksetConfig` and
> `vf.HarnessConfig` (system_prompt, user, toolsets, objects, bindings,
> artifacts, lifecycle lists, scoring, max_turns, program, ...) with types and
> defaults. Mark which are IMPLICIT vs CONFIRMED against source.

---

## 5. Tasks & Datasets

`RULE 5.1 IMPLICIT` — Tasksets load both train and eval data through a single
method:

```python
class MyTaskset(vf.Taskset[MyTasksetConfig]):
    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        ...
```

`RULE 5.2 IMPLICIT` — `vf.Tasks` may be a `datasets.Dataset`, an iterable of
serializable records, or an iterable of `vf.Task` objects. `get_dataset()` calls
`load_tasks("train")`; `get_eval_dataset()` calls `load_tasks("eval")`.

`RULE 5.3 IMPLICIT` — Return a `datasets.Dataset` directly when the source
columns already match the task contract (e.g. `question`, `answer`). Transform
rows only to meet the contract, add reference fields, build multimodal content,
or attach per-example state the rollout actually uses.

`RULE 5.4 IMPLICIT` — Task records are JSON-serializable and become immutable
`vf.Task` during rollout. The recognized top-level fields are `prompt`,
`system_prompt`, `answer`, `info`, `max_turns`, `toolsets`, `tools`, `sandbox`,
`program`, and `artifacts`.

`RULE 5.5 IMPLICIT` — `task["prompt"]` must not contain system messages. System
prompts are resolved separately; see Section 6.

`RULE 5.6 IMPLICIT` — Do not copy config defaults into every task row. Use
`max_turns`, `sandbox`, `program`, and tool-visibility fields per-row only when
they genuinely vary by example. Do not inject framework-managed IDs into rows.

> AUTOMATIC 5.a IMPLICIT: If a row provides `question` (string) instead of `prompt`,
> the framework derives the user `prompt` from it. Confirm exact behavior and
> whether `system_prompt` interacts here.

> OPEN QUESTION 5.b: For procedurally generated tasksets (calendar-scheduling
> generates 512 train / 128 eval via seeds), the data is produced in
> `load_tasks`, not loaded from disk. Is there a preferred pattern for
> determinism, caching, and lazy generation, or is eager generation in
> `load_tasks` the golden path? Document it.

---

## 6. System Prompts

`RULE 6.1 IMPLICIT` — System prompts are config-first. Static task policy goes
in `TasksetConfig.system_prompt`. Static execution policy goes in
`HarnessConfig.system_prompt`.

`RULE 6.2 IMPLICIT` — Resolution is per task, computed during
`Harness.setup_state(...)`.
- Taskset side `T` = `task["system_prompt"]` if present, otherwise
  `TasksetConfig.system_prompt`.
- Harness side `H` = `HarnessConfig.system_prompt`.

`RULE 6.3 IMPLICIT` — `HarnessConfig.system_prompt_strategy` combines the two
sides. Strategies: `HT` (default), `TH`, `H_OR_T`, `T_OR_H`, `H`, `T`, `REJECT`.

`RULE 6.4 IMPLICIT` — File-backed prompts (e.g. for GEPA optimization) use
`vf.SystemPromptConfig(path="system_prompt.txt")`. Override
`load_system_prompt(config)` only when prompt construction is computed.

`RULE 6.5 PROPOSED` — Prompts are content (see RULE 1.9): a module-level string
constant or a `.format()` template with data placeholders, or a packaged
`prompts/*.txt` file. Never a prompt built by composing helper functions, and
never a shared sentence factored out of multiple prompts to deduplicate it.
When two prompts share wording, repeat it so each prompt reads as one literal.

> AUTOMATIC 6.a IMPLICIT: The author never assembles the final system message; the
> harness injects it into the message list during `setup_state`. The author only
> supplies the two sides + strategy.

> OPEN QUESTION 6.b: The default strategy is `HT` (harness then taskset), but the
> taskset usually owns the substantive instructions. Should the cookbook default
> be taskset-first (`TH`) instead of `HT`?

---

## 7. Rewards, Metrics, Advantages, Scoring

`RULE 7.1 IMPLICIT` — Reward and metric functions are methods on the owner
class (usually the taskset), declared with decorators:

```python
@vf.reward(weight=1.0)
async def exact(self, task: vf.Task, state: vf.State) -> float:
    ...

@vf.metric
async def submission_valid(self, state: vf.State) -> float:
    ...
```

`RULE 7.2 IMPLICIT` — Reward and metric functions request rollout data by
parameter name (`task`, `state`, `answer`, and so on). The framework injects
what they ask for. Declare exactly the parameters the body reads.

`RULE 7.3 PROPOSED` — Monitor-only signals use `@vf.metric`, not
`@vf.reward(weight=0.0)`. A reward — even a zero-weighted one — is part of the
scoring surface; a metric is observability. Use `@vf.reward(weight=...)` only
for signals that contribute to the final score, and `@vf.metric` for everything
tracked for diagnosis (submission validity, oracle ratios, budget usage, parse
success). `weight=0.0` rewards in new environments are a code smell.

> STUB 7.a: Exact set of injectable parameter names for rewards/metrics, group
> vs single semantics, and how `weight` combines into final reward.

> STUB 7.b: `advantages` and `scoring` config — what they are, when to use them.

---

## 8. Lifecycle Hooks

`RULE 8.1 IMPLICIT` — Lifecycle behavior is declared with decorators on the owner
class: `@vf.setup`, `@vf.update`, `@vf.stop`, `@vf.cleanup`, `@vf.teardown`.

`RULE 8.2 IMPLICIT` — `@vf.setup` initializes per-rollout `state` fields at the
start of a rollout. `@vf.stop(priority=...)` defines stop conditions checked
after turns (higher priority runs first).

> STUB 8.a: Precise firing order and timing of each hook within a rollout; which
> run per-rollout vs once; idempotency requirements for cleanup/teardown.

> OPEN QUESTION 8.b: When must state mutation be an `@vf.update` handler rather
> than inline in a tool or reward? One principle in play: setup/update handlers
> are for state changes that add no messages. Resolve to a crisp rule.

> OPEN QUESTION 8.c: Can lifecycle hooks live on a `Toolset` (which exposes
> `setups`/`stops`/`updates`/`cleanups`/`teardowns`) as well as on the taskset?
> If both, what is the precedence/merge order, and which is golden? (Connects to
> Section 11.)

---

## 9. State & Runtime Handles

`RULE 9.1 IMPLICIT` — `State` is mutable during rollout and serializable before
return. It holds trajectory, completion, metrics, reward, timing, artifacts,
errors, and environment output.

`RULE 9.2 IMPLICIT` — Reach live resources through state helpers:
`state.get_model()`, `state.get_client(...)`, `state.get_endpoint_config(...)`,
`state.get_max_turns(default)`, `state.get_tools()`, `state.add_tool(name,
tool)`.

> AUTOMATIC 9.a IMPLICIT: Borrowed runtime handles (via `state.for_task(...,
> borrow=...)`) remain owned by the source runtime and are stripped before
> serialization. Document the full list of what gets stripped.

> STUB 9.b: Canonical list of `state` keys the framework manages vs author-owned
> keys; serialization boundary rules; what must never be put on state.

> OPEN QUESTION 9.c: Authors currently read ad hoc keys like
> `state["trajectory"]` and call `state.get_max_turns(10)` directly inside tools
> (see calendar-scheduling). Is reading `state["trajectory"]` part of the
> supported public surface, or should there be a helper? The hardcoded default
> `10` for `get_max_turns` indicates a missing default; where should the real
> default come from?

---

## 10. Harness & Programs

`RULE 10.1 IMPLICIT` — Start with the base harness (endpoint-backed, default
tool loop). Add a custom `vf.Harness` subclass only when the package owns a
reusable execution protocol: command agents, framework adapters, endpoint
interception, primary sandbox placement, or program runners.

`RULE 10.2 IMPLICIT` — `HarnessConfig.program` selects executable behavior:
`vf.ProgramConfig()` (base loop), `ProgramConfig(base=True)`,
`ProgramConfig(fn="pkg:run")`, `ProgramConfig(command=[...])`.

> STUB 10.a: The base rollout loop, step by step (model call -> tool exec ->
> stop checks -> render completion). Where users/tools/updates fire within it.

> STUB 10.b: Custom program signature `async def program(task, state) -> state`
> and what it is responsible for vs what the framework still does.

---

## 11. Tools & Toolsets

> Design taskset-first. Reach for tools only once the task contract and rewards
> are clear.

`RULE 11.1 IMPLICIT` — Tools are exposed through `vf.Toolset`. Tasks may only
show or hide toolsets and tools, not define them inline.

```python
class SearchTaskset(vf.Taskset[SearchTasksetConfig]):
    def load_toolsets(self, config: SearchTasksetConfig) -> vf.Toolsets:
        return {"search": vf.Toolset(tools=[self.search])}

    @staticmethod
    async def search(query: str, task: vf.Task, state: vf.State) -> str:
        ...
```

`RULE 11.2 IMPLICIT` — A tool is any callable. The model-visible schema is
extracted from the function signature and docstring (name, typed parameters,
and the Args section).

`RULE 11.3 CONFIRMED` — `task`, `state`, `runtime`, and (when applicable)
`sandbox` are injected by parameter name and stripped from the model-visible
schema. A tool opts in to these by declaring them as parameters. The model
cannot pass them.

`RULE 11.4 CONFIRMED` — A single `Toolset` instance is shared across all
concurrent rollouts. Per-rollout data must come from the injected `task` and
`state` (or rollout-scoped `objects`), not from mutable attributes on the
toolset or taskset instance.

`RULE 11.5 CONFIRMED` — `vf.Toolset` is a `frozen=True` dataclass with a
`@final __init__`. Treat it as the data/config object that exposes tools, not as
a subclass extension point. Do not subclass `vf.Toolset` in cookbook examples.

`RULE 11.6 CONFIRMED` — Put task-specific tools directly on the `Taskset` class
by default. Use static methods when the tool only needs injected `task` and
`state`; use instance methods when the tool needs taskset-owned objects. Attach
those methods to a plain `vf.Toolset` from `load_toolsets`.

`RULE 11.7 CONFIRMED` — Closure or module-level tool callables are acceptable
when the tool assembly is local and clearer that way, but they are not a
separate toolset abstraction. They are still ordinary callables attached to a
plain `vf.Toolset`.

> STUB 11.d: `Toolset` capabilities beyond `tools=`: `show`/`hide`, `bindings`,
> `objects`, `artifacts`, `write`, `scope` (`rollout`/`group`/`global`),
> `sandbox`, and lifecycle lists. When each is needed; worked examples.

> STUB 11.e: MCP tools (`vf.MCPTool`, MCP config dicts) inside a toolset.

> AUTOMATIC 11.f CONFIRMED: Hidden-arg set is `{"runtime", "task", "state"}` plus
> `"sandbox"` when the owner declares a sandbox, plus any binding-provided arg
> names. The model schema is the function signature minus those. Document the
> binding-arg case with an example.

---

## 12. Users (Simulators)

`RULE 12.1 IMPLICIT` — User simulators subclass `vf.User` and implement
`get_response(...)`; configure them with `UserConfig` subclasses. Do not pass
callable users.

`RULE 12.2 IMPLICIT` — Use **users** for environment replies between model turns,
**tools** for schema actions, and **setup/update handlers** for state changes
that add no messages.

> STUB 12.a: `User.get_response` signature, injection surface, multi-turn
> negotiation example.

---

## 13. Sandboxes

`RULE 13.1 IMPLICIT` — Sandboxed tools are normal tools: put them in a
`vf.Toolset`, pass the sandbox handle via bindings/state helpers, keep task rows
serializable.

`RULE 13.2 IMPLICIT` — Keep the sandbox contract on the toolset that exposes the
tools. Standalone tasksets use an owned `vf.SandboxConfig(...)`; share a
program/CLI sandbox via `vf.SandboxConfig(prefer="program", ...)`; use
`sandbox="program"` only when the toolset cannot run without the harness-owned
program sandbox.

> STUB 13.a: Sandbox lifecycle, scoping keys, timeouts, and the math_python /
> swe_grep patterns as worked examples.

---

## 14. Validation Workflow

`RULE 14.1 PROPOSED` — An environment is not "done" until it has been
**installed, loaded, and evaluated** — not merely imported.

`RULE 14.2 IMPLICIT` — Canonical loop: `prime env install <env>` ->
`prime eval run <env> ...` (results upload by default; do not add
`--skip-upload` unless explicitly requested).

> STUB 14.a: Minimal smoke-eval command, expected artifacts, how to read
> `prime eval tui`, and a pre-publish checklist.

---

## 15. TOML & CLI Config Surface

`RULE 15.1 IMPLICIT` — Run settings (eval/training) are owned by the eval/training
config; environment behavior is owned by the v1 child config under
`[eval.taskset]` and `[eval.harness]`.

```toml
[[eval]]
env_id = "my-v1-env"

[eval.taskset]
system_prompt = "Answer exactly."

[eval.harness]
max_turns = 4
```

`RULE 15.2 IMPLICIT` — CLI overrides target typed child fields, e.g.
`--taskset.system-prompt "..."` and `--harness.max-turns 4`.

> STUB 15.a: Package composition via `[eval.taskset].id` / `[eval.harness].id`;
> precedence vs local loaders (see OPEN QUESTION 3.b).

---

## Appendix A: Index of Open Questions

- 0.a — Naming: "TH" vs "v1 environment".
- 1.a — Where task-flavored execution tools (e.g. `submit_window`) belong.
- 2.a — Multi-module splitting rules.
- 3.b — Local loader vs packaged `id` precedence.
- 4.a — Is `final __init__` permanent design across owners?
- 5.b — Golden pattern for procedurally generated tasksets.
- 6.b — Default system-prompt strategy for the cookbook (`HT` vs `TH`).
- 8.b — `@vf.update` vs inline state mutation.
- 8.c — Lifecycle hooks on toolset vs taskset; precedence.
- 9.c — Public surface for `state["trajectory"]` / `get_max_turns` default.
- 11.a — Should `vf.Toolset` be `final`?
- 11.b — Should tool methods be taskset methods with `self`?
- 11.c — Anti-clutter rule vs module-level tool functions.
