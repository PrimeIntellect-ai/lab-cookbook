# Environment Best Practices Log (WIP)

When constructing an environment, your goal is not simply to get it working, but to build it in a way which is idiomatic and maintainable. No shortcuts, patches, or hacks. Ensure you understand the underlying framework and best practices before making any key implementation decisions.

- Always ensure you have consulted the Verifiers documentation and best practices.
- Do not make assumptions about how things work internally, or which features are available (or not).
- Types should be incredibly strict and follow the typing guide from Verifiers. Union-types to assuage type checkers are not allowed; this is a signal that you are not using the provided types correctly. Only use in narrow cases like `str | None` for config defaults.
- Prefer a single golden path for everything.
- No loose types (e.g. `Any`, `object`, `Mapping`, etc). Use precise types instead.
- Asserts are OK, and are the preferred path for massaging type-checkers especially if the culprit is a third-party library.
- Absolutely no fallback safety logic or defensive programming. Prefer loud failure over degraded performance.
- Avoid excess helper functions and private methods or variables. This is usually a sign that you are not using the framework correctly.
- Avoid reaching into the framework internals unless a pattern is clearly documented as recommended for user code.

---

## Intended v1 rules, organized by confidence

These notes synthesize local `~/dev/verifiers` v1 code and docs, plus WIP PR 1429 (`pr-1429`). The goal is to track rules we should converge on, not to freeze every detail forever.

### Scope: user environment code vs framework internals

Rules in this document are primarily for **user-authored environment packages**. Framework internals are allowed to use broader unions, `object`, dynamic mappings, registries, import machinery, and private runtime plumbing because they define the boundary. User env code should not copy those internal patterns unless it is building reusable framework infrastructure.

A good review question is: “Is this broad/dynamic construct part of the public env contract, or is it just making the local implementation easier?” If it is only local convenience, tighten it.

### High confidence: hard rules already enforced or clearly documented

#### Loader and package shape

- New v1 environment packages should expose `load_environment(config: ...) -> vf.Env`; the loader takes one concrete config argument, not mirrored keyword args. [source: `verifiers/v1/ENVIRONMENT_BEST_PRACTICES.md`, `verifiers/utils/env_utils.py`]
- Envs may intentionally be taskset-only. If an env does not support harness customization, it may bind a concrete taskset config and return `vf.Env(taskset=...)` with the default harness. Do not invent an empty custom harness only to satisfy a pattern unless the target template/version requires package-composable harness loading. [source: `verifiers/v1/env.py`, current docs vs PR 1429]
- Environment-specific fields belong on `TasksetConfig` or `HarnessConfig`, never as extra root fields on `EnvConfig`. Current `EnvConfig.__pydantic_init_subclass__` rejects unsupported root fields. [source: `verifiers/v1/config.py`]
- `EnvConfig.taskset` must be typed as a `vf.TasksetConfig` subclass and `EnvConfig.harness` as a `vf.HarnessConfig` subclass when those fields are explicitly annotated. `None` child configs are invalid for the env envelope. [source: `verifiers/v1/config.py`]
- The typed annotation matters: loader/config annotations determine how TOML/CLI data is parsed before user code runs. Do not treat annotations as cosmetic. [source: `verifiers/utils/env_utils.py`, `verifiers/v1/config.py`]
- Normal v1 env packages should compose `vf.Env(taskset=..., harness=...)`; do not subclass `vf.Env` for ordinary environments. [source: `verifiers/v1/README.md`, `verifiers/v1/env.py`]

#### Taskset vs harness ownership

- Tasksets own task data and task-local behavior: row/eval-row loading, prompts, reference answers, per-task controls, task-owned tools/toolsets, user behavior, rewards, metrics, advantages, stops, cleanup, and task-local objects. [source: `verifiers/v1/README.md`, `verifiers/v1/taskset.py`]
- Harnesses own rollout/execution behavior: programs, command/sandbox execution, model/client defaults for standalone harnesses, rollout limits, trajectory handling, harness toolsets, harness user behavior, lifecycle wiring, and execution-owned metrics/scoring. [source: `verifiers/v1/README.md`, `verifiers/v1/harness.py`]
- Do not split one logical setting across layers. Put a field where the behavior is implemented and where lifecycle ownership lives. [source: `verifiers/v1/ENVIRONMENT_BEST_PRACTICES.md`]
- If a taskset can be paired with many harnesses, keep it free of harness-specific assumptions. If a harness only works for one task protocol, make that protocol explicit in config, validation, or task row requirements rather than implicit globals.

#### Data model and runtime controls

- `Task` is immutable input data; `State` is mutable rollout output data. Keep both JSON-serializable at user-visible boundaries. Runtime handles belong in framework runtime stores/hidden metadata, not ordinary state fields. [source: `verifiers/v1/README.md`, `verifiers/v1/state.py`]
- Task prompts must not include system messages. Use `task["system_prompt"]`, `TasksetConfig.system_prompt`, or `HarnessConfig.system_prompt`; multiple system prompt sources reject by default unless a harness intentionally sets `system_prompt_merge`. [source: `verifiers/v1/README.md`, `verifiers/v1/harness.py`]
- Rewards/metrics should read reference data from `task` and rollout data from `state`; do not assume `task["answer"]` is copied to top-level state. [source: `verifiers/v1/README.md`]
- Task rows may set top-level controls such as `max_turns`, `tools`, `toolsets`, `sandbox`, and `program`. Runtime precedence is explicit `state.runtime` controls over task controls over harness defaults. [source: `verifiers/v1/README.md`, `verifiers/v1/harness.py`]
- User code should write task-local knobs as task fields, component defaults as config fields, and rollout-local overrides as state/runtime controls only through documented framework paths. Do not hand-edit private runtime structures.
- Use task top-level controls for per-example variability, not hidden module globals. Examples: per-row tool visibility, per-row sandbox image/resources, per-row `max_turns`, per-row program option fragments.
- Use config fields for package-level defaults that should be visible in TOML/CLI. Examples: dataset split, max rows, scoring mode, timeout, system prompt, default sandbox, default toolset specs.
- Use state fields for observable rollout outputs and artifacts. Examples: submitted answer, parsed answer, validation logs, task-specific counters, user-visible error summaries. Avoid putting clients, sandboxes, sessions, locks, file handles, or large caches in state.
- Runtime handles should be addressed by stable serializable identifiers in state/task, then resolved by the owning taskset/harness/toolset/runtime when needed.

#### Object and binding control

- Prefer loaders, config refs, owned object factories, and bindings over global refs. User env code should not create global mutable clients, sandboxes, sessions, caches, datasets, or service handles and then close over them from tools/rewards. [source: `verifiers/v1/config.py`, `verifiers/v1/runtime.py`]
- Module-level constants are fine for immutable literals and small static tables. Module-level locks may be acceptable only for true process-wide coordination. Module-level mutable runtime resources should be treated as a smell and require justification.
- If a callable needs a hidden dependency, inject it through bindings instead of exposing it in the model-visible tool schema or closing over a global. Hidden args should be obvious from the binding map.
- Binding roots should reflect ownership: task data from `task.*`, rollout data from `state.*`, runtime/framework handles from `runtime.*`, owner-managed dependencies from `objects.*`, and resolved tools from `tools.*` where supported by the target version.
- `objects.*` should be owner-private dependencies, preferably importable zero-arg loaders or config refs. They should not be preinitialized heavyweight instances smuggled through env loaders.
- If a dependency needs `task` or `state` to be constructed, do not model it as a global object factory. Use a binding source/callable or lifecycle hook that receives the relevant rollout context.
- Tool schemas shown to the model must not include hidden framework/runtime arguments. If the model should not choose a value, bind it.
- Rewards, metrics, stops, updates, cleanups, and tools should declare only the args they logically need; avoid `**kwargs` catchalls in user code unless the handler intentionally accepts an extensible framework surface.
- Setup/update/cleanup handlers should use bound/runtime resources owned by their component; they should not reach into another component's private attributes or private runtime stores.
- Teardown should clean component-owned process resources. Rollout cleanup should clean rollout/group resources. Do not rely on module process exit for cleanup.

#### Plugging in pieces

- Add task-owned behavior by configuring or subclassing `vf.Taskset`: rows/eval rows, task tools/toolsets, task user, task rewards/metrics/stops/advantages, task cleanup, task objects/bindings.
- Add execution behavior by configuring or subclassing `vf.Harness`: program, sandbox, model/client defaults for standalone use, harness tools/toolsets, harness user, rollout limits, trajectory policy, harness metrics, teardown.
- Use decorators (`@vf.reward`, `@vf.metric`, `@vf.stop`, `@vf.cleanup`, `@vf.teardown`) and default handler tuples/lists for lifecycle participation. Prefer these over ad hoc calls from inside `load_environment`.
- Use task rows or taskset methods for dataset transformation. Avoid doing dataset downloads, filtering, or row materialization in `load_environment` unless the framework loader is explicitly responsible for that component.
- Override `eval_rows()` only when eval data differs from training/default rows. Otherwise let it reuse `rows()`.
- Use per-task `tools`/`toolsets` visibility for narrowing action space on individual examples. Do not dynamically mutate global tool lists mid-rollout unless the harness/toolset lifecycle owns that behavior.
- If an environment needs multiple task categories with the same lifecycle/scoring, one taskset with explicit category fields is fine. If categories need different harnesses, lifecycle, or scoring contracts, expose separate typed loaders/components.

#### Types and failure behavior in user env code

- Use `import verifiers as vf` in environment code. Prefer top-level public API over internal `verifiers.v1` imports. [source: `verifiers/v1/ENVIRONMENT_BEST_PRACTICES.md`]
- Use precise Pydantic config models for structured settings. Raw mappings belong at real external/dynamic boundaries: TOML/CLI payloads, task rows, protocol messages, sandbox/program specs, or arbitrary user data fields. [source: `verifiers/v1/ENVIRONMENT_BEST_PRACTICES.md`, `verifiers/v1/types.py`]
- Avoid `Any`, broad `object`, and untyped mappings in user env internals unless arbitrary data is genuinely the public contract. Prefer named boundary aliases/types.
- Union types are acceptable in user config only when they express a real supported input surface, such as `str | None` defaults, literal mode switches, or a deliberately documented “path or inline value” field. They are not acceptable just to silence a type checker or support stale call patterns.
- Prefer loud failure over silent fallbacks. Do not write `config = config or MyConfig()` in v1 loaders; the framework supplies a concrete validated config. [source: `verifiers/utils/env_utils.py`, docs diff in PR 1429]
- `assert` is acceptable for internal invariants and type narrowing when the invariant should always hold; do not use try/except fallbacks to mask unsupported shapes. [source: current v1 docs and framework style]

#### Lifecycle and async behavior

- Lifecycle hooks and rollout code are async-capable; avoid blocking the event loop. Use async libraries or `asyncio.to_thread` for blocking filesystem, subprocess-adjacent, or CPU-ish local work. [source: `verifiers/envs/experimental/opencode_env.py`, `verifiers/serve/server/env_worker.py`]
- Use framework lifecycle decorators/defaults (`@vf.reward`, `@vf.metric`, `@vf.cleanup`, `@vf.teardown`, default handler tuples, config handler lists) instead of manually calling private runtime internals. [source: `verifiers/v1/taskset.py`, `verifiers/v1/harness.py`]
- For stateful external resources, create/cleanup them in the component that owns the lifecycle; group or rollout state should identify handles, not store unserializable clients directly. [source: `verifiers/v1/runtime.py`, `verifiers/v1/state.py`]

#### Validation workflow

- Environments should be installable with `prime env install <env-name>` and expose `load_environment(...) -> vf.Environment`/`vf.Env`. [source: project `AGENTS.md`, Verifiers docs]
- Validate behavior with `prime eval run <env-name> ...`; do not add `--skip-upload` unless explicitly requested. [source: project `AGENTS.md`]
- Keep each lab environment self-contained under `environments/<env_name>/` with its own `pyproject.toml`, implementation, and README. Create new environments with `prime env init`, not by manually scaffolding files. [source: project `AGENTS.md`]

### Medium confidence: intended direction, especially from PR 1429

#### Canonical v1 loader direction

- PR 1429 appears to move the canonical template toward explicit `load_taskset(config: MyTasksetConfig)`, optional explicit `load_harness(config: MyHarnessConfig)` when the env supports harness customization, and `load_environment(config: vf.EnvConfig)` that can call component loaders. [source: PR 1429 diff in `verifiers/scripts/init.py`, `verifiers/v1/README.md`, `docs/byo-harness.md`]
- In that direction, `load_taskset`/`load_harness` signatures determine concrete package component config types, while base `vf.EnvConfig` can act as an unresolved package-loading envelope. This differs from current `main` docs that prefer environment-local `EnvConfig` subclasses binding child config types. [source: PR 1429 diff in `verifiers/utils/env_utils.py`, `verifiers/v1/config.py`]
- PR 1429 adds component/package resolution utilities and default child `id` injection from the env id when using base `vf.EnvConfig`. If this lands, prefer component loaders over environment-local root config subclasses for package-composable envs. [source: PR 1429 `verifiers/v1/utils/component_utils.py`, `verifiers/utils/env_utils.py`]
- Even under PR 1429, taskset/harness configs still own leaf fields; do not move settings back to root env kwargs. [source: PR 1429 docs diffs]

#### Enforcing and surfacing binding/global-ref rules

- Best enforcement should be layered: template guidance first, runtime/config validation second, lint/static checks third. Runtime can validate binding roots and config shapes; static checks can flag module-level mutable constructors and handlers that close over forbidden globals.
- A useful linter rule for user env files: flag module-level calls to known resource constructors (`Client`, sandbox/session factories, dataset loaders, HTTP clients, DB clients, locks except allowlisted process locks) unless the symbol is a function named like `load_*`/`create_*` and referenced from `objects`/config.
- A useful handler inspection rule: for decorated handlers/tools, inspect closure variables and referenced globals; allow immutable constants/functions/classes, reject mutable/resource instances. Surface the fix as “move this behind a loader and bind it via `objects.*`/`bindings`.”
- A useful schema rule: if a tool/reward/setup parameter is not model-supplied and not a public framework arg, require a binding entry. Fail with a message naming the parameter and suggesting valid binding roots.
- A useful docs/template rule: every nontrivial template should show at least one hidden dependency injected through `objects` + `bindings`, so users copy the right pattern instead of globals.

#### Harness class usage

- Current `main` says omit `harness=` when the base endpoint-backed harness is enough. PR 1429 templates include a typed harness config/class and explicit `load_harness` even when empty, likely to make package composition uniform. Treat “always include harness loader” as a package-composition option, not a universal env requirement.
- Do not create thin custom harnesses only to restate `Harness` unless following a package-composable loader shape or reserving a real typed surface. If no harness customization is intended, taskset-only remains a valid env design.

#### Config defaults

- Current `main` docs say use explicit nested config objects (`taskset: MyTasksetConfig = MyTasksetConfig()`) and not `Field(default_factory=...)`. PR 1429 changes base `EnvConfig` to `Field(default_factory=...)`, but examples still use explicit objects in some taskset-first paths. Treat this as unsettled: follow whichever pattern is in the target Verifiers version/template. [source: current `verifiers/v1/ENVIRONMENT_BEST_PRACTICES.md`; PR 1429 `verifiers/v1/config.py`]

### Lower confidence / needs iteration

#### Strict typing absolutism

- “No unions ever” is too strong. Framework code uses unions at real boundaries (`TasksetInput`, `HarnessInput`, config fields that accept import refs or structured specs, optional values). The user-env rule should be: no broad unions to appease type checkers; unions are acceptable only when supporting multiple shapes is a deliberate public contract. [source: `verifiers/v1/env.py`, `verifiers/v1/config.py`]
- “No `object` ever” is too strong. v1 uses `object` at validation and arbitrary-data boundaries. The user-env rule should be: avoid broad `object` in business logic and public helper signatures; use it only for validators/dynamic payloads where unknown input is the point. [source: `verifiers/v1/config.py`, `verifiers/v1/types.py`]
- “No loose mappings ever” is too strong. Task rows, JSON config maps, protocol payloads, and sandbox/program specs are mapping-shaped by design. The rule should be to confine those mappings to named boundaries and convert to typed config/classes as soon as possible. [source: `verifiers/v1/types.py`, `verifiers/v1/taskset.py`]

#### Binding details still to pin down

- The exact valid binding matrix should be made explicit in framework docs and errors: which roots are valid for taskset-owned handlers, harness-owned handlers, tools, toolsets, setups, updates, cleanups, rewards, metrics, stops, and users.
- We should decide whether `objects.*` is allowed only for callable tools or for broader owner-owned handlers, then enforce that in validation. Until then, user code should use the narrowest pattern that works and avoid cross-owner object access.
- We should decide how much closure/global inspection belongs in Verifiers itself versus Lab lint/pre-commit. Runtime validation gives better errors for loaded components; lint catches issues earlier without importing side-effectful env modules.

#### v0 env classes vs v1 API

- `ToolEnv`, `StatefulToolEnv`, and `MCPEnv` remain present and useful for v0-style/stateless/stateful tool environments, but new lab environments should prefer v1 Taskset/Harness unless the user intentionally targets the older API or an integration that has not migrated. [source: `verifiers/envs/tool_env.py`, `verifiers/envs/stateful_tool_env.py`, `verifiers/envs/experimental/mcp_env.py`, project guidance]
- If using old classes, `ToolEnv` is for stateless tools; `StatefulToolEnv` is for per-rollout mutable tool args/state; `MCPEnv` is intended for globally available read-only MCP servers. Do not mix these into v1 code unless there is a clear adapter pattern. [source: class docstrings/constructors]

## Practical review checklist

- Is this user env code, framework code, or reusable framework-like infrastructure? Apply strict user-code rules only to the first category.
- Does the package expose the loader shape expected by the target Verifiers version (`main` vs PR 1429), and does it intentionally support or not support custom harnesses?
- Are all environment-specific fields on the owning taskset/harness config?
- Are task rows and returned state JSON-serializable? Are live handles hidden behind runtime/object mechanisms?
- Are prompts/system prompts represented in the documented fields, with no system messages inside `task["prompt"]`?
- Are rewards/metrics reading `task` for references and `state` for rollout outputs?
- Are hidden args injected through bindings rather than globals or model-visible tool parameters?
- Are module globals limited to immutable constants, classes/functions, and rare justified process-wide controls?
- Are runtime resources created by loaders/lifecycle hooks and cleaned by the owner component?
- Are types precise at internal boundaries, with broad mappings/unions only where they are intentional public input shapes?
- Does the env install with `prime env install` and pass a narrow `prime eval run` smoke test?
