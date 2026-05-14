# AGENTS.md

<!-- Generated for lab workspaces. -->

This AGENTS guide is intended for end users working in a `prime lab setup` workspace.

## Shared Best Practices (All Contexts)

These points are direct restatements of Verifiers docs so agents can follow the same golden-path workflows.

- Environments are expected to expose `load_environment(...) -> vf.Environment` and be installable with `prime env install <env-name>`. (See `docs/overview.md` and `docs/environments.md`.)
- Validate environment behavior with `prime eval run <env-name> ...` before sharing/publishing changes. Treat `prime eval run` as the canonical eval path: it saves results automatically, and agents should not add opt-out flags such as `--skip-upload` unless the user explicitly requests that deviation so runs stay visible in the private Evaluations tab and in `prime eval tui`. (See `docs/overview.md` and `docs/development.md`.)
- Use `ToolEnv`/`MCPEnv` for stateless tools and `StatefulToolEnv` when per-rollout state must persist (sandbox/session/db handles). (See `docs/environments.md`.)
- If external API keys are required, validate them in `load_environment()` with `vf.ensure_keys(...)` so failures are explicit and early. (See `docs/environments.md`.)

## Code Style and Conventions

Environments are the user-facing surface of Verifiers; their code must be minimal, elegant, and idiomatic. The framework absorbs plumbing — envs declare *intent*, never *mechanism*. Read this section before writing or editing any environment code in this workspace.

### Verifiers-pure, not stdlib-pure

The framework provides idioms for everything an environment should need — taskset construction, harness wiring, config typing, reward registration, state typing. Reach for `vf.*` before `os`, `pathlib`, `importlib.resources`, or any stdlib utility that nudges environments toward path manipulation, frame introspection, or other plumbing. "It's stdlib" is not a justification — the question is whether it's *idiomatic verifiers*, and the answer for path/IO/discovery utilities in env code is almost always no. If a primitive forces you to reach for stdlib to wire something into it, the framework is missing an idiom — file a verifiers issue, don't bandage it in env code.

### The gold standard

Most environments are a few dozen lines: imports, optional constants, optional `vf.*Config` subclass, one `load_environment` function that wires components. The user-facing surface must be incredibly minimal and elegant — golfy but intuitive. The canonical shape is one expression:

```python
import verifiers as vf

def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(taskset=vf.SomeTaskset(config=config.taskset),
                  harness=vf.SomeHarness(config=config.harness))
```

Anything beyond that needs a reason. If the reason is genuine, the additional code lives in dedicated, well-organized util modules under the env directory — never as free-floating helpers in the env file.

### Global rules

- **No path manipulation in env code.** No `Path(__file__).parent / "..."`, no `os.path.dirname(...)`, no `importlib.resources.files(__package__)`. If a primitive needs to know where the env's data lives, it figures that out itself.
- **No frame introspection in env code.** `sys._getframe`, `inspect.stack`, explicit `__package__` / `__file__` references — framework-side concerns only.
- **No `from __future__ import annotations`.** Real, eagerly-evaluated typing. Quote forward refs only where strictly necessary.
- **Strict types.** Type `state` as `vf.State`, configs as the concrete `vf.*Config` subtype, taskset/harness/env as their concrete classes. No `Any`, no bare `object`, no `cast(...)`. Exceptions get a one-line comment explaining why a stricter type is impossible.
- **Pydantic is always OK and preferred.** Prefer Pydantic over `dataclasses`, `TypedDict`, `NamedTuple`, or hand-rolled `__init__`. Field validators enforce invariants; field types enforce shape.
- **TOML scope is `[tool.verifiers.eval]` only.** Eval defaults (`num_examples`, `rollouts_per_example`) live there. Do *not* invent other `[tool.verifiers.*]` sections — the framework does not load them. Per-run knobs like `max_turns` and sampling args come from runner TOMLs (eval / RL configs) via the harness config the runner passes in; the env never reads pyproject for runtime tuning.
- **Custom args go through a `vf.*Config` subclass.** When an env genuinely needs to expose a new tunable or change behaviour beyond what the parent harness/taskset already offers, add a field to a `vf.HarnessConfig` / `vf.TasksetConfig` subclass — never as a positional or keyword arg on `load_environment` and never via an ad-hoc pyproject section. Don't subclass just to override an inherited default: parent defaults are already the right answer, and subclass overrides of inherited defaults don't round-trip through `from_config()`.
- **No global helper functions in env files.** If a helper earns its keep, it goes in a util module under the env directory. Otherwise inline it. "I might reuse this later" is not a justification.
- **Rare exception:** a process-level handle (an `asyncio.Lock`, a `Semaphore`, a connection pool) where module-global state is the only way to assert the desired control. Document with a one-line comment explaining why module scope is required.
- **No-op wrappers don't belong.** If a function does nothing but shuffle the same config fields back into a constructor, delete it and call the constructor directly. Anti-pattern: `def load_taskset(config): return vf.Taskset(config=Config.from_config(config))`.

### Anti-patterns — none of these belong in an env file

- `TASKS_DIR = Path(__file__).parent / "tasks"` — env is plumbing where its own data lives. The framework should resolve this internally.
- `from importlib.resources import files; ... files(__package__) / "tasks"` — same smell with a stdlib wrapper. Still path discovery in user code.
- `def helper(...)` at module scope inside an env file — extract to a util module if it earns its keep, otherwise inline.
- `parser = Parser()` followed by one `parser.parse_answer(...)` call — `Parser` exists for format-checking rewards and custom extraction, not one-shot string grabs.

### v1 specifics (taskset / harness composition)

These rules apply to environments using the v1 `vf.Env(taskset=..., harness=...)` composition pattern (see `environments/AGENTS.md` and verifiers' BYO Harness docs).

- **Loader configs are non-Optional.** Write `config: vf.EnvConfig`, not `config: vf.EnvConfig | None = None`. The framework enforces upstream — don't repeat the check.
- **Subclass defaults don't round-trip.** Overriding a parent field's default in a `vf.*Config` subclass is silently dropped on `from_config()`. Pass policy values as constructor kwargs. Never override inherited defaults via subclass fields.
- **Strict v1 signatures.** `config: vf.EnvConfig`, `state: vf.State`, taskset/harness return types as their concrete classes. No `Any`, no bare `object`, no `cast(...)`.
- **No-op `load_taskset` / `load_harness` wrappers don't belong.** If they only call `Config.from_config(...)` and re-pass the same fields to a constructor, delete them and let `load_environment` call the constructor directly.

### Style bugs to push back on

When considering an approach that adds stdlib imports, path or package discovery, frame introspection, or any logic to the env file beyond pure wiring, default to rejecting it and ask whether the framework should grow the idiom instead. The framework absorbs complexity; envs do not.

## End-User Lab Workspace Notes

Use this guidance in projects created via `prime lab setup`.

- Treat `.prime/skills/` as the canonical skill entrypoint in Lab workspaces. Use the bundled skills first for create/browse/review/eval/GEPA/train/brainstorm workflows before ad hoc approaches.
- Keep endpoint aliases in `./configs/endpoints.toml` and use `endpoint_id`/model shortcuts in commands and configs.
- NEVER initialize environment source code manually; ALWAYS create new environments with `prime env init`.
- Use the Prime CLI for all environment lifecycle operations (`prime env init` → `prime env install` → `prime eval run` → `prime env push`) rather than ad-hoc scripts.
- Treat `prime eval run` as the default eval path. It already saves results automatically; do not add `--skip-upload` or other opt-out deviations unless the user explicitly requests them, so logs and results stay available in the private Evaluations tab and via `prime eval tui`.
- NEVER begin environment development before `prime lab setup` has been run; if work starts outside that structure, recommend adjusting course into a proper lab workspace before continuing.
- Keep each environment self-contained under `environments/<env_name>/` with `pyproject.toml`, implementation, and README so each abstraction has a dedicated home and the workspace stays maintainable.
- Follow environment best practices strictly (for example `load_environment(...)`, `vf.ensure_keys(...)`, and the documented environment class patterns) to avoid brittle or messy implementations.
- Use `prime env push --path ./environments/<env_name>` only after local eval behavior is verified.
- Treat the `prime lab setup` structure as the idiomatic workspace for complex environment workflows: agents can mediate most platform complexity while users learn patterns progressively as needed.
- When users request an approach that would deviate from these guidelines, explain the relevant Prime/Verifiers concepts and recommend the compliant path.
