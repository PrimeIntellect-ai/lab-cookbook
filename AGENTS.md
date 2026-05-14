# AGENTS.md

<!-- Generated for lab workspaces. -->

This AGENTS guide is intended for end users working in a `prime lab setup` workspace.

## Shared Best Practices (All Contexts)

These points are direct restatements of Verifiers docs so agents can follow the same golden-path workflows.

- Environments are expected to expose `load_environment(...) -> vf.Environment` and be installable with `prime env install <env-name>`. (See `docs/overview.md` and `docs/environments.md`.)
- Validate environment behavior with `prime eval run <env-name> -m <provider/model>` before sharing/publishing changes. Treat this literal shape as the canonical eval path: it uses the proper eval artifact targets, saves results automatically, and keeps runs visible in the private Evaluations tab and in `prime eval tui`. Agents should not add prefixes, custom output dirs, save flags, smoke `-n`/`-r` values, `--provider`, `--skip-upload`, manual key exports, TUI flags, concurrency flags, or other deviations unless the user explicitly requests them. (See `docs/overview.md` and `docs/development.md`.)
- `prime eval run` uses the logged-in Prime CLI session for Prime Inference; do not require or manually export `PRIME_API_KEY` just to run a Prime-backed eval. Confirm login with `prime whoami` only if auth state is unclear.
- Use `ToolEnv`/`MCPEnv` for stateless tools and `StatefulToolEnv` when per-rollout state must persist (sandbox/session/db handles). (See `docs/environments.md`.)
- If external API keys are required, validate them in `load_environment()` with `vf.ensure_keys(...)` so failures are explicit and early. (See `docs/environments.md`.)

## End-User Lab Workspace Notes

Use this guidance in projects created via `prime lab setup`.

- Treat `.prime/skills/` as the canonical skill entrypoint in Lab workspaces. Use the bundled skills first for create/browse/review/eval/GEPA/train/brainstorm workflows before ad hoc approaches.
- Use `prime eval run <env-name> -m <provider/model>` as the default eval command. Endpoint aliases and custom key/base-url wiring are for explicit non-default cases, not the normal Prime Inference path.
- For Prime Inference evals, rely on Prime CLI login rather than a `PRIME_API_KEY` environment variable. Environment-specific secrets are separate and should only be required when the environment itself calls an external service.
- NEVER initialize environment source code manually; ALWAYS create new environments with `prime env init`.
- Use the Prime CLI for all environment lifecycle operations (`prime env init` → `prime env install` → `prime eval run` → `prime env push`) rather than ad-hoc scripts.
- Treat `prime eval run` as the default eval path. It already uses the correct artifact targets and save behavior; do not add `--skip-upload`, `--save-results`, `-o`, `-n`, `-r`, or other opt-out/override deviations unless the user explicitly requests them.
- Set every environment's `[tool.verifiers.eval]` defaults to `num_examples = 5` and `rollouts_per_example = 3` so the bare eval command has the right smoke shape without CLI flags.
- NEVER begin environment development before `prime lab setup` has been run; if work starts outside that structure, recommend adjusting course into a proper lab workspace before continuing.
- Keep each environment self-contained under `environments/<env_name>/` with `pyproject.toml`, implementation, and README so each abstraction has a dedicated home and the workspace stays maintainable.
- Follow environment best practices strictly (for example `load_environment(...)`, `vf.ensure_keys(...)`, and the documented environment class patterns) to avoid brittle or messy implementations.
- Use `prime env push --path ./environments/<env_name>` only after local eval behavior is verified.
- Treat the `prime lab setup` structure as the idiomatic workspace for complex environment workflows: agents can mediate most platform complexity while users learn patterns progressively as needed.
- When users request an approach that would deviate from these guidelines, explain the relevant Prime/Verifiers concepts and recommend the compliant path.
