# opencode-harbor

### Overview
- **Environment ID**: `opencode-harbor`
- **Short description**: Environment for running an agent with OpenCode on Harbor tasks
- **Tags**: opencode, cli_agent, harbor

### Datasets
- **Primary dataset(s)**: Harbor tasks
- **Source links**: <https://github.com/laude-institute/harbor>
- **Split sizes**: 11 bundled tasks

### Task
- **Type**: multiturn, cli_agent
- **Rubric overview**: Binary, returned by running task tests

### Quickstart
Run the environment:

```bash
prime eval run opencode-harbor
```

Configure model and sampling:

```bash
prime eval run opencode-harbor -m openai/gpt-4.1-mini -n 20 -r 3 -t 1024 -T 0.7
```

Notes:
- Use `-a` / `--env-args` for nested Taskset/Harness overrides.
- Use `taskset` for Harbor task selection and `harness` for OpenCode runtime settings.

### Environment Arguments

Taskset settings use fields from `tasksets.HarborTasksetConfig`:

```bash
prime eval run opencode-harbor \
  -a '{"taskset": {"task_names": ["regex-log"]}}'
```

Harness settings use fields from `harnesses.OpenCodeConfig`; OpenCode program
settings live under `harness.program`:

```bash
prime eval run opencode-harbor \
  -a '{"harness": {"max_turns": 4, "program": {"disabled_tools": ["webfetch", "question"]}}}'
```

The same shape works in an eval TOML:

```toml
[[eval]]
env_id = "opencode-harbor"
taskset = { task_names = ["regex-log", "qemu-startup"] }

[eval.harness]
max_turns = 4

[eval.harness.program]
disabled_tools = ["webfetch", "question"]
```

By default this environment uses `harnesses.OpenCode` with only `webfetch` and
`question` disabled. Set `harness.program.disabled_tools` to override that list.

### Metrics
Summarize key metrics your rubric emits and how they’re interpreted.

| Metric | Meaning |
| ------ | ------- |
| `reward` | Main scalar reward (weighted sum of criteria) |


## How It Works

1. `tasksets.HarborTaskset` loads Harbor tasks and contributes sandbox settings,
   task uploads, env vars, and the Harbor reward.
2. `harnesses.OpenCode` contributes the reusable OpenCode CLI program, install/setup,
   intercepted endpoint config, MCP tool proxy, and log artifact collection.
3. The v1 runtime resolves both sides into one sandboxed command program at rollout time.
4. Reward is computed by running the Harbor test scripts after the rollout.

`HarborTaskset` and `OpenCode` live in the standalone `tasksets` and
`harnesses` packages.

## Requirements

- Harbor tasks directory with `task.toml` and `instruction.md` files
- Docker images specified in task configs


## Reward

Uses Harbor's standard reward mechanism:

- Runs `tests/test.sh` after agent completion
- Reads reward from `/logs/verifier/reward.txt` or `/logs/verifier/reward.json`
- Returns float reward value (typically 0 or 1)

## Notes

- OpenCode is installed at runtime.
- Agent logs are saved to `/logs/agent/opencode.txt` in the sandbox
- Uses `@ai-sdk/openai-compatible` provider for API interception
