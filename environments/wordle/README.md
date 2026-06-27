# wordle

A v1 wrapper around the built-in `verifiers.v1.tasksets.textarena` taskset pinned to `Wordle-v0`.

- Taskset: `WordleTaskset`
- User simulator: TextArena game engine from the built-in taskset
- Config: `num_tasks`, plus the inherited `user` placement config
- Reward: TextArena's game-authoritative reward read from the rollout runtime

Run:

```bash
uv run eval @ configs/04/wordle-eval.toml
```

Example config:

```toml
num_tasks = 20
num_rollouts = 1
max_turns = 6

[taskset]
id = "wordle"
num_tasks = 100

[harness]
id = "default"
```
