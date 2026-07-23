# code-golf-v1

Write a short, fast Python program — a tiny taskset that showcases the full v1 scoring toolbox, and the centerpiece of the [Designing Rewards](../../tutorials/7_rewards.md) tutorial.

Each task asks for a self-contained Python program with a known stdout. Rollouts are scored four ways:

- `evaluate` — per-rollout `@vf.metric`: runs the program in the rollout's runtime, records `passed` + `latency`.
- `correct` — per-rollout `@vf.reward`: reads `passed` off the trace.
- `most_concise` — `@vf.group_reward`: the shortest source in the group wins.
- `fastest` — `@vf.group_reward`: the lowest recorded `latency` in the group wins.

Run with `num_rollouts >= 2` so the group rewards have siblings to compare.

## Taskset

- **Source:** three inline task specs (`simple-sum`, `fibonacci`, `reverse-str`)
- **Size:** 3 tasks

## Changelog

- 2026-07-20: Copied from the official `verifiers` examples for the cookbook rewards tutorial.
