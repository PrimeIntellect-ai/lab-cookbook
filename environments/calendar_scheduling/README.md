# calendar-scheduling

V1 `Taskset` environment for meeting-time negotiation under realistic calendar constraints.

## Overview

- **Environment ID**: `calendar-scheduling`
- **Type**: multi-turn tool use (`vf.Taskset` + `vf.Toolset`)
- **Objective**: submit one meeting window that maximizes weighted attendee utility while satisfying hard constraints
- **Reward**:
  - `0.0` if final submission is invalid or missing
  - otherwise weighted average attendee utility in `[0, 1]`

Each generated `vf.Task` contains a user prompt, oracle answer metadata, and a
serializable calendar world under `task["info"]["calendar_task"]`. The calendar
world contains:

- attendees with busy calendars,
- required and optional participants,
- timezone offsets,
- hard local-time bounds (for a subset of attendees),
- soft preferences (day, early/late, back-to-back),
- room availability,
- fixed meeting duration.

Attendee importance weights are normalized to sum to `1.0` per task.

## Tools

- `check_attendee_calendar(attendee_id, day_index)`
- `view_attendee_constraints(attendee_id)`
- `check_proposal(day_index, start_time_utc, duration_minutes, room_id)`
- `submit_window(day_index, start_time_utc, duration_minutes, room_id)`

All tool responses include `remaining_turns`. `check_proposal` is limited by a per-task score-check budget to discourage brute-force probing.

## Deterministic Oracle

Generation uses deterministic rejection sampling plus exhaustive search over candidate windows to compute:

- `optimal_score`
- `best_proposals`
- valid candidate count and search diagnostics

This metadata is stored with each example and used for metrics (for example submission-to-optimal ratio).

## Quickstart

Install local environment:

```bash
prime env install calendar-scheduling
```

Run a small eval:

```bash
prime eval run calendar-scheduling -n 5 -r 1 -m qwen3-30b-i -e configs/endpoints.toml --harness.max-turns 16
```

Run with typed taskset and harness config:

```bash
prime eval run calendar-scheduling -n 8 -r 1 -m qwen3-30b-i -e configs/endpoints.toml --taskset.difficulty hard --harness.max-turns 12
```

For repeated runs, put taskset-owned fields under `[eval.taskset]` and
harness-owned fields under `[eval.harness]`. Put fine-grained generator overrides
under `[eval.taskset.generator_overrides]`.

## Standalone TUI

Render a generated problem in terminal:

```bash
uv run calendar-scheduling-tui --difficulty medium --seed 42 --show-oracle
```

The TUI shows attendees, room lanes, busy blocks, and highlights oracle windows when `--show-oracle` is enabled.

## Taskset And Harness Config

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `difficulty` | `str` | `"medium"` | High-level task preset (`easy`, `medium`, `hard`) |
| `num_train` | `int` | `512` | Number of generated training tasks |
| `num_eval` | `int` | `128` | Number of generated evaluation tasks |
| `seed` | `int` | `7` | Base deterministic seed |
| `generator_overrides` | `GenerationOverrides` | empty | Typed fine-grained generation overrides |
| `harness.max_turns` | `int` | `10` | Turn cap per rollout |

Common `generator_overrides` keys:

- `attendee_count_range`
- `window_days_range`
- `meeting_duration_choices`
- `score_check_budget`
- `min_valid_candidates`
- `max_valid_ratio`
- `max_random_baseline_score`
- `max_generation_attempts`

## Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | Final reward (submitted valid score or `0.0`) |
| `submission_made` | Whether agent called `submit_window` |
| `submission_valid` | Whether submitted window passed hard constraints |
| `oracle_optimal_score` | Exhaustive best possible score for the task |
| `submitted_to_optimal_ratio` | `submitted_score / oracle_optimal_score` (clamped) |
| `optimality_gap` | `oracle_optimal_score - submitted_score` |
| `score_checks_used` | Number of `check_proposal` calls consumed |
| `score_checks_remaining` | Remaining proposal-check budget |
