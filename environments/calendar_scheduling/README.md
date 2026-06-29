# calendar-scheduling

A v1 synthetic scheduling environment.

- Taskset: generates calendar constraints and meeting requests
- Toolset: calendar inspection, proposal scoring, and final submission tools
- State: `CalendarState` records score checks, proposals, and final submission
- Stop: fires when a proposal is submitted
- Reward: normalized final proposal score

Run:

```bash
uv run eval @ configs/11/calendar-scheduling-eval.toml
```

Example config:

```toml
num_tasks = 5
num_rollouts = 2
max_turns = 8

[taskset]
id = "calendar-scheduling"
difficulty = "medium"
seed = 7
num_tasks = 32
```
