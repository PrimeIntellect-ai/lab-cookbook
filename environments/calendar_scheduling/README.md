# calendar-scheduling

A v1 synthetic scheduling environment — the reference example of the synthetic-agent pattern: world state lives in typed task data and per-rollout `vf.State`, interaction happens through tools, and the reward scores the final state.

- Taskset: generates calendar constraints and meeting requests
- Toolset: calendar inspection, proposal scoring, and final submission tools
- State: `CalendarState` records score checks, proposals, and final submission
- Stop: fires when a proposal is submitted
- Reward: normalized final proposal score

The three pieces of the pattern, from `calendar_scheduling/taskset.py`:

```python
class CalendarState(vf.State):
    score_checks_remaining: int = -1
    proposal_checks: list[dict[str, object]] = []
    submitted: bool = False
    submitted_score: float = 0.0
```

```python
@vf.stop(priority=50)
async def has_submission(self, trace: vf.Trace) -> bool:
    return trace.state.submitted
```

```python
@vf.reward(weight=1.0)
async def final_score_from_submission(self, trace: vf.Trace) -> float:
    if not trace.state.submitted_valid:
        return 0.0
    return trace.state.submitted_score
```

Because the environment generates its own worlds, ground truth is exact and tasks are unlimited; metrics like `optimality_gap` decompose a mediocre mean into didn't-submit vs. invalid vs. suboptimal (see the [Designing Rewards](../../tutorials/7_rewards.md) tutorial, which uses this environment as its metrics example).

Run:

```bash
uv run eval @ configs/15/calendar-scheduling-eval.toml
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
