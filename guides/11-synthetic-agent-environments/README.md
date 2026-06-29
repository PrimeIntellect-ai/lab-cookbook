# Synthetic Agent Environments

Synthetic environments keep world state in typed task data and per-rollout `vf.State`, then expose interaction through tools or a user simulator.

`calendar-scheduling` generates meeting constraints, stores per-rollout proposal state in `CalendarState`, and exposes calendar tools through `CalendarToolset`.

```python
class CalendarState(vf.State):
    score_checks_remaining: int = -1
    proposal_checks: list[dict[str, object]] = []
    submitted: bool = False
    submitted_score: float = 0.0
```

The taskset stops when the model submits a valid or invalid final proposal:

```python
@vf.stop(priority=50)
async def has_submission(self, trace: vf.Trace) -> bool:
    return trace.state.submitted
```

Metrics and rewards read the final `trace.state`:

```python
@vf.reward(weight=1.0)
async def final_score_from_submission(self, trace: vf.Trace) -> float:
    if not trace.state.submitted_valid:
        return 0.0
    return trace.state.submitted_score
```

Run it with:

```bash
uv run eval @ configs/11/calendar-scheduling-eval.toml
```

Keep simulated-world state serializable. Runtime handles, clients, and caches belong on tool/user server instances, not inside `vf.State`.
