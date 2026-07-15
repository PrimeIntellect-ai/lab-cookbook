# Designing Rewards

The reward is the product. Everything else in an environment — tasks, tools, harnesses — exists to produce behavior; the reward is what turns that behavior into a number, and that number is what evals report and what RL optimizes. A sloppy reward doesn't just mismeasure — under training, it actively teaches the wrong thing. This tutorial covers the full scoring toolbox: weighted rewards, metrics, stop conditions, and group rewards, plus the design judgment for when to use which.

**You need:** [Build Your First Environment](5_build_first_environment.md) — this tutorial builds directly on its `reverse_text` example.

## The scoring decorators, at a glance

A taskset's scoring surface is a set of decorated async methods. The framework injects arguments **by parameter name** — declare only what you need:

| Decorator | Runs | May request | Returns | Counts toward score? |
| --- | --- | --- | --- | --- |
| `@vf.reward(weight=...)` | per rollout | `task`, `trace`, `runtime` | `float` | yes, weighted |
| `@vf.metric` | per rollout | `task`, `trace`, `runtime` | `float` | no — observability |
| `@vf.group_reward(weight=...)` | per *group* of rollouts | `task`, `traces` | `list[float]` | yes, weighted |
| `@vf.stop(priority=...)` | after each turn | `trace` | `bool` | controls rollout end |

Two consequences of name-based injection worth knowing: a pure-trace reward doesn't need to accept `runtime`, and a reward that *does* request `runtime` runs while the rollout's runtime is still alive — which is how coding environments run tests inside the container as part of scoring.

## Weighted rewards: primary signal + shaping

You can register any number of `@vf.reward` methods; the rollout's overall reward is their weighted combination, and each named reward is also recorded individually on the trace. The standard pattern is one dominant correctness reward plus small *shaping* rewards:

```python
@vf.reward(weight=1.0)
async def lcs(self, task: ReverseTextTask, trace: vf.Trace) -> float:
    """Primary: similarity between the extracted answer and ground truth."""
    ...

@vf.reward(weight=0.2)
async def format(self, trace: vf.Trace) -> float:
    """Shaping: did the model use the <reversed_text> tags at all?"""
    completion = trace.assistant_messages[-1].content if trace.assistant_messages else ""
    return float(TAG.search(completion or "") is not None)
```

Why shape at all? Early in training, a weak model may *never* produce a correct answer — but it can learn to produce the right *format* immediately, and the format reward hands it that first rung of the ladder. The weight ratio is the guardrail: at `1.0` vs `0.2`, correct-but-unformatted still beats formatted-but-wrong. Get the ratio wrong and you've built a model that produces beautifully formatted garbage — shaping rewards should never be winnable *instead of* the real objective, only *on the way to* it.

Because each reward is recorded by name in `trace.rewards`, you can always decompose a score: a rollout at `0.2` that's all `format` and no `lcs` tells a different story than one at `0.2` that's all `lcs`.

## Binary vs. smooth: pick the signal's resolution

The other axis of reward design is granularity:

- **Binary** (exact match, tests pass) is honest and unhackable, and it's the right choice for *evaluation* — "the answer is right or it isn't."
- **Smooth** (similarity ratio, fraction of tests passed, distance from optimal) grants partial credit — and that's what *training* wants. RL compares rollouts against each other; if every attempt scores 0.0, there is nothing to reinforce ([tutorial 3](3_first_rl.md)). `reverse_text`'s `SequenceMatcher` ratio is exactly this move: a model that reverses 80% of characters gets 0.8, and the gradient exists from step one.

The trade-off: smoothness creates surface area for gaming (what scores 0.6 without being 60% correct?). A good compromise pattern is *binary primary, smooth shaping* — or smooth rewards whose partial credit provably tracks partial progress, like fraction-of-tests-passed.

## Metrics: measure without scoring

A `@vf.metric` has the same shape as a reward but contributes nothing to the score — it's recorded on the trace for you, not for the model:

```python
@vf.metric
async def num_tool_calls(self, trace: vf.Trace) -> float:
    return float(len(trace.tool_messages))
```

Use metrics deliberately; they're cheap and they make traces self-explaining. The `calendar-scheduling` environment is a master class: alongside a single reward it records `optimality_gap` (how far from the oracle-best answer), `submission_valid`, and `score_checks_used` — so a mediocre mean reward can be immediately decomposed into *didn't submit* vs. *submitted invalid* vs. *submitted suboptimal*. Anything you'd otherwise compute in a notebook after the fact is a candidate metric. A useful discipline: **if you're tempted to fold a quantity into the reward "just to track it," make it a metric instead** — the reward is for what you want optimized, metrics are for what you want observed.

## Stop conditions: when is a rollout done?

A `@vf.stop` method is checked after each turn; returning `True` ends the rollout:

```python
@vf.stop
async def single_turn(self, trace: vf.Trace) -> bool:
    return trace.num_turns >= 1
```

Three layers of stopping compose, and it's worth keeping them distinct:

- **Task-logic stops** (`@vf.stop`) — the rollout reached a terminal state: one reply given, a final answer submitted (`calendar-scheduling` stops on `trace.state.submitted`, with `priority=50` so it's checked before lower-priority conditions).
- **Natural conversation end** — with a [user simulator](8_user_simulators.md), the simulator returning no further messages ends the rollout without any explicit stop.
- **The `max_turns` cap** — env-level config, enforced by the framework across all harnesses, so nothing can loop forever. This is the safety net, not the design: a taskset that *relies* on `max_turns` to end is usually leaving reward on the table (the model gets no signal that it should have committed earlier).

## Group rewards: score rollouts against each other

Everything so far scores one rollout in isolation. A `@vf.group_reward` receives **all rollouts of the same task** and returns one score per rollout:

```python
@vf.group_reward(weight=0.3)
async def most_concise_correct(self, task, traces) -> list[float]:
    """Among correct rollouts, favor the shortest solution."""
    lengths = [
        len(t.assistant_messages[-1].content or "") if is_correct(task, t) else None
        for t in traces
    ]
    best = min((l for l in lengths if l is not None), default=None)
    return [1.0 if l is not None and l == best else 0.0 for l in lengths]
```

This unlocks *relative* criteria that no per-rollout reward can express: favor the most concise correct answer, reward diversity across attempts, implement preference-style comparisons. Practical notes:

- A taskset with group rewards **requires `num_rollouts` ≥ 2** — there's no group to compare otherwise.
- Group rewards see `traces`, not the live runtime; anything runtime-derived should be captured per rollout as a `@vf.metric` first, then compared across traces in the group.
- In training, a group reward is **scoring, not the trainer's advantage computation** — the trainer's group-relative machinery ([tutorial 3](3_first_rl.md)) operates on top of whatever reward you define. The env server keeps a task's rollouts together so your group scores are computed over complete groups.

## Errors are not zeros

One rule keeps your numbers meaningful: **a reward of 0.0 must mean "the model failed," never "my scoring code broke."** If the verifier itself hits an invalid condition — malformed ground truth, a scoring dependency missing — raise an ordinary exception. The framework records it as a `TasksetError` on the trace, which keeps it out of your means and visible in your error accounting ([Model Report Card](13_eval_report_card.md) shows why that separation matters). Swallowing scoring bugs into zeros is how environments quietly rot.

The flip side, for tool-heavy environments: *model-facing* errors (a timeout, a bad tool argument) should usually be returned to the model as text so it can recover — see [Coding Agent Environments](11_coding_agents.md) for that pattern. Raise for broken environments; reply for recoverable model mistakes.

## Try it

- Add the `format` shaping reward to your `reverse_text` copy, then deliberately break the ratio (`weight=2.0`) and run 20 tasks: watch formatted-but-wrong rollouts outscore correct ones. Recalibrate.
- Add a `verbosity` metric (response length) to any environment and check whether high-reward rollouts are systematically longer — the cheapest reward-hacking probe there is.
- Write a group reward that scores 1.0 only for rollouts whose answer *disagrees* with the majority answer in the group, run with `-r 8`, and look at what it surfaces on ambiguous tasks.
- Audit a smooth reward: sample ten rollouts scoring 0.4–0.7 and check the partial credit actually corresponds to partial progress. If it doesn't, your gradient is noise.

## Next

→ [User Simulators](8_user_simulators.md): the environment plays the human side of a multi-turn conversation.
