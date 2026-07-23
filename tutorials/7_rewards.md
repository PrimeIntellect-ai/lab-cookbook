# Designing Rewards

In reinforcement learning, the reward is what turns a behavior into a score, which is what evals report and what training maximizes.

This tutorial covers a scoring toolbox including weighted rewards, metrics, stop conditions, and group rewards. We will use the `code_golf_v1` taskset, where the goal is to write a short, fast Python program with a known output. In this taskset, one task carries three different signals: **correctness** (did it print the right thing?), **conciseness** (is it the shortest solution?), and **speed** (is it the quickest?) — and, as you'll see, each one wants a different scoring mechanism.

**You need:** [Build Your First Environment](5_build_first_environment.md).

## Scoring decorators

A taskset's scoring surface is a set of decorated async methods. Here are the 4 decorators:


| Decorator                      | Runs                    | May request        | Returns                       | Counts toward score? |
| ------------------------------ | ----------------------- | ------------------ | ----------------------------- | -------------------- |
| `@vf.reward(weight=...)`       | per rollout             | `trace`, `runtime` | `float`                       | yes, weighted        |
| `@vf.metric`                   | per rollout             | `trace`, `runtime` | `float` or `dict[str, float]` | no — observability   |
| `@vf.group_reward(weight=...)` | per *group* of rollouts | `traces`           | `list[float]`                 | yes, weighted        |
| `@vf.stop(priority=...)`       | after each turn         | `trace`            | `bool`                        | controls rollout end |


A pure-trace reward does not need to accept `runtime`. On the other hand, a method that does request `runtime` runs while the rollout's runtime is still alive, which is how this environment executes the model's program as part of scoring, for example in programming environments..

## Scoring surface

Here is `code_golf_v1`'s complete `Task` (`environments/code_golf_v1/code_golf_v1/taskset.py`):

```python
class CodeGolfTask(vf.Task[CodeGolfData]):
    @vf.metric
    async def evaluate(self, trace: vf.Trace, runtime: vf.Runtime) -> dict[str, float]:
        """Run the program once in the rollout's runtime; record correctness + latency."""
        program = extract_program(trace)
        if not program:
            return {"passed": 0.0, "latency": 1e6}
        await runtime.write("solution.py", program.encode())
        start = time.perf_counter()
        result = await runtime.run(["python3", "solution.py"], {})
        latency = time.perf_counter() - start
        passed = float(
            result.exit_code == 0 and result.stdout.strip() == self.data.expected
        )
        return {"passed": passed, "latency": latency}

    @vf.reward
    async def correct(self, trace: vf.Trace) -> float:
        return trace.metrics.get("passed", 0.0)

    @vf.group_reward(weight=0.5)
    async def most_concise(self, traces: list[vf.Trace]) -> list[float]:
        """The shortest program in the group wins; ties share."""
        lengths = [len(extract_program(t)) or 10**9 for t in traces]
        best = min(lengths)
        return [1.0 if length == best else 0.0 for length in lengths]

    @vf.group_reward(weight=0.5)
    async def fastest(self, traces: list[vf.Trace]) -> list[float]:
        """The lowest recorded `latency` in the group wins; ties share."""
        times = [t.metrics.get("latency", 1e6) for t in traces]
        best = min(times)
        return [1.0 if t == best else 0.0 for t in times]
```

Four methods, four different jobs. The rest of the tutorial takes them one at a time.

## Metrics: measured once in the runtime

The `evaluate` metric is doing the environment's only expensive work: it writes the program into the rollout's runtime, executes it, and records `passed` and `latency` onto the trace. Note the return type — a metric may return a single float or a **dict of named floats**, and every value lands in `trace.metrics`.

This is a deliberate design move, not a convenience:

- **Anything that needs the runtime happens exactly once, per rollout.** The runtime is alive during per-rollout scoring but *not* during group scoring — so runtime-derived facts must be captured per rollout first. `evaluate` is that capture step.
- **Metrics contribute nothing to the score** — they're recorded on the trace for you, not for the model. `latency` here doesn't reward speed by itself; it just makes speed *visible* and comparable.

Use metrics deliberately; they're cheap and they make traces self-explaining. The `calendar-scheduling` environment is a master class: alongside a single reward it records `optimality_gap`, `submission_valid`, and `score_checks_used` — so a mediocre mean reward decomposes immediately into *didn't submit* vs. *submitted invalid* vs. *submitted suboptimal*. A useful discipline: **if you're tempted to fold a quantity into the reward "just to track it," make it a metric instead** — the reward is for what you want optimized, metrics are for what you want observed.

## Weighted rewards: primary signal + shaping

`correct` is the primary reward: weight 1.0, binary, honest. It doesn't re-run anything — it reads `passed` off the trace that `evaluate` already recorded. The two group rewards act as *shaping*: at weight 0.5 each, they reward being the best sibling on style axes the primary reward is blind to.

Every reward is also recorded by name in `trace.rewards`, so you can always decompose a score: a rollout at 1.0 that's all `most_concise` + `fastest` and no `correct` tells a very different story than a correct one.

And that decomposition matters here, because **this weight ratio has a deliberate edge**: a wrong-but-tiny program can collect `most_concise` + `fastest` = 1.0, exactly matching a correct rollout's `correct` = 1.0. Run it and you'll see it happen. That's the central discipline of shaping: **shaping rewards should never be winnable *instead of* the real objective, only *on the way to* it.** The standard fix is conditioning the shaping on correctness (only correct programs compete on length and speed) or shrinking the weights until correct-but-ugly always beats elegant-but-wrong. The environment leaves the flaw in on purpose — the "Try it" section asks you to fix it.

## Binary vs. smooth: pick the signal's resolution

The other axis of reward design is granularity:

- **Binary** (`passed`: the output matches or it doesn't) is honest and unhackable, and it's the right choice for *evaluation* — "the program works or it doesn't."
- **Smooth** (similarity ratio, fraction of tests passed, distance from optimal) grants partial credit — and that's what *training* wants. RL compares rollouts against each other; if every attempt scores 0.0, there is nothing to reinforce ([tutorial 3](3_first_rl.md)).

The trade-off: smoothness creates surface area for gaming (what scores 0.6 without being 60% correct?). A good compromise pattern is *binary primary, smooth shaping* — or smooth rewards whose partial credit provably tracks partial progress, like fraction-of-tests-passed. Group rewards offer a third way to get gradient without partial credit: `most_concise` is binary per rollout, but *relative* across the group — even among all-correct attempts, there's still a signal about which one to prefer.

## Stop conditions: when is a rollout done?

`code_golf_v1` has no `@vf.stop` at all — no tools, no user simulator, so the model replies once and the rollout ends. When an environment does run multi-turn, three layers of stopping compose, and it's worth keeping them distinct:

- **Task-logic stops** (`@vf.stop`) — the rollout reached a terminal state: a final answer submitted (`calendar-scheduling` stops on `trace.state.submitted`, with `priority=50` so it's checked before lower-priority conditions), a game won or lost.
  ```python
  @vf.stop(priority=50)
  async def submitted(self, trace: vf.Trace) -> bool:
      return trace.state.submitted
  ```
- **Natural conversation end** — with a [user simulator](8_user_simulators.md), the simulator returning no further messages ends the rollout without any explicit stop.
- **The** `max_turns` **cap** — env-level config, enforced by the framework across all harnesses, so nothing can loop forever. This is the safety net, not the design: a taskset that *relies* on `max_turns` to end is usually leaving reward on the table (the model gets no signal that it should have committed earlier).



## Group rewards: score rollouts against each other

Everything so far scores one rollout in isolation. A `@vf.group_reward` receives **all rollouts of the same task** and returns one score per rollout — which unlocks *relative* criteria that no per-rollout reward can express. "Is this program short?" has no absolute answer; "is it the shortest of the attempts?" does.

Look at how `fastest` gets its data: it never touches the runtime — it compares `t.metrics.get("latency")` across the group's traces. That's the pattern in full:

1. `@vf.metric` + `runtime` captures the fact, per rollout, while the runtime is alive.
2. `@vf.group_reward` + `traces` compares the recorded facts across siblings.

Practical notes:

- A taskset with group rewards **requires** `num_rollouts` **≥ 2** — there's no group to compare otherwise. (The cookbook config uses 2, so every task becomes a head-to-head duel.)
- In training, a group reward is **scoring, not the trainer's advantage computation** — the trainer's group-relative machinery ([tutorial 3](3_first_rl.md)) operates on top of whatever reward you define. The env server keeps a task's rollouts together so your group scores are computed over complete groups.



## Errors are not zeros

One rule keeps your numbers meaningful: **a reward of 0.0 must mean "the model failed," never "my scoring code broke."** If the verifier itself hits an invalid condition — malformed ground truth, a scoring dependency missing — raise an ordinary exception. The framework records it as a `TaskError` on the trace, which keeps it out of your means and visible in your error accounting ([Model Report Card](../recipes/eval_report_card.md) shows why that separation matters). Swallowing scoring bugs into zeros is how environments quietly rot.

Notice the distinction inside `evaluate`: a program that *runs and prints the wrong thing* is scored `passed = 0.0` — that's the model failing, a legitimate zero. If instead the runtime itself couldn't start, that should surface as an error, not a zero. The flip side, for tool-heavy environments: *model-facing* errors (a timeout, a bad tool argument) should usually be returned to the model as text so it can recover — see [Coding Agent Environments](11_coding_agents.md) for that pattern. Raise for broken environments; reply for recoverable model mistakes.

## Run it

```bash
uv run eval @ configs/07/code-golf-eval.toml
```

Each task runs twice (`num_rollouts = 2`), and each rollout's total decomposes in the summary and in `traces.jsonl`: `correct` from the trace's own metrics, `most_concise` and `fastest` from the duel against its sibling. Find a pair where both rollouts are correct and watch the group rewards break the tie — that preference signal exists nowhere in any single trace.

## Try it

- Fix the deliberate flaw: make `most_concise` and `fastest` competitions *among correct rollouts only* (filter on `t.metrics.get("passed")`), so elegant-but-wrong can no longer tie with correct. Re-run and compare reward decompositions.
- Deliberately break the ratio instead (`weight=2.0` on both group rewards) and run with `--num-rollouts 4`: watch wrong-but-tiny programs outscore correct ones. Recalibrate. This is the cheapest reward-hacking lab you'll ever run.
- Add a `verbosity` metric (length of the full reply, not just the program) and check whether high-reward rollouts are systematically longer — the cheapest reward-hacking probe there is.
- Write a third group reward that scores 1.0 only for rollouts whose *program text* differs from every sibling's, run with `--num-rollouts 8`, and see whether the model actually explores distinct solutions or repeats one.



## Next

→ [User Simulators](8_user_simulators.md): the environment plays the human side of a multi-turn conversation.