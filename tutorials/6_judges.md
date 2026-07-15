# Judges

Every reward so far was computable: string similarity, exact match, a solved game. However, we cannot use regex for softer criteria, such as "does this response sound upbeat?". In this tutorial you will score semantic criteria with an **LLM judge**, using the `simple-judge` environment — six instruction-following tasks, each graded against one natural-language criterion.

One rule before the mechanics: **prefer deterministic verification whenever the task's artifact allows it.** A judge adds cost, latency, and a second model's failure modes to every rollout. Reach for one only when semantic judgment is unavoidable, such as for tone, style, adherence to soft instructions, and more.

**You need:** tutorials [1](1_setup.md)–[2](2_first_eval.md) and [Build Your First Environment](5_build_first_environment.md).

## The task shape

A judged task carries its rubric as ordinary task data (`environments/simple_judge/simple_judge.py`):

```python
class SimpleJudgeTask(vf.Task):
    criterion: str

TOY_TASKS = [
    {
        "prompt": "Write one cheerful sentence about mornings.",
        "criterion": "The response sounds upbeat and enthusiastic.",
    },
    ...
]
```

Same boundary as always: the criterion is per-row ground truth, so it lives on the task.

## The judge

`vf.Judge` packages the pattern — a prompt template, a call to the judge model, and a parser for its verdict:

```python
import verifiers.v1 as vf
from functools import cached_property

class CorrectnessJudge(vf.Judge[bool]):
    prompt = """Criterion: {criterion}
    User message: {question}
    Response: {response}
    Does the response satisfy the criterion? Reply with exactly one word: yes or no."""

    def parse(self, response: vf.JudgeResponse[bool]) -> bool:
        return "yes" in response.text.lower()


class Config(vf.TasksetConfig):
    # inherits base_url and API keys from the prime config by default
    judge: vf.JudgeConfig = vf.JudgeConfig(model="openai/gpt-4.1-mini")


class JudgedTaskset(vf.Taskset[SimpleJudgeTask, Config]):
    @cached_property
    def judge(self) -> CorrectnessJudge:
        return CorrectnessJudge(self.config.judge)

    @vf.reward(weight=1.0)
    async def judged(self, task: SimpleJudgeTask, trace: vf.Trace) -> float:
        result = await self.judge.evaluate(
            trace=trace,
            criterion=task.criterion,
            question=task.prompt,
            response=trace.last_reply,   # the last assistant message
        )
        return 1.0 if result.parsed else 0.0
```

Read it as three responsibilities:

1. **The rubric** (`prompt`) is a template; `evaluate(...)` keyword arguments fill its placeholders. Keep it narrow — one question with explicit allowed answers ("yes or no").
2. **The parser** (`parse`) turns free text into a typed verdict. In this example, we parse defensively, so a judge that answers "Yes, because..." will still count.
3. **The reward** stays a normal `@vf.reward` — it feeds the judge the typed task and the finished trace, then maps the verdict to a float. The `cached_property` builds the judge lazily from config; no `__init__` override needed.

The cookbook's `simple_judge.py` implements this same reward one level lower, with `vf.resolve_client` and a hand-rolled prompt — read it to see exactly what `vf.Judge` does for you.

## Configuring the judge

Because the judge config is a normal nested taskset config, everything is overridable without code changes (`configs/07/simple-judge-eval.toml`):

```toml
model = "openai/gpt-5.4-nano"   # the model being evaluated
num_tasks = 6
num_rollouts = 2

[sampling]
max_tokens = 512

[taskset]
id = "simple-judge"

[taskset.judge]
model = "openai/gpt-4.1-mini"   # the model doing the grading
```

Note there are two models in play: `model` at the top is the *subject*, `taskset.judge.model` is the *grader*. Judge sampling knobs nest one level deeper, e.g. `taskset.judge.sampling.max_tokens`. Run it:

```bash
prime eval run @ configs/07/simple-judge-eval.toml
```

Then audit the judge like you would any reward ([tutorial 2](2_first_eval.md)): pick a few traces, read the response, and ask whether *you* agree with the verdict. A judge is part of your environment — an unaudited judge is an unaudited reward.

## Design rules for judge rewards

- **One rubric question per judge.** "Is it polite AND correct AND concise?" produces incoherent verdicts; use separate rewards (or metrics) per dimension.
- **Constrain the output format** and parse it defensively.
- **Pin the judge model** in config so results are comparable across runs — a silently-upgraded judge is a moving goalpost.
- **Keep the judge cheap.** It runs once per rollout per judged reward; a frontier model as grader can cost more than the rollout itself.
- For training ([tutorial 3](3_first_rl.md)), remember the judge is called inside the reward path — its latency and failure rate directly gate rollout throughput.

## Try it

- Flip a criterion to its negation ("The response sounds gloomy...") and confirm the reward flips too — the fastest sanity check that the judge reads the rubric at all.
- Add a `@vf.metric` that records the judge's raw text verdict into the trace, so disagreements are auditable later.
- Swap the judge model via `--taskset.judge.model` and measure verdict agreement across the two graders.

## Next

→ [Designing Rewards](7_rewards.md): weights, metrics, stop conditions, and group rewards — the full scoring toolbox.
