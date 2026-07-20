# Judges

Every reward so far was computable -- a boxed integer either matches the gold answer or it doesn't. But suppose the thing you care about is the **tone** of a response, like 

- Whether an apology actually sound apologetic?
- Is the explanation accessible enough for a five-year-old?
- Does the cheerful sentence *sound* cheerful? 
- No regex can check that, yet these are exactly the qualities that make an assistant worth talking to. In this tutorial you'll grade tone with an **LLM judge**, using the cookbook's v1 `simple-judge` taskset: six short writing tasks — sound upbeat, decline politely, apologize sincerely — each scored against a natural-language criterion.

We cannot verify this with regex, but they constitute real signals that we might want to use for evaluation or training. In this tutorial you'll grade tone with an **LLM judge**, in a taskset where six short writing tasks each come with a natural-language criterion that they should be verified against.

**You need:** tutorials [1](1_setup.md)–[2](2_first_eval.md) and [Build Your First Environment](5_build_first_environment.md).

## The task shape

A judged task carries its rubric as immutable `TaskData` — the criterion states, in plain language, what the right tone is:

```python
class SimpleJudgeTaskData(vf.TaskData):
    criterion: str
```

The tasks themselves live in a plain JSON file next to the module (`simple_judge/tasks.json`). Both the prompts and criteria are *data*, so they belong in a data file you can extend without touching code. In this toy example, they are saved locally, but typically you would save these data files as HuggingFace datasets.

```json
[
  {
    "prompt": "Write one cheerful sentence about mornings.",
    "criterion": "The response sounds upbeat and enthusiastic."
  },
  ...
]
```

The criterion is per-row ground truth, so it lives on `TaskData`; behavior belongs on `Task`.

## The judge

`vf.Judge` packages a prompt template, a call to the judge model, and a parser for its verdict:

```python
import verifiers.v1 as vf

class CorrectnessJudge(vf.Judge[bool]):
    prompt = """Criterion: {criterion}
    User message: {question}
    Response: {response}
    Does the response satisfy the criterion? Reply with exactly one word: yes or no."""

    def parse(self, response: vf.JudgeResponse[bool]) -> bool:
        return "yes" in response.text.lower()


class SimpleJudgeTaskConfig(vf.TaskConfig):
    judge: vf.JudgeConfig = vf.JudgeConfig(model="openai/gpt-4.1-mini")


class SimpleJudgeConfig(vf.TasksetConfig):
    task: SimpleJudgeTaskConfig = SimpleJudgeTaskConfig()


class SimpleJudgeTask(vf.Task[SimpleJudgeTaskData, vf.State, SimpleJudgeTaskConfig]):
    @vf.reward(weight=1.0)
    async def judged(self, trace: vf.Trace) -> float:
        result = await CorrectnessJudge(self.config.judge).evaluate(
            trace=trace,
            criterion=self.data.criterion,
            question=self.data.prompt_text,
            response=trace.last_reply,
        )
        return float(result.parsed)


class SimpleJudgeTaskset(vf.Taskset[SimpleJudgeTask, SimpleJudgeConfig]):
    def load(self) -> list[SimpleJudgeTask]:
        rows = json.loads(TASKS_FILE.read_text())
        return [
            SimpleJudgeTask(
                SimpleJudgeTaskData(idx=i, prompt=row["prompt"], criterion=row["criterion"]),
                self.config.task,
            )
            for i, row in enumerate(rows)
        ]
```

The implementation lives in `environments/simple_judge/simple_judge/taskset.py`. Passing `trace=` to `evaluate(...)` records the judge response, tokens, and cost instead of making grading an invisible side call.

## Configuring the judge

Because the judge belongs to `TaskConfig`, its v1 config nests under `taskset.task`:

```toml
model = "openai/gpt-5.4-nano"   # the model being evaluated
num_tasks = 6
num_rollouts = 2

[sampling]
max_tokens = 512

[taskset]
id = "simple-judge"

[taskset.task.judge]
model = "openai/gpt-4.1-mini"   # the model doing the grading
```

There are two models: `model` is the subject; `taskset.task.judge.model` is the grader. Judge sampling knobs nest one level deeper, for example `taskset.task.judge.sampling.max_tokens`.

```bash
uv run eval @ configs/06/simple-judge-eval.toml
```

Audit the resulting `traces.jsonl`. An unaudited judge is an unaudited reward.

## Design rules for judge rewards

- **One rubric question per judge.** "Is it polite AND correct AND concise?" produces incoherent verdicts; use separate rewards (or metrics) per dimension.
- **Constrain the output format** and parse it defensively.
- **Pin the judge model** in config so results are comparable across runs — a silently-upgraded judge is a moving goalpost.
- **Keep the judge cheap.** It runs once per rollout per judged reward; a frontier model as grader can cost more than the rollout itself.
- For training ([tutorial 3](3_first_rl.md)), remember the judge is called inside the reward path — its latency and failure rate directly gate rollout throughput.



## Try it

- Add a seventh row to `tasks.json` — a tone you care about (sarcastic, reassuring, businesslike) — and re-run. No code changes needed: that's why the tasks are a data file.
- Flip a tone criterion to its negation ("The response sounds gloomy and defeated...") and confirm the reward flips too — the fastest sanity check that the judge reads the rubric at all.
- Inspect the judge response in `trace.info["judge"]` and compare it with your own verdict.
- Swap the judge model via `--taskset.task.judge.model` and measure verdict agreement across the two graders.



## Next

→ [Designing Rewards](7_rewards.md): weights, metrics, stop conditions, and group rewards — the full scoring toolbox.