# Build Your First Environment

In the Basics we ran existing environments. Now let us build one.

We will take AIME 2026 — the American Invitational Mathematics Examination, a real competition whose 30 problems each have a known integer answer between 0 and 999, and build a verifiers environment around it. The task is genuinely hard (frontier models don't ace it), it's **single-turn** (one question, one reply), and it's **verifiable** (an integer either matches the gold answer or it doesn't — no judge required).

The finished version ships with the cookbook at `environments/aime26_v1/aime26_v1/`, so you can run everything in this tutorial immediately and read along in the real files.

## Pieces

For your own environments, you can use the CLI to initialize an environment with the package skeleton:

```bash
prime env init my-env
```

This creates a single-file package under `./environments` with a stub module and `pyproject.toml`. (The generated stub currently uses the legacy `load_environment` shape — we'll replace its contents with the v1 taskset we build below.) Pass `--multi-file` when the environment needs separate tool or user server modules.

Let's go over the components we need. First of all, our environment wraps the real 2026 problem set for AIME, and that will be our immutable `TaskData`. 

### 1. TaskData: fields for a task

```python
class AIME26Data(vf.TaskData):
    answer: str
```

`TaskData` contains problem's serializable fields. The base class carries `idx`, `prompt`, and `system_prompt`, and then we add whatever scoring needs. In this case, we've added the golden answer we need for scoring.

### 2. The Task: behavior and scoring

In verifiers, a `Task` wraps `TaskData` , so our `AIME26Task` wraps one `AIME26Data` row. Decorated scoring methods live here and they read ground truth task data via `self.data`:

```python
class AIME26Task(vf.Task[AIME26Data, vf.State, AIME26TaskConfig]):
    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace) -> float:
        return vf.verify_boxed_math_answer(
            trace.last_reply,
            self.data.answer,
            timeout_seconds=self.config.math_verify_timeout,
        )
```

A `@vf.reward` reads the finished trace and returns a score. Here, `vf.verify_boxed_math_answer` extracts the last `\boxed{...}` from the reply and checks mathematical equivalence against the gold; no need for writing custom parsers most of the time.

**Note that there is no** `@vf.stop` **here.** A taskset with no tools and no user simulator is naturally single-turn under the `default` harness, so the model replies once and the rollout is over. Explicit `@vf.stop` conditions will be needed for multi-turn environments that need to end early; they are covered in [Designing Rewards](7_rewards.md).

### 3. The Config: knobs tunable by the runner

Config comes in two layers, and this environment uses both:

```python
class AIME26TaskConfig(vf.TaskConfig):
    math_verify_timeout: int = 5

class AIME26Config(vf.TasksetConfig):
    dataset_name: str = "MathArena/aime_2026"
    dataset_split: str = "train"
    dataset_revision: str = "10b4e45b7a503075d4da8a0d57916a4f06ce6bd2"
    task: AIME26TaskConfig = AIME26TaskConfig()
```

Config fields expose things a user may want to change at run time without touching the environment code. They surface automatically in the `[taskset]` section of a TOML config and as `--taskset.*` in the CLI:

```bash
uv run eval aime26_v1 --num-tasks 3 --taskset.task.math-verify-timeout 10
```

Note the `dataset_revision` pin: the taskset names an exact commit of the Hugging Face dataset, so a run today and a run next year score the same problems. Since many datasets often live in someone else's repository, pinning it is what keeps an eval or a training run reproducible. If your taskset has nothing to configure, use the empty base `vf.TasksetConfig` directly.

### 4. `Taskset.load`: loading rows

`Taskset.load()` turns a data source into typed task objects. Here it pulls the pinned Hugging Face dataset, builds each `AIME26Data` row (prepending the instruction, normalizing the answer to a string), and wraps it with `AIME26Task` plus the shared task config.

```python
AIME26Task(
    AIME26Data(idx=i, prompt=INSTRUCTION + row["problem"], answer=str(int(row["answer"]))),
    self.config.task,
)
```



### 5. The export

`__all__` exposes exactly one taskset class; this is how the loader resolves `taskset.id = "aime26_v1"` to your class. Everything else in the package is implementation detail.

## Run it

```bash
uv run eval @ configs/05/aime26-eval.toml
```

These are real competition problems, so a small model will miss most of them, and that's the point — the reward is honest. You can then open the run's `traces.jsonl` (see [tutorial 2](2_first_eval.md)) and trace the reward by hand, find its last `\boxed{...}`, and compare against the task's `answer`. In code, the same text is `trace.last_reply`.

## Recap

- **TaskData fields** are the immutable per-row truth used by scoring.
- **Task methods** own hooks, stopping, tools, user simulation, metrics, and rewards.
- **Config fields** are runner-tunable knobs exposed through `[taskset]` / `--taskset.`*; pin external data sources there.
- **Rewards read the** `Trace`; they never parse framework internals.
- `__all__` **exposes exactly one taskset class** for loader resolution.

## Next

→ [Judges](6_judges.md): when correctness is semantic — like the *tone* of a reply — and exact match won't do, score with an LLM judge, controlled through config.