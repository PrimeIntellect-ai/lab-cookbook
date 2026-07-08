# 04 — Prompt Optimization

Optimizing a model to get better in an environment (or multiple) can be done in different ways. Commonly, these methods train the weights of a model, but this this requrires access to the model's weights, as well as . A cheaper, but still powerful method of optimizing the performance of frontier, closed-source models is by editing the prompt to maximize performance. In this guide you will learn where prompts live in a v1 environment, how to run a disciplined prompt-iteration loop with the eval CLI, and crucially, how to run prompt optimization (via the [GEPA](https://arxiv.org/abs/2507.19457) algorithm) to automatically evolve a better prompt.

## Where prompts live

In v1, prompts are **task data**, not framework configuration. You saw this in Guide 02: `reverse_text.py` keeps its instruction in a module constant and attaches it to every task:

```python
SYSTEM = "Reverse the text character-by-character. Put your answer in <reversed_text> tags."

ReverseTextTask(idx=i, prompt=row["prompt"], system_prompt=SYSTEM, ...)
```

Static instructions go in constants or packaged text files, attached per task as `system_prompt` or woven into `prompt`. This has a practical consequence: changing a prompt means editing the taskset (or a file it reads) — the change is versioned with the environment, and every harness sees the same instruction.

If you expect to iterate on a prompt a lot, promote it to a config field so variants need no code edit (recall that a `vf.TasksetConfig` allows changing task fields without modifying environment code):

```python
class ReverseTextConfig(vf.TasksetConfig):
    system_prompt: str = SYSTEM
```

Now `--taskset.system-prompt "..."` or a `[taskset]` TOML entry selects the variant. This is the same task-field vs config-field boundary we saw in Guide 02, but applied to prompts.

## The iteration loop

Prompt evaluation is an A/B comparison, and it is only as trustworthy as what you hold fixed. The loop:

```bash
# baseline
prime eval run @ configs/04/wordle-eval.toml

# edit the taskset prompt text (or pass a --taskset.* override)

# variant
prime eval run @ configs/04/wordle-gepa-eval.toml
```

The eval config stays v1-native:

```toml
model = "openai/gpt-5.4-nano"
num_tasks = 20
num_rollouts = 1
max_turns = 6

[sampling]
max_tokens = 1024

[taskset]
id = "wordle"
num_tasks = 100

[harness]
id = "default"
```

Rules that keep the comparison honest:

- **Fix everything except the prompt**: same model, taskset, task indices, harness, runtime, and sampling. If two things changed, the delta tells you nothing. (Note the two configs in `configs/04/` differ in `sampling.temperature` — that pair compares sampling, so a prompt comparison should copy one config, not diff the two.)
- **Use enough rollouts to beat noise.** 20 tasks × 1 rollout is a smoke test; direction-of-effect only. Bump `num_tasks`/`num_rollouts` before believing a small delta.
- **Read traces, not just the mean.** A prompt change often fixes one failure mode and introduces another; the per-trace view (outlined in Guide 01) shows which.

Each run leaves its own `outputs/.../` folder with the resolved `config.toml`, so every variant is reproducible after the fact.

## The GEPA boundary

GEPA is verifiers' automated prompt optimizer: it evaluates a prompt, reflects on failures with a reflection model, and proposes revisions. **The GEPA CLI in this checkout still loads legacy v0 environments** through `load_environment` — it does not consume v1 tasksets yet. The files under `configs/gepa/` (and `configs/04/wordle-gepa.toml`) are kept only for that legacy workflow; do not use them as v1 taskset examples.

A native v1 prompt optimizer should run traces through the same v1 eval path and mutate prompt-bearing taskset config or packaged prompt files. That adapter is tracked in [v1 Authoring Gaps](../../reference/v1-authoring-gaps.md). Until it lands, the manual loop above — made cheap by config-field prompts — is the supported path.

## Try it

- Promote `reverse_text`'s `SYSTEM` constant to a config field and run two variants back-to-back with `--taskset.system-prompt`, keeping the config file untouched.
- Take the Wordle baseline, tighten the system prompt (e.g. demand an explicit candidate list before each guess), and measure the delta at `num_rollouts = 4`.



## Next

→ [07 — Judges and Instruction Following](../07-judges-and-instruction-following/README.md): when the reward itself needs a model — scoring semantics with an LLM judge.