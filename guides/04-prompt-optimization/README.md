# Prompt Optimization

For v1 tasksets, prompts are task data. Put static instructions in constants or packaged text files, then attach them to each `vf.Task` as `system_prompt` or as part of `prompt`.

A tight prompt iteration loop is:

```bash
uv run eval @ configs/04/wordle-eval.toml
# edit the taskset prompt text
uv run eval @ configs/04/wordle-gepa-eval.toml
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

## GEPA Boundary

The GEPA CLI in this checkout still loads legacy v0 environments through `load_environment`. The files under `configs/gepa/` are kept only for that legacy workflow. Do not use them as v1 taskset examples.

A native v1 prompt optimizer should build on `vf.Environment`, run traces through the same v1 eval path, and mutate prompt-bearing taskset config or packaged prompt files. That adapter is called out in [v1 Authoring Gaps](../../reference/v1-authoring-gaps.md).
