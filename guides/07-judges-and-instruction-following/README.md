# Judges and Instruction Following

This guide uses [simple-judge](../../environments/simple_judge/simple_judge.py), a local taskset with one yes/no judge criterion per task. It's the smallest useful pattern for adding LLM-judged instruction following.

## Eval

```bash
prime eval run prime/simple-judge -m openai/gpt-4.1-mini -n 6 -r 2
```

Or via config:

```toml
# [configs/07/simple-judge-eval.toml](../../configs/07/simple-judge-eval.toml)
model = "openai/gpt-4.1-mini"
save_results = true

[[eval]]
env_id = "prime/simple-judge"
num_examples = 6
rollouts_per_example = 2
sampling_args = { max_tokens = 256 }
```

## How Simple-Judge Is Built

[environments/simple_judge/simple_judge.py](../../environments/simple_judge/simple_judge.py) is one file with three pieces: a few toy tasks, a config that holds judge endpoint settings, and a taskset with one reward that calls the judge.

**Tasks.** A module-level list of dicts. Each task is just a prompt and a one-line `criterion` stored under `info`:

```python
TOY_TASKS: list[vf.JsonData] = [
    {
        "prompt": [{"role": "user", "content": "Write one cheerful sentence about mornings."}],
        "info": {"criterion": "The response sounds upbeat and enthusiastic."},
    },
    ...
]
```

`task["prompt"]` is a list of messages (no system message — system prompts are config-owned, never in task rows). `task["info"]` is the recognized field for per-task structured data, and the judge reads `criterion` off it.

**Config.** Judge endpoint settings live on `TasksetConfig`, not on the reward function or buried in a global:

```python
class SimpleJudgeTasksetConfig(vf.TasksetConfig):
    judge_model: str = "openai/gpt-4.1-mini"
    judge_base_url: str = "https://api.pinference.ai/api/v1"
    judge_api_key_var: str = "PRIME_API_KEY"
    system_prompt: vf.SystemPrompt = (
        "Follow the user instruction carefully. Keep answers short."
    )
```

The env-var name is on the config, not the key value — secrets stay in the environment. Override any of these per run from `[eval.taskset]` in TOML:

```toml
[[eval]]
env_id = "prime/simple-judge"

[eval.taskset]
judge_model = "openai/gpt-5-mini"
```

**Taskset and reward.** `load_tasks` just returns the static list. The reward is one async method. Rewards run *after* the rollout loop finishes — they read the completed `state["completion"]`, not a live conversation — which is why a reward is free to do its own slow work like calling a judge model:

```python
class SimpleJudgeTaskset(vf.Taskset[SimpleJudgeTasksetConfig]):
    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        _ = split
        return TOY_TASKS

    @vf.reward(weight=1.0)
    async def judge_reward(self, task: vf.Task, state: vf.State) -> float:
        vf.ensure_keys([self.config.judge_api_key_var])
        assistant_messages = vf.get_messages(state.get("completion") or [], role="assistant")
        response_text = str(assistant_messages[-1].content or "") if assistant_messages else ""
        user_messages = vf.get_messages(task.get("prompt") or [], role="user")
        user_message = str(user_messages[-1].content or "") if user_messages else ""
        info = task.get("info") or {}
        criterion = str(info.get("criterion") or "")
        judge = AsyncOpenAI(
            api_key=os.getenv(self.config.judge_api_key_var, ""),
            base_url=self.config.judge_base_url,
        )
        try:
            response = await judge.chat.completions.create(
                model=self.config.judge_model,
                messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                    criterion=criterion, user_message=user_message, response=response_text,
                )}],
            )
        finally:
            await judge.close()
        text = response.choices[0].message.content or ""
        return 1.0 if "yes" in text.lower() else 0.0
```

Three things worth copying:

- **Validate keys at use time.** `vf.ensure_keys([self.config.judge_api_key_var])` raises a clean `MissingKeyError` listing the missing env var instead of letting the request fail mid-rollout with a confusing 401.
- **Build a dedicated `AsyncOpenAI` client per call and close it.** The reward owns the judge connection; the rollout's own model client is separate. Sharing them would couple environment scoring to the model under test.
- **Parse the verdict deterministically.** A one-word `yes`/`no` instruction in `JUDGE_PROMPT` plus a `"yes" in text.lower()` check keeps the reward binary and reproducible. If the judge ever rambles, the prompt is the place to tighten.

**Loaders.** The standard shape — no custom harness, no extra root args:

```python
def load_taskset(config: SimpleJudgeTasksetConfig) -> SimpleJudgeTaskset:
    return SimpleJudgeTaskset(config=config)


def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(
        taskset=vf.load_taskset(config=config.taskset),
        harness=vf.load_harness(config=config.harness),
    )
```

## Next

[Tool Use and Search](../08-tool-use-and-search/README.md) layers the same direct-judge wiring on top of retrieval tools in `wiki-search`.

Unmigrated Hub packages (`JudgeRubric`, `source()`, etc.): [Legacy Environments](../14-legacy-environments/README.md).
