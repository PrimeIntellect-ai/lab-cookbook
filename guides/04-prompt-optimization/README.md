# Prompt Optimization

Improve an environment's system prompt with GEPA before you spend compute changing model weights.

GEPA is a prompt optimizer. It runs the environment, reads the reward each rollout earns, reflects on where the model failed, and proposes a revised system prompt — repeating until the prompt stops improving. It is the right first move when the environment already scores rollouts sensibly but the model's behavior depends heavily on the system prompt, tool instructions, output format, or task strategy. RL changes the model's weights; GEPA changes the text you send it, which is far cheaper.

This guide uses the local `wordle` environment, also published as [prime/wordle](https://app.primeintellect.ai/dashboard/environments/prime/wordle). Wordle is a good subject because every rollout is easy to read: the model sees the game state, makes a guess, gets letter-by-letter feedback, and is scored on whether it solves the word within six turns.

## How Wordle Is Built

[environments/wordle/wordle.py](../../environments/wordle/wordle.py) builds on `tasksets.textarena`, a published wrapper around the [TextArena](https://github.com/LeonGuertler/TextArena) game library. The wrapper already knows how to run a TextArena game as a multi-turn rollout; wordle supplies the game name, the prompts, the feedback formatting, and the rewards.

**Config.** One Pydantic class sets wordle's three task fields. `system_prompt = None` means "no inline default" — the taskset loads the prompt from a file instead, which the section below explains:

```python
class WordleTasksetConfig(TextArenaTasksetConfig):
    game: str = "Wordle-v0"
    answer_state_key: str = "secret_word"
    system_prompt: vf.SystemPrompt = None
```

**User.** A `vf.User` is the simulated counterpart that replies to the model between its turns — here, the Wordle game itself. After each guess, TextArena emits a verbose `[GAME] Feedback: ...` string; `WordleUser.get_response` trims it down to just the grading row so the model sees clean feedback:

```python
class WordleUser(TextArenaUser):
    async def get_response(
        self, task: vf.Task, state: vf.State, messages: list[vf.Message]
    ) -> list[vf.UserMessage]:
        response = await super().get_response(task, state, messages)
        if state.get("done") is True:
            return response
        if not response:
            return []
        content = str(response[-1].content or "")
        latest_feedback = content.split("[GAME]")[-1].strip()
        if "Feedback:" in latest_feedback:
            latest_feedback = latest_feedback.split("Feedback:")[-1]
        return [vf.UserMessage(content=latest_feedback)]
```

`str(response[-1].content or "")` is the explicit transformation from the framework's `str | list[ContentPart]` union down to the `str` wordle works with. No assert, no helper, no cast — just an inline coercion that satisfies the type checker and reflects the wordle assumption (text-only).

**Taskset.** `WordleTaskset` subclasses `TextArenaTaskset[WordleTasksetConfig]` to inherit the task generation loop, then adds the prompt loader, the user override, and the rewards:

```python
class WordleTaskset(TextArenaTaskset[WordleTasksetConfig]):
    guess_pattern = r"<guess>(.*?)</guess>"
    config: WordleTasksetConfig

    def load_user(self, config: vf.UserConfig) -> WordleUser:
        return WordleUser(config=config)

    def load_system_prompt(self, config: WordleTasksetConfig) -> vf.SystemPrompt:
        if config.system_prompt is not None:
            return config.system_prompt
        return vf.SystemPromptConfig(path="prompts/system_prompt.txt")

    def guesses(self, content: str) -> list[str]:
        return re.findall(self.guess_pattern, content, re.DOTALL)
```

`load_user` tells the taskset which `vf.User` class to use: it constructs `WordleUser` directly, so the trimmed-feedback behavior above is what runs.

`load_system_prompt` decides where the system prompt text comes from. If a config or config file set `system_prompt`, it uses that. Otherwise it returns `vf.SystemPromptConfig(path="prompts/system_prompt.txt")`, which loads the prompt from a text file shipped inside the environment package. This file-backed default is what makes the prompt optimizable: GEPA writes its improved prompt to that same file (shown later), and because the taskset reads the file on every run, the new prompt takes effect on the next eval with no code change.

**Rewards.** Four `@vf.reward` methods on the taskset, weighted to make correctness dominate while still rewarding partial progress:

```python
@vf.reward(weight=1.0)
async def correct_answer(self, task: vf.Task, state: vf.State) -> float:
    answer = str(task["answer"])
    completion = state.get("completion") or []
    for message in reversed(vf.get_messages(completion, role="assistant")):
        matches = self.guesses(str(message.content or ""))
        if matches:
            return 1.0 if matches[-1].strip() == f"[{answer}]" else 0.0
    return 0.0
```

The other three (`length_bonus`, `partial_answer`, `format_reward`) follow the same shape. Each method requests only the rollout data it reads: `state` always, `task` when it needs the secret word. The framework injects whatever the signature names, so a reward that never reads `task` should not declare it.

`str(task["answer"])` and `str(message.content or "")` are the same inline coercion the user method uses — framework data narrowed to `str` at the point of use, with no assert, helper, or cast.

**Loaders.** The bottom of the file is the canonical loader shape — typed child loaders and a `vf.EnvConfig` root loader, nothing else:

```python
def load_taskset(config: WordleTasksetConfig) -> WordleTaskset:
    return WordleTaskset(config=config)


def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(
        taskset=vf.load_taskset(config=config.taskset),
        harness=vf.load_harness(config=config.harness),
    )
```

No custom `load_harness` because wordle uses the framework default. The taskset's typed config flows through `[eval.taskset]` and `[env.taskset]` blocks in TOML; nothing in the framework is replaced.

## How a Multi-Turn Rollout Runs

Reverse-text and GSM8K were single-turn: the model answered once and the environment scored it. Wordle is the first environment here that goes back and forth, so it's worth seeing the loop the harness actually runs. Every rollout follows the same shape:

1. **Setup** — the harness prepares per-rollout state and resolves the system prompt (next section).
2. **Loop**, until a stop condition fires:
   - send the conversation so far to the model and get one response;
   - check the stop conditions (answer submitted, max turns reached, error, …);
   - if none fired, the environment produces a reply and appends it as the next user turn, then the loop repeats.
3. **Render** — the finished conversation is assembled into `state["completion"]`.
4. **Cleanup** — per-rollout resources are released.

For wordle, step 2's "environment produces a reply" is `WordleUser.get_response`: after each guess the Wordle game returns letter feedback, which becomes the next user message. The loop ends when the word is solved or six guesses are used. A single-turn environment is just this loop with one model turn and no `get_response`.

Each phase of this loop has a hook the taskset can attach to — `@vf.setup` for step 1, `@vf.stop` for the stop checks, `@vf.cleanup` for step 4 — and rewards run after the loop, over the finished `completion`. Wordle only needs the user and the rewards; later guides introduce the other hooks one at a time as environments need them.

## System Prompt Resolution

Wordle's `load_system_prompt` returns the prompt for the *taskset side*. There are two sides in every rollout:

- **Taskset side** — task policy: `TasksetConfig.system_prompt` (or a per-task `task["system_prompt"]` override). This is what `WordleTaskset.load_system_prompt` supplies.
- **Harness side** — execution policy: `HarnessConfig.system_prompt`. Wordle leaves this unset.

`HarnessConfig.system_prompt_strategy` decides how the two combine. The default is `HT` (harness messages, then taskset). Other values: `TH` (taskset first), `H` or `T` (one side only), `H_OR_T` / `T_OR_H` (first non-empty side), `REJECT` (error if both are set, forcing a choice). Wordle sets only the taskset side, so under the default `HT` the resolved system prompt is just the wordle prompt.

The author never assembles the final system message — the harness resolves the strategy and injects it during setup. Your job is to put task instructions on the taskset side, execution instructions on the harness side, and pick a strategy if both are in play.

## Check the Baseline

Run a small eval first:

```bash
prime eval run prime/wordle \
  -m openai/gpt-5.4-nano \
  -n 20 \
  -r 1 \
  -t 1024
```

Or run with a config file:

```toml
# [configs/04/wordle-eval.toml](../../configs/04/wordle-eval.toml)
model = "openai/gpt-5.4-nano"
save_results = true

[[eval]]
env_id = "prime/wordle"
num_examples = 20
rollouts_per_example = 1
sampling_args = { max_tokens = 1024 }

[eval.taskset]
num_train_examples = 100
num_eval_examples = 20

[eval.harness]
max_turns = 6
```

```bash
prime eval run configs/04/wordle-eval.toml
```

GEPA is most useful when the model is trying the task but needs better guidance. If the scoring is broken, the task is impossible, or the model cannot follow the environment loop at all, fix that before optimizing the prompt.

## Run GEPA

Run GEPA with a config file:

```toml
# [configs/04/wordle-gepa.toml](../../configs/04/wordle-gepa.toml)
model = "openai/gpt-5.4-nano"
reflection_model = "openai/gpt-5.4-nano"
save_to_environment = true

[[env]]
env_id = "prime/wordle"

[env.taskset]
num_train_examples = 100
num_eval_examples = 50

[env.harness]
max_turns = 6

[gepa]
max_calls = 500
num_train = 100
num_val = 50
minibatch_size = 3
max_concurrent = 32

[sampling]
max_tokens = 1024
```

```bash
prime gepa run configs/04/wordle-gepa.toml
```

GEPA evaluates prompt candidates against environment rewards and prints optimization progress as it runs. The most important artifact is `system_prompt.txt` in the results directory — the optimized prompt.

With `save_to_environment = true`, GEPA also writes that prompt into the local environment's `prompts/` folder. `WordleTaskset.load_system_prompt` reads `prompts/system_prompt.txt`, so the saved prompt becomes the environment default on the next eval.

## Evaluate the Optimized Prompt

Run the same eval shape with the optimized prompt:

```bash
prime eval run prime/wordle \
  -m openai/gpt-5.4-nano \
  -n 20 \
  -r 1 \
  -t 1024
```

Or run with a config file:

```toml
# [configs/04/wordle-gepa-eval.toml](../../configs/04/wordle-gepa-eval.toml)
model = "openai/gpt-5.4-nano"
save_results = true

[[eval]]
env_id = "prime/wordle"
num_examples = 20
rollouts_per_example = 1
sampling_args = { max_tokens = 1024, temperature = 0.7 }

[eval.taskset]
num_train_examples = 100
num_eval_examples = 20

[eval.harness]
max_turns = 6
```

```bash
prime eval run configs/04/wordle-gepa-eval.toml
```

Keep the model, sample count, rollout count, and sampling settings fixed while
comparing prompts. The only thing that should differ between runs is the prompt
file loaded by `WordleTaskset.load_system_prompt`.

## Next

In [Warm Starts with SFT](../05-warm-starts-with-sft/README.md), you will use SFT to give a model a stronger starting policy before further RL.
