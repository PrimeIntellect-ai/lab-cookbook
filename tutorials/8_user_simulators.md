# User Simulators

Multi-turn tasks need someone on the other side of the conversation. In Wordle ([tutorial 2](2_first_eval.md)), that's game logic; in real assistant tasks, it's a *person* — who answers follow-up questions, reveals information gradually, and eventually says "go ahead." A **user simulator** is the environment playing that person: a server that produces the user's turns, driven by per-task ground truth the model can't see.

The example is the cookbook's v1 `shape-detective` taskset: the model sees a 4×4 grid of colored shapes and must identify a hidden target tile. The simulated user reveals one clue per turn.

**You need:** [Build Your First Environment](5_build_first_environment.md).

## The contract

A user simulator is a `vf.User` server — the same server pattern as a toolset, but instead of exposing tools it produces the conversation's user messages (`environments/shape_detective/shape_detective/servers/user.py`, abbreviated):

```python
class ShapeDetectiveState(vf.State):
    clue_index: int = 1
    user_finished: bool = False


class ShapeDetectiveUser(vf.User[vf.UserConfig, ShapeDetectiveState]):
    async def setup_task(self, task: ShapeDetectiveTaskData) -> None:
        self.target_tile = task.target_tile      # per-task ground truth

    async def respond(self, message: str) -> vf.Messages:
        if self.state.clue_index == 1:
            self.state.clue_index = 2
            return [vf.UserMessage(content=f"{clue_line(...)}\n\nNarrow the candidates...")]
        if self.state.clue_index == 2:
            self.state.clue_index = 3
            return [vf.UserMessage(content="...Commit your answer now as \\boxed{N}.")]
        self.state.user_finished = True
        return []


if __name__ == "__main__":
    ShapeDetectiveUser.run()
```

The contract in three parts:

- **`setup_task(task)`** receives the typed `ShapeDetectiveTaskData` row, not the behavior object. Here it reads the hidden target so it can phrase truthful clues.
- **`respond(message)`** is called after each assistant turn and returns typed messages such as `vf.UserMessage`. When the script is exhausted it sets shared state and returns `[]`.
- **Turn progress lives in `self.state`**, a typed `vf.State`: serializable, per-rollout, and visible to rewards on `trace.state`. It's the same state channel toolsets use ([Tool Use and Search](10_tools.md)) — which also means a simulator and the rewards can share state: a simulator that tracks "did the model ask before acting?" makes that judgment scoreable.

This simulator is deliberately scripted — fixed clues, fixed order — which keeps it deterministic and free. When the user's side needs to be *adaptive* (a customer who answers arbitrary questions), `respond` can call an LLM instead; you then own the same calibration duties as with a [judge](6_judges.md), plus the cost per turn.

## Wiring it into the task

In v1, the task class declares its user server. The taskset's `load()` method supplies immutable rows:

```python
class ShapeDetectiveTaskData(vf.TaskData):
    answer: str
    mode: Mode
    target_tile: Tile


class ShapeDetectiveTaskConfig(vf.TaskConfig):
    user: vf.UserConfig = vf.UserConfig()


class ShapeDetectiveTask(
    vf.Task[ShapeDetectiveTaskData, ShapeDetectiveState, ShapeDetectiveTaskConfig]
):
    user = ShapeDetectiveUser

    @vf.stop
    async def user_finished(self, trace: vf.Trace) -> bool:
        return trace.state.user_finished

    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace) -> float:
        return float(extract_boxed(trace.last_reply) == self.data.answer)


class ShapeDetectiveConfig(vf.TasksetConfig):
    num_tasks: int = 12
    seed: int = 0
    task: ShapeDetectiveTaskConfig = ShapeDetectiveTaskConfig()


class ShapeDetectiveTaskset(vf.Taskset[ShapeDetectiveTask, ShapeDetectiveConfig]):
    def load(self) -> list[ShapeDetectiveTask]:
        return [
            ShapeDetectiveTask(build_data(i), self.config.task)
            for i in range(self.config.num_tasks)
        ]
```

The shipped `mode = "single"` variant uses the same task class. Its user server marks `user_finished` after the first answer, while the initial prompt carries all three clues.

One capability check: user simulation is a harness feature, advertised as `SUPPORTS_USER_SIM`. The built-in `default` harness supports it; many CLI-agent harnesses do not — they own their loop and don't expect an environment-driven user.

## Run it

User placement stays under `taskset.task.user`:

```toml
model = "openai/gpt-5.4-nano"
num_tasks = 6
num_rollouts = 2
max_turns = 6                   # 3 clue exchanges fit comfortably

[sampling]
max_tokens = 1024

[taskset]
id = "shape-detective"
num_tasks = 12
seed = 0

[taskset.task.user]
colocated = false

[harness]
id = "default"
```

```bash
uv run eval @ configs/09/shape-detective-eval.toml
```

In `traces.jsonl`, typed `vf.UserMessage` records show the simulator turns, while `trace.last_reply` is the final answer seen by the reward. The transcript reveals whether the model used the clues or guessed early.

## Try it

- Expose a separate single-turn taskset with all clues in its initial prompt, then compare solve rates.
- Make the simulator adversarial: have `respond` answer a direct question from the model ("is it striped?") only if the model listed candidates in its previous turn — a simulator that *rewards* good conversational behavior with information.
- Move the commit-check into shared state: have the simulator record `asked_before_committing` in `self.state` and add a small shaping reward on it ([Designing Rewards](7_rewards.md)).

## Next

→ [Multimodal Environments](9_multimodal.md): the other half of shape-detective — the same game's tasks now carry the *image* the clues describe.
