# User Simulators

Multi-turn tasks need someone on the other side of the conversation. In Wordle ([tutorial 2](2_first_eval.md)), that's game logic; in real assistant tasks, it's a *person* — who answers follow-up questions, reveals information gradually, and eventually says "go ahead." A **user simulator** is the environment playing that person: a server that produces the user's turns, driven by per-task ground truth the model can't see.

The example is `shape-detective`: the model sees a 4×4 grid of colored shapes (it's an image task — [Multimodal Environments](10_multimodal.md) covers that half) and must identify a hidden target tile. The simulated user reveals one clue per turn — pattern, then color, then shape — and only then asks the model to commit an answer.

**You need:** [Build Your First Environment](5_build_first_environment.md).

## The contract

A user simulator is a `vf.User` server — the same server pattern as a toolset, but instead of exposing tools it produces the conversation's user messages (`environments/shape_detective/shape_detective.py`, abbreviated):

```python
class ShapeDetectiveState(vf.State):
    clue_index: int = 1


class ShapeDetectiveUser(vf.User[vf.UserConfig, ShapeDetectiveState]):
    async def setup_task(self, task: ShapeDetectiveTask) -> None:
        self.target_tile = task.target_tile      # per-task ground truth

    async def respond(self, message: str) -> vf.Messages:
        if self.state.clue_index == 1:
            self.state.clue_index = 2
            return [vf.UserMessage(content=f"{clue_line(...)}\n\nNarrow the candidates...")]
        if self.state.clue_index == 2:
            self.state.clue_index = 3
            return [vf.UserMessage(content="...Commit your answer now as \\boxed{N}.")]
        return []                                 # nothing left to say → rollout ends


if __name__ == "__main__":
    ShapeDetectiveUser.run()
```

The contract in three parts:

- **`setup_task(task)`** hands the simulator its per-task script — here, which tile is the target, so it can phrase truthful clues. This is the information asymmetry that makes the task work: the simulator knows the answer; the model has to earn it.
- **`respond(message)`** is called after each assistant turn and returns the next user message(s). Returning `[]` means the user has nothing more to say, which ends the conversation — a natural stop condition without any `@vf.stop`.
- **Turn progress lives in `self.state`**, a typed `vf.State`: serializable, per-rollout, and visible to rewards on `trace.state`. It's the same state channel toolsets use ([Tool Use and Search](9_tools.md)) — which also means a simulator and the rewards can share state: a simulator that tracks "did the model ask before acting?" makes that judgment scoreable.

This simulator is deliberately scripted — fixed clues, fixed order — which keeps it deterministic and free. When the user's side needs to be *adaptive* (a customer who answers arbitrary questions), `respond` can call an LLM instead; you then own the same calibration duties as with a [judge](6_judges.md), plus the cost per turn.

## Wiring it in — per task

The taskset decides, task by task, whether a simulator drives the conversation:

```python
class ShapeDetectiveConfig(vf.TasksetConfig):
    mode: Mode = "multi"            # "single": all clues upfront, one turn
    user: vf.UserConfig = vf.UserConfig()

class ShapeDetectiveTaskset(vf.Taskset[...]):
    def user(self, task: ShapeDetectiveTask) -> vf.User | None:
        if task.mode == "single":
            return None             # no simulator: single-turn eval
        return ShapeDetectiveUser(self.config.user)
```

Returning `None` means no simulator — which is how one environment cleanly supports both a single-turn and a conversational variant of the same tasks, switched by config.

One capability check: user simulation is a harness feature, advertised as `SUPPORTS_USER_SIM`. The built-in `default` harness supports it; many CLI-agent harnesses do not — they own their loop and don't expect an environment-driven user.

## Run it

`configs/09/shape-detective-eval.toml` selects multi-turn mode:

```toml
model = "openai/gpt-5.4-nano"
num_tasks = 6
num_rollouts = 2
max_turns = 6                   # 3 clue exchanges fit comfortably

[sampling]
max_tokens = 1024

[taskset]
id = "shape-detective"
mode = "multi"
num_tasks = 12
seed = 0

[harness]
id = "default"
```

```bash
prime eval run @ configs/09/shape-detective-eval.toml
```

In the traces, watch the conversation's rhythm: the model narrows candidates after each scripted clue, and the simulator's final turn forces the commit. The reward only checks the final `\boxed{N}` — but the *transcript* shows you whether the model actually used the clues or guessed early, which is the diagnostic a single-turn task can never give you.

## Try it

- Run the same config with `--taskset.mode single` and compare solve rates: how much does distributing the clues across turns actually cost the model?
- Make the simulator adversarial: have `respond` answer a direct question from the model ("is it striped?") only if the model listed candidates in its previous turn — a simulator that *rewards* good conversational behavior with information.
- Move the commit-check into shared state: have the simulator record `asked_before_committing` in `self.state` and add a small shaping reward on it ([Designing Rewards](7_rewards.md)).

## Next

→ [Tool Use and Search](9_tools.md): the other interaction primitive — toolsets the model calls, and how to read what it did with them.
