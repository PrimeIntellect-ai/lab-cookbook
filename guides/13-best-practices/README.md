# Best Practices

A deliberate walkthrough of the habits that keep environments clean.

By now you've built and read several environments: a single-turn scorer, a multi-turn game, a judged task, a tool-using searcher, a sandboxed agent, a synthetic world. They look different, but the ones that stay maintainable share a handful of habits. This guide collects them in one place. None of it is new machinery — it's how to use what you've already seen well.

## Let the types carry your assumptions

Every environment works with a known set of shapes: a `vf.Task`, a `vf.State`, assistant messages that are strings, an answer that's a string. Write code that states those assumptions plainly and trusts them.

When you read text off a message, convert it where you use it:

```python
text = str(message.content or "")
```

That one line says "I expect text here, and I'm taking it as text." You'll be tempted to write a helper that walks every possible content shape, or to scatter `isinstance` checks for cases your environment never produces — resist it. A text-only task never receives image parts; a reward never receives a half-built state. Code written to handle shapes that can't occur hides what the environment actually assumes, and it ages badly. If a value genuinely arrives in more than one shape, narrow it once, explicitly, at the point it enters your code.

The payoff is that the type checker becomes a collaborator. When it complains, it's usually telling you a real thing about your assumptions — follow it rather than silencing it. Code that needs an `# ignore` comment to pass is almost always code that can be restructured to pass honestly.

## Prompts are content, not code

A prompt is text the model reads. Write it as text you can read: a module-level string constant, or a single template whose blanks are filled with task data.

```python
CRITIC_PROMPT = """Critique the argument below. Be specific.

Argument:
{argument}"""
```

```python
content = CRITIC_PROMPT.format(argument=argument)
```

You should avoid building prompts by stitching together helper functions that each return a fragment. When a prompt is assembled from `turn_contract()` plus `role_intro()` plus a suffix, no one can see what the model actually receives without chasing the pieces. If two prompts share a sentence, repeat the sentence — for prompts, being able to read the whole thing in one place is worth more than avoiding the duplication. Save interpolation for *data* (`{question}`, `{argument}`), not for composing the prompt's structure.

## The config is your control panel

Everything someone might want to tune lives on a typed config class — dataset name, splits, difficulty, judge model, sandbox image. That config is the surface users reach through `[eval.taskset]` and `[env.taskset]` in TOML, so it should hold plain, serializable values: strings, numbers, booleans, nested config objects. Store the *name* of an environment variable, never the secret itself:

```python
class MyTasksetConfig(vf.TasksetConfig):
    judge_model: str = "openai/gpt-4.1-mini"
    judge_api_key_var: str = "PRIME_API_KEY"
```

When a field's value can vary by run, it belongs on the config. When it's the same for every run, it can be a constant in the module. The test is whether a user would ever want to change it without editing code.

## Reach for subclasses only when you need them

Most environments are a `Taskset` subclass plus the standard loaders. You usually do not need a custom `Harness` — the default one runs the rollout loop, calls tools, and hands results to your rewards. Add a custom harness only when the package genuinely owns a reusable way of *executing* rollouts, like wrapping an external agent runtime (guide 12).

The same restraint applies to wiring. When you need a custom `vf.User` or a specific component, the typed loader is the clean way to select it:

```python
def load_user(self, config: vf.UserConfig) -> MyUser:
    return MyUser(config=config)
```

That reads better than indirect tricks to make a framework registry pick your class for you. Say what you want directly.

## Rewards score, metrics observe

Two kinds of signal come out of a rollout, and they're declared differently. A **reward** contributes to the number the model is trained against:

```python
@vf.reward(weight=1.0)
async def correct_answer(self, task: vf.Task, state: vf.State) -> float:
    ...
```

A **metric** is something you want to watch but not optimize — how many turns the agent took, whether it submitted, how close it got to the oracle:

```python
@vf.metric
async def submitted(self, state: vf.State) -> float:
    ...
```

Reach for `@vf.metric` for anything diagnostic. Metrics show up next to the reward in eval output and during training, and they're the fastest way to catch a reward that's being gamed — if the reward climbs while a metric you expected to move stays flat, something is off. Each reward and metric should ask for only the rollout data it actually reads (`task`, `state`, `answer`, …); declaring a parameter you don't use is a small signal that the function and its intent have drifted apart.

## Lifecycle hooks follow the rollout loop

The [rollout loop from guide 04](../04-prompt-optimization/README.md#how-a-multi-turn-rollout-runs) has natural moments, and each has a hook:

- `@vf.setup` runs at the **start** of a rollout — initialize per-rollout bookkeeping here.
- `@vf.stop` runs after each model turn and decides when the rollout **ends**.
- `@vf.cleanup` runs at the **end** — release per-rollout resources.
- `@vf.teardown` runs once when the whole environment shuts down.

Use these for state changes at the boundaries of a rollout. Mid-rollout, your tools change state through the `task` and `state` the framework hands them. Keeping setup in `@vf.setup` rather than lazily inside a tool's first call makes the rollout's starting state obvious in one place, and keeps cleanup honest — write cleanup so it tolerates a rollout that was cancelled partway.

## Keep tools small and let the framework inject

A tool is a method on the taskset, exposed through a `vf.Toolset`. Its signature is its contract: the parameters the model fills (`query: str`) plus the ones the framework injects by name (`task`, `state`, a `sandbox` when the toolset has one). The injected ones never appear in what the model sees, so you get clean tool schemas for free — declare exactly the injected values the tool uses, and nothing else. Because one toolset instance is shared across all concurrent rollouts, anything per-rollout comes from the injected `state`, never from an attribute you set on the toolset.

## Validate by running

An environment isn't done when it imports — it's done when it installs, loads, and produces sensible rollouts. Run a small eval and read a few of them:

```bash
prime eval run <env> -m openai/gpt-4.1-mini -n 5
```

Read the rollouts, not just the average. Check that the prompt looks right, the reward matches your judgment, and the spread isn't all-zero or all-one. Most environment bugs are visible in three rollouts and invisible in a single aggregate number.

## Next

In [Legacy Environments](../14-legacy-environments/README.md), see the older Rubric and `source()` patterns you'll encounter in unmigrated Hub packages, and how they map to everything above.
