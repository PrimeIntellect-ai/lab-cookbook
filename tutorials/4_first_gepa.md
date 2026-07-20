# Your First GEPA Run

Tutorial 3 improved a model by training its weights. That required an open-weights model and GPU resources. But wif the model is closed, such as models in the GPT, Claude, Gemini, Grok series, or you just want improvement without touching the weights of a model, then you optimize the other thing the score depends on: **the prompt**. In this tutorial you'll run [GEPA](https://arxiv.org/abs/2507.19457), an algorithm that evolves a better prompt automatically.

You need the setup from [Tutorial 1](1_setup.md). Everything runs on inference credits, so no GPUs are needed here.

## Training without weights

You've probably experienced prompt engineering by hand, where you've tweaked a system prompt, eyeballed a few responses, and repeated the procedure. This is a slow and unsystematic procedure, as it takes time to manually update parts of a long prompt, see the differential effects on performance, and know when to stop tuning.

`verifiers` enfironments already comes with the ability to score rollouts, so that means that the effectiveness of different prompts can be compared. Prompt optimization is just a search over prompts, with your environment as the judge. The only missing ingredient is something to *propose* better prompts than we could do manually. This is what GEPA (Genetic-Pareto prompt evolution) does -- it automates the loop you'd do by hand. GEPA works as follows:

1. **Evaluate** the current prompt on a small batch of tasks.
2. **Reflect**, where a second LLM (the *reflection model*) reads the actual failed attempts and writes a diagnosis, such as *"the model keeps repeating letters it already ruled out; the instructions never tell it to track them."*.
3. **Propose** a revised prompt based on that diagnosis.
4. **Select**: GEPA keeps every prompt that is best at *something* — the [Pareto frontier](https://en.wikipedia.org/wiki/Pareto_front). A prompt that fixes hard tasks but slips on easy ones survives alongside the all-rounder, and their strengths can be combined later.
5. Repeat until the evaluation budget runs out, after which the winner is validated on held-out tasks.

Notice this is the same shape as RL — try, score, improve, repeat — with prompts evolving instead of weights, and reflection standing in for gradients.

## Applying it to the Wordle task

We'll optimize a prompt for Wordle, the word game from Tutorial 2. It's a great subject because strategy lives in the prompt: a model told merely "play Wordle" plays sloppily, while a more strategic prompt that instructs how to track confirmed letters, eliminate impossible ones, and choose information-rich guesses plays measurably better.

## Run it

```bash
uv run gepa wordle_v1 \
  --model openai/gpt-5.4-nano \
  --reflection-model openai/gpt-5.4 \
  --num-train 50 --num-val 50 \
  --max-turns 6 \
  --max-total-rollouts 200 \
  --max-concurrent 32 \
  --sampling.max-tokens 1024
```

What each choice means:

| Flag | Meaning |
| --- | --- |
| `wordle_v1` | The local environment whose reward we're optimizing against. |
| `--model` | The model that *plays* — the one your optimized prompt is for. Optimize for the model you'll actually use: prompts don't transfer perfectly between models. |
| `--reflection-model` | The model that *diagnoses failures and writes new prompts*. This is the thinking-heavy job, so a stronger model here is money well spent — it runs far less often than the player. |
| `--num-train 100` / `--num-val 50` | 100 tasks to optimize against, 50 held out for honest validation. The split matters: a prompt can overfit to its training tasks just like a model can, and the validation set is what catches it. |
| `--max-total-rollouts 500` | **Your budget.** This caps all optimization rollouts at 500. More budget means more evolution steps, usually with diminishing returns. |
| `--max-concurrent 32` | Keep at most 32 rollouts in flight. |
| `--sampling.max-tokens 1024` | Cap each player-model response at 1024 tokens. |

The current GEPA CLI uses the same v1 taskset and harness loader as eval, so local tasksets work directly and their knobs use the same `--taskset.*` flags. A config-file form also exists — see `configs/04/wordle-gepa.toml`.

While it runs, you'll see generations of candidate prompts being evaluated, each with its score. This takes a while — you budgeted 500 rollouts, after all.

## Read the result

The run writes to the same v1 output shape as eval: `outputs/<taskset>--<model>--<harness>/<run-id>/`. It contains `config.toml`, streamed `traces.jsonl`, and GEPA's `candidates.json` and `run_log.json`.

Open the winning prompt and *read it* — this is the part no weight update can offer. You will usually find the reflection model has articulated genuine strategy: rules about tracking eliminated letters, instructions about guess format, reminders the base prompt never had. It's an artifact you can review like code, edit further by hand, and check into version control.

Then close the loop the honest way, like every tutorial before this one: take the optimized prompt, run a plain eval with it against the baseline prompt — same model, same tasks, same sampling — and let the environment tell you what it's worth.

## Weights or words?

You now have two levers and a sense of when to pull each:

| | RL training (Tutorial 3) | GEPA (this tutorial) |
| --- | --- | --- |
| Changes | model weights | the prompt |
| Needs | open-weights model, GPU time | any model you can call, inference credits |
| Result | a checkpoint | a human-readable prompt |
| Best when | you own the model and want deep, durable gains | the model is closed, budget is small, or you want interpretable wins *now* |

They compose, too: a well-optimized prompt raises the reward floor an RL run starts from.

## Things to try

- Re-run with a *weaker* reflection model and compare winning prompts — you'll see, concretely, why the diagnosis step deserves the strong model.
- Raise `--max-total-rollouts` to 1000 on a cheap player model. Where do the gains flatten?
- Point GEPA at a different Hub environment and see what strategy it discovers — the algorithm doesn't know anything about Wordle; everything it learned, it learned from your environment's reward.

## Recap — and where to next

Across these tutorials you built one complete mental model: environments package tasks with scoring; evals measure models against them; the same scores train weights (RL) or evolve prompts (GEPA); and in every case, you verified claims by reading traces, not trusting means.

The natural next step is building an environment of *your own* — your task, your scoring — because everything you just did works instantly on it. Start with [Build Your First Environment](5_build_first_environment.md), and continue through the [Ramping up series](README.md#ramping-up) for the rest of the toolbox.
