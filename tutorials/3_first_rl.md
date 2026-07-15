# Your First RL Run

In Tutorial 2 you *measured* models. Now you'll *improve* one: take a small open model that is genuinely bad at a task, train it with reinforcement learning on hosted GPUs, and watch its reward climb in something close to real time. You don't need a GPU, and you don't need any RL background — both are covered.

You need the setup from [Tutorial 1](1_setup.md) and training credits on your account.

## Reinforcement learning in four sentences

The nice thing about environments is that the score isn't just a grade — it's a teaching signal. RL training runs in a loop: the model attempts a task (a rollout, exactly like in your evals), the environment scores it (the reward), and the trainer nudges the model's weights so that whatever produced *high* reward becomes more likely and whatever produced *low* reward becomes less likely. Repeat thousands of times and the model climbs the reward. That's it — everything else is engineering.

One refinement is worth knowing because you'll see it in the config. Modern LLM training (in the [GRPO](https://huggingface.co/learn/llm-course/en/chapter12/3b) family) doesn't judge a rollout in isolation — it samples a *group* of attempts at the **same** task (say, 8) and reinforces the ones that scored *above their group's average*. This has a practical consequence you can reason about: if all 8 attempts score identically — all perfect, or all hopeless — there's no above/below average, and that task teaches nothing this round. The task has to be *sometimes-solvable* for learning to happen.

## The task: reversing text

We'll train on `primeintellect/reverse-text` from the Environment Hub: given a string, write it backwards. It's a deliberately humble task with three properties that make it the perfect first RL run:

- **Trivially verifiable** — the correct answer is computable, so the reward is never wrong.
- **Small models are bad at it** — character-level manipulation is genuinely hard for small LLMs, so there's real headroom to climb.
- **The reward is smooth** — it scores *similarity* to the correct reversal, not just exact match. A model that gets 80% of characters right earns partial credit. Remember the sometimes-solvable requirement? A smooth reward means even a weak model's attempts differ from each other, so there is learning signal from step one. (An all-or-nothing reward on a task the model always fails would leave the trainer starving.)



## Step 1: generate a config

The CLI writes a starter config for any environment:

```bash
prime rl init primeintellect/reverse-text
```

Open the generated TOML. Trimmed to what matters today:

```toml
model = "Qwen/Qwen3.5-0.8B"      # the model whose weights we'll change
loss = "rl"
max_steps = 100                   # how many training steps to run

batch_size = 128                  # rollouts consumed per training step
rollouts_per_example = 8          # the "group of 8" from the explainer above

[sampling]
max_tokens = 2048                 # cap per model response during rollouts

[[env]]
id = "primeintellect/reverse-text"

[eval]                            # optional but do it: periodic held-out eval
interval = 25                     # every 25 steps...

[[eval.env]]
id = "primeintellect/reverse-text"
num_examples = 50                 # ...measure on 50 tasks
```

Read it top to bottom and notice the two layers:

- **Trainer knobs** at the top: the model, how long to train, how many rollouts feed each step. `Qwen3.5-0.8B` is a 0.8-billion-parameter model — small enough to train cheaply, weak enough at reversing text to visibly improve. (`prime rl models` lists everything trainable on the platform.)
- **The environment**, embedded under `[[env]]` — the very same environment definition your evals used. Nothing about the task or its scoring changes between evaluation and training; that is the whole design.

The `[eval]` block earns its keep: it evaluates the model on held-out tasks *before training starts* (your baseline, for free) and every 25 steps after. Reward-during-training tells you the model is improving on what it practices; the periodic eval tells you it actually got better.

## Step 2: launch

```bash
prime rl run reverse-text.toml
```

The CLI shows you what the run will cost and asks you to confirm. Then the platform takes over: it provisions GPUs, spins up the environment and the trainer, and starts the loop. You rent nothing and SSH nowhere.

## Step 3: watch it learn

Your run gets an ID (visible in `prime rl list`). The dashboard on [app.primeintellect.ai](https://app.primeintellect.ai) shows everything graphically, and the same telemetry is available from the terminal:

```bash
prime rl get <run-id>            # status at a glance
prime rl logs <run-id>           # live logs
prime rl metrics <run-id>        # reward curve & training stats
prime rl progress <run-id>       # step counter
prime rl rollouts <run-id>       # peek at actual rollouts, mid-training
prime rl distributions <run-id>  # how rewards are spread, per step
```

What you're hoping to see, and roughly will:

- **Mean training reward trending upward** — noisily! RL curves wobble; judge the trend over tens of steps, not step to step.
- **The eval score at step 0 vs later evals** — the honest before/after.
- **Reward distributions shifting right** — early on most rollouts score low with a few lucky ones (that spread is exactly what the trainer feeds on); later the mass moves toward high scores.

If you're curious what the model actually *says* at step 40, `prime rl rollouts` shows you — the same read-the-evidence habit from Tutorial 2, mid-training.

## Step 4: the result

When the run finishes, the platform saves the trained model weights — checkpoints along the way (see the `[checkpoints]` options in the generated config) and the final adapter at the end, listed by:

```bash
prime rl checkpoints <run-id>
```

You've closed the loop the whole Lab stack is built around: **evaluate → train on the same environment → evaluate again**. The number that told you the model was bad is the same number that proves it improved.

Housekeeping: `prime rl stop <run-id>` halts a run early (you keep its checkpoints); `prime rl delete <run-id>` removes a finished one.

## Things to try

- Halve `rollouts_per_example` to 4 and re-run: smaller groups are cheaper per step but give the trainer a noisier signal. Compare the two reward curves.
- Look at `prime rl distributions` at the first and last eval: the histogram tells the improvement story better than the mean does.
- Prefer running the trainer yourself on your own GPUs? The same config concepts drive the open-source [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) directly — [Guide 03](../guides/03-training-with-rl/README.md) walks that path, including how to validate an environment *before* burning GPU-hours on it.



## Recap

You learned the try→score→nudge loop, why groups of rollouts and sometimes-solvable tasks matter, launched a hosted training run from one TOML file, and read its progress with the same trace-level skepticism you learned for evals.

RL changed the weights. There's one more way to make a model better that doesn't touch weights at all —

Next → [4 — Your First GEPA Run](4_first_gepa.md): evolve the *prompt* instead.