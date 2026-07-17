# Your First RL Run

In the previous tutorial you measured models. Now you'll improve one, by taking a small open model that is bad at a task initially, training it with reinforcement learning on hosted GPUs, and watching its ability climb almost in real time. You don't need a GPU, and you don't need any RL background.

You need the setup from [Tutorial 1](1_setup.md) and training credits on your account.

RL works as follows: the model attempts a task (a rollout, exactly like in an eval), the environment scores it (the reward), and the trainer nudges the model's weights so that whatever produced *high* reward becomes more likely and whatever produced *low* reward becomes less likely. This process is repeated many times across tasks so that the model maximizes the reward. The kind of RL training that is most commonly done with LLMs training is in the [GRPO](https://huggingface.co/learn/llm-course/en/chapter12/3b) lineage, which samples a group of attempts for the same task (let's say 8), and then reinforces ones that scored above the group's average score. This has a practical consequence you can reason about: if all 8 attempts score identically (perfectly or horribly), that task teaches nothing this round. The task has to be solvable sometimes for learning to happen.

We'll train on the cookbook's local `reverse_text_v1`. The goal is: given a string, write it backwards. It's a deliberately humble task with three useful properties: it's trivially verifiable because the correct answer is computable, small models are bad at it so there's plenty of room to learn, and the reward allows for partial credit, so there is incremental learning signal available for the model.

Training our model to get good at reversing text using `prime-rl` is dead simple: we specify our config, run a single command, and then monitor (either through the CLI or on the platform)!

## Step 1: create a config

The hosted-training CLI writes a starter config (named `reverse-text.toml`):

```bash
prime train init reverse-text.toml
```

Open it and you can modify fields to set up the eval you want.

For this tutorial, we have provided a config in `configs/03/reverse-text-rl.toml`, running which costs below 20 cents:

```toml
model = "meta-llama/Llama-3.2-1B-Instruct"
max_steps = 100

batch_size = 128
rollouts_per_example = 8

[sampling]
max_tokens = 1024

[[env]]
name = "reverse_text_v1"
taskset = { id = "reverse_text_v1" }
harness = { id = "default", runtime = { type = "subprocess" } }
```

Notice the two layers of knobs:

- **Trainer knobs** at the top: the model, how long to train, and how many rollouts feed each step. `prime train models` lists the models currently available on hosted training.
- **The environment**, embedded under `[[env]]`works the same a the taskset/harness pair used for evals. Nothing about the task or scoring changes between evaluation and training.

You could add a periodic eval block as well by appending the following to the config:

```bash
[eval]
interval = 10
num_examples = 10
rollouts_per_example = 3
```

To see all the possible config options, run `prime train configs`  in your terminal. 

## Step 2: launch training

Once you are set with your config, running the training is just a command away:

```bash
prime train configs/03/reverse-text-rl.toml
```

The CLI shows you what the run will cost per 1M tokens of usage and asks you to confirm. Then the platform takes over: it provisions GPUs, spins up the environment and the trainer, and starts the loop. It's fully hosted and requires no specialized hardware or SSH connections on your side.

## Step 3: monitor

Your run gets an ID, shown both after confirming the run, and also visible in `prime train list`. The dashboard on [app.primeintellect.ai](https://app.primeintellect.ai) shows everything graphically, and the same telemetry is available from the terminal:

```bash
prime train get <run-id>            # status at a glance
prime train logs <run-id>           # live logs
prime train metrics <run-id>        # reward curve and training stats
prime train progress <run-id>       # sampled steps and distributions
prime train rollouts <run-id>       # actual rollouts, mid-training
prime train distributions <run-id>  # reward spread by step
```

A healthy run will generally look like:

- Reward trending upward, but noisily. RL curves wobble, so judge the trend over tens of steps.
- The eval score should increase from step 0 to later evals.
- Reward distribution shifting right, since early on most rollouts score low with a few lucky ones, later the probability mass moves toward high scores.

If you're curious what the model actually says at step 40, `prime train rollouts <run-id>` shows you.

## Step 4: the result

When the run finishes, the platform saves the trained model weights — checkpoints along the way (see the `[checkpoints]` options in the generated config) and the final adapter at the end, listed by:

```bash
prime train checkpoints <run-id>
```

You've closed the loop the whole Lab stack is built for: **evaluate → train → evaluate**. You could deploy a checkpoint for inference as well, but we will cover that in a later tutorial.

To stop a run, use `prime train stop <run-id>`. `prime train delete <run-id>` removes a finished one.

## Recap

You learned the try → score → nudge loop, why groups of rollouts and sometimes-solvable tasks matter, launched a hosted training run from one TOML file, and read its progress with the same trace-level skepticism you learned for evals.

Apart from RL, there's one more way to make a model better that doesn't touch weights at all, which we will cover in the next tutorial.

Next → [4 — Your First GEPA Run](4_first_gepa.md): evolve the *prompt* instead.