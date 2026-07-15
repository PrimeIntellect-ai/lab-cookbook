# Your First Eval

How good is GPT-5.4 mini at grade-school math? In the next ten minutes you will answer that with a real number, understand exactly where the number came from, and then raise the difficulty with a multi-turn game of Wordle. Along the way you'll pick up the four or five words of vocabulary that everything else in Lab is built on.

You need the setup from [Tutorial 1](1_setup.md): an authenticated CLI, and a terminal sitting inside the `lab-cookbook` directory.

## Run it first, understand it second

```bash
prime eval run gsm8k -n 10 -r 2 --model openai/gpt-5.4-mini --sampling.max-tokens 1024
```

While it runs, here is what you asked for, flag by flag:


| Part                          | Meaning                                                                                                                                                                                                    |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `gsm8k`                       | The environment to run — GSM8K is a classic dataset of grade-school word problems ("Natalia sold clips to 48 of her friends..."), each with a known numeric answer.                                        |
| `-n 10`                       | Evaluate 10 tasks (problems). Without it, you'd run the whole dataset.                                                                                                                                     |
| `-r 2`                        | Give the model **2 attempts per task**. Language models are sampled, so the same model can succeed once and fail once on the same problem — multiple attempts show you that variance instead of hiding it. |
| `--model openai/gpt-5.4-mini` | Which model to evaluate, routed through Prime Inference.                                                                                                                                                   |
| `--sampling.max-tokens 1024`  | Cap each model response at 1024 tokens, so a rambling answer can't run forever (or run up your bill).                                                                                                      |


You'll see 20 attempts (10 tasks × 2) stream by, each finishing with a score, and then a summary with the mean **reward** — for GSM8K, `1.0` means the extracted final answer matched the known one, `0.0` means it didn't. So a mean reward of `0.9` reads as: *90% of attempts produced the correct answer.* That's your number.

## What just happened

Four things you couldn't see, each with a name worth learning:

1. The CLI loaded the **taskset** — the "what": the problems, plus the scoring code that knows each correct answer.
2. For each attempt, a **harness** — the "how" — sent the problem to the model and collected its reply. Today's harness (`default`) is a simple chat loop; later you'll meet harnesses that wrap entire coding agents.
3. Each attempt produced a **rollout**, and its complete record — every message, the score, why it stopped — was saved as a **trace**.
4. The scoring code read each trace and computed the **reward**.

One sentence to remember: *a taskset and a harness make an environment; running it produces traces; rewards are computed from traces.* That's the whole model.

## Look at the evidence

A mean score tells you *how often* the model failed, never *why*. The traces do. Every run writes a folder:

```
outputs/<taskset>--<model>--<harness>/<run-id>/
├── config.toml      # the fully-resolved settings of this run
├── results.jsonl    # one trace per line — the raw evidence
└── eval.log         # logs
```

The comfortable way to browse it is the interactive viewer:

```bash
prime eval view
```

It lists your recent runs and lets you click through each trace: the problem, the model's full reasoning, and the reward it earned. Find a `0.0` trace and read it — usually the model either genuinely got the math wrong, or got it right but formatted the final answer somewhere the scorer didn't look. Learning to tell those apart by reading traces is *the* core skill of evaluation work, and you just did it once.

## Make it reproducible

Command-line flags are great for exploring, but the moment a result matters you want it written down. Any eval can be a TOML file — this one ships at `configs/01/first-eval.toml` (with a cheaper model):

```toml
model = "openai/gpt-5.4-mini"
num_tasks = 10
num_rollouts = 2

[sampling]
max_tokens = 1024

[taskset]
id = "gsm8k"
```

Run a config file with the `@` prefix:

```bash
prime eval run @ configs/01/first-eval.toml
```

Flags and files compose: flags win when both set the same thing, so `prime eval run @ configs/01/first-eval.toml -n 2` is a quick two-task smoke test of a saved config. Add `--dry-run` to check a config resolves without spending a single token. And since every run saves its resolved `config.toml` next to its results, every number you produce is reproducible after the fact.

## Now make it multi-turn: Wordle

GSM8K is one question, one answer. Most interesting tasks are *conversations*: the model acts, the environment responds, the model adapts. The classic word game Wordle is a perfect miniature — guess a five-letter word, learn which letters are right, guess again:

```bash
prime eval run wordle -n 5 --model openai/gpt-5.4-mini --max-turns 6 --sampling.max-tokens 1024
```

The one genuinely new flag is `--max-turns 6`: a hard ceiling on conversation length, enforced by the framework, so no rollout can loop forever. Six turns matches the real game's six guesses.

Open the run in `prime eval view` and read a full transcript. You'll see the model's guess, the environment's feedback on each letter, the next guess narrowing down — and at the end, a reward reflecting whether (and how efficiently) it solved the word. Notice what did *not* change: same command shape, same output folder, same trace reading. Single-turn math and a multi-turn game are the same machinery, which is exactly the point of environments.

## Things to try

- **Sampling matters:** re-run GSM8K with `--sampling.temperature 1.0` and compare the two runs' mean rewards. Higher temperature means more randomness — you should see more variance between the two attempts per task.
- **Models differ:** swap in a cheaper model (say `openai/gpt-5.4-nano`) and see what you lose. The task, scoring, and sample size stay fixed, so the delta is real.
- **Scale honestly:** 10 tasks is a smoke test, not a benchmark. Try `-n 100` on the cheaper model and watch the number stabilize.
- **No laptop required:** add `--hosted` to any of these and the platform runs the eval on its own infrastructure — handy for big runs. Results land in your dashboard.



## Recap

You measured two models on two environments, learned the vocabulary (taskset, harness, rollout, trace, reward), read failing traces instead of trusting a mean, and made a run reproducible with a config file.

So far the model was fixed and we just *measured* it. Time to change the model itself.

Next → [3 — Your First RL Run](3_first_rl.md): train a small model to get better at a task, live on hosted GPUs.