# On-Policy Distillation

**On-policy distillation in Lab is currently in beta and will roll out to all users shortly.**

Use a teacher model to give dense token-level feedback while the student rolls out on-policy.

In [Warm Starts with SFT](../05-warm-starts-with-sft/README.md), a stronger teacher generated full trajectories and the student learned to imitate them. That works, but the student never visits the states it will actually produce at deployment. On-policy distillation closes the gap: the **student** samples rollouts from the environment, and the **teacher** scores each token so the update happens where the student actually errs.

## SFT vs On-Policy Distillation vs RL

| | SFT | On-policy distillation | RL |
| --- | --- | --- | --- |
| Rollout source | Teacher | Student | Student |
| Training signal | Teacher text | Teacher logprobs + optional env reward | Environment reward |

## Write a Distillation Config

Start from the SFT adapter on `prime/wordle`, add a teacher, and set `teacher_tau` to control distillation strength:

```toml
# [configs/06/wordle-opd.toml](../../configs/06/wordle-opd.toml)
model = "openai/gpt-oss-20b:my-sft-lora-distilled-from-oss-120b-distill"

[teacher]
model = "openai/gpt-oss-120b"
teacher_tau = 0.5

[sampling]
max_tokens = 512
reasoning_effort = "medium"

[[env]]
id = "prime/wordle"

[env.taskset]
num_train_examples = 512
num_eval_examples = 128

[env.harness]
max_turns = 6
```

The key fields are:

- `model` — student adapter
- `[teacher].model` — teacher for token-level logprob feedback
- `[teacher].teacher_tau` — distillation weight (`0` disables distillation)
- `[teacher].adv_tau` — optional weight on the RL advantage signal
- `[[env]]` — environment id
- `[env.taskset]` / `[env.harness]` — per-environment task and rollout overrides

## Launch Training

```bash
prime train configs/06/wordle-opd.toml
```

Follow logs the same way as RL:

```bash
prime train logs <run_id> -f
```

## Evaluate the Distilled Adapter

Run the same wordle eval shape you used in [Prompt Optimization](../04-prompt-optimization/README.md):

```bash
prime eval run prime/wordle \
  -m openai/gpt-oss-20b:<your-opd-adapter> \
  -n 20 \
  -r 1 \
  -t 1024
```

Compare against the SFT-only adapter on the same eval config.

## Continue with RL

After distillation, you can run a standard RL job on top of the new adapter to optimize directly against environment reward:

```toml
# [configs/05/wordle-rl.toml](../../configs/05/wordle-rl.toml)
model = "openai/gpt-oss-20b:<your-opd-adapter>"

[sampling]
max_tokens = 512
reasoning_effort = "medium"

[[env]]
id = "prime/wordle"

[env.taskset]
num_train_examples = 512
num_eval_examples = 128

[env.harness]
max_turns = 6
```

```bash
prime train configs/05/wordle-rl.toml
```

## Next

In [Judges and Instruction Following](../07-judges-and-instruction-following/README.md), walk through `simple-judge`, IFEval, and AdvancedIF.
