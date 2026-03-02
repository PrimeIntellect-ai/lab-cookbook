# Getting Started

Get from zero to a running RL training job in ~10 minutes.

## 1. Install

```bash
# Clone the repo
git clone https://github.com/primeintellect/prime-cookbook
cd prime-cookbook

# Install in editable mode (installs verifiers + lab deps)
pip install -e .

# Verify
python -c "import prime_cookbook; print('ok')"
```

## 2. Set API Keys

```bash
export OPENAI_API_KEY=sk-...       # required for LLM judge rewards
export PRIME_API_KEY=pi-...        # required for Prime Inference training

# Optional: persist across sessions
echo 'export OPENAI_API_KEY=sk-...' >> ~/.bashrc
echo 'export PRIME_API_KEY=pi-...' >> ~/.bashrc
```

## 3. Install a Recipe

```bash
prime env install prime_cookbook/recipes/math_rl
```

This registers the environment with the `prime` CLI so you can reference it by ID (`recipe-math-rl`).

To verify installation:
```bash
prime env list
# recipe-math-rl   prime_cookbook.recipes.math_rl:MathRLEnv
# ...
```

## 4. Run Eval

Smoke-test the environment against a capable model before training:

```bash
prime eval run recipe-math-rl --model gpt-4.1-mini
```

Expected output:
```
Evaluating recipe-math-rl with gpt-4.1-mini...
  examples: 200
  reward mean:  0.94 ± 0.12
  reward > 0.5: 97.0%
  tool_calls:   0 (N/A for single-turn)
Done in 42s
```

A high score here confirms the environment is set up correctly. Math RL saturates quickly — see [math-reasoning.md](recipes/math-reasoning.md) for why this is expected.

## 5. Launch Training

```bash
prime rl run prime_cookbook/recipes/math_rl/config.toml
```

Example config (`config.toml`):
```toml
[model]
name = "Qwen/Qwen2.5-1.5B-Instruct"

[training]
max_steps = 500
batch_size = 64
rollouts_per_example = 8

[[env]]
id = "recipe-math-rl"
weight = 1.0
```

Pass secrets without editing the config:
```bash
prime rl run config.toml --env-var OPENAI_API_KEY=$OPENAI_API_KEY
```

## 6. Monitor

Training metrics stream to [Weights & Biases](https://wandb.ai) and the [Prime Dashboard](https://app.primeintellect.ai):

```bash
# Open dashboard in browser
prime rl status --open

# Tail logs locally
prime rl logs --follow
```

Key metrics to watch:
- `reward/mean` — should increase over training
- `reward/std` — high variance early is fine; should stabilize
- `token_efficiency` — tokens used vs max context

## Next Steps

- Explore [Environment Types](environment-types.md) to choose the right base class for your task
- Read [Reward Design](reward-design.md) before writing your own rubric
- Browse [Recipes](recipes/) for complete examples
