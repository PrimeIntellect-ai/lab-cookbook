<h1 align="center">Lab Cookbook</h1>

A practical collection of environment recipes for training and evaluating language models with Prime Lab.

## Installation

1. Install Python 3.10+.
2. Install dependencies:

```bash
uv sync
```

3. Ensure `prime` CLI is configured for your workflow.

If you plan to run tests/linting locally, use:

```bash
uv sync --extra dev
```

## prime-rl + verifiers basics

At a high level, evaluate first, then train:

```bash
# evaluate
prime eval run lab_cookbook/recipes/math_rl/config.toml

# train
prime rl run lab_cookbook/recipes/math_rl/config.toml
```

You define:

- a dataset (prompts + answers + optional metadata),
- an environment (single-turn, tool-use, multi-turn, sandbox, grouped),
- a rubric (deterministic and/or judge-based rewards).

## Cookbook Recipes

This repo includes production-style recipe examples in `lab_cookbook/recipes/`:

1. **[Math RL](lab_cookbook/recipes/math_rl/)**  
   Single-turn arithmetic reasoning with `MathRubric`.

2. **[Tool Use](lab_cookbook/recipes/tool_use/)**  
   Tool-calling environment with deterministic rewards.

3. **[Word Game](lab_cookbook/recipes/word_game/)**  
   Multi-turn gameplay environment with stateful interactions.

4. **[Sandbox Code](lab_cookbook/recipes/sandbox_code/)**  
   Code generation and execution using sandbox/python-style verification.

5. **[Document Search](lab_cookbook/recipes/document_search/)**  
   3-level curriculum (L1/L2/L3): retrieval → reasoning → open-ended synthesis.

6. **[Multi Env](lab_cookbook/recipes/multi_env/)**  
   Multi-task training with `EnvGroup` across several environments.

Each recipe folder includes:

- `README.md` (design + usage)
- `*.py` environment implementation
- `config.toml` training/eval config
- `pyproject.toml` recipe package metadata

## Skills

Under `lab_cookbook/skills/` you’ll find reusable skill docs for:

- creating environments,
- browsing/reviewing/evaluating environments,
- training and optimization workflows.

These are designed as practical operating guides for iterative environment development.

## Documentation

- Main docs: [`docs/`](docs/)
- Recipe docs: [`docs/recipes/`](docs/recipes/)
- Contribution guide: [`CONTRIBUTING.md`](CONTRIBUTING.md)

Rendered docs can be found here: [docs.primeintellect.ai](https://docs.primeintellect.ai/guides)

Prime Lab: [primeintellect.ai/blog/lab](https://www.primeintellect.ai/blog/lab)

## Contributing

We welcome improvements to:

- new environment recipes,
- reward/rubric design patterns,
- evaluation harnesses,
- documentation and reproducibility tooling.

Please open an issue or PR with a problem statement, expected behavior, and reproducible steps.

## License

MIT
