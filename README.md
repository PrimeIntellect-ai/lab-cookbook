<h1 align="center">Prime Cookbook</h1>

A practical collection of RL environment recipes for Lab + verifiers.

## Installation

1. Install Python 3.10+.
2. Install dependencies:

```bash
uv sync
```

Optional (tests/lint):

```bash
uv sync --extra dev
```

## Evaluate and train

Typical workflow:

```bash
# Evaluate a packaged recipe by id
prime eval run recipe-math-rl --model gpt-4.1-mini

# Train from a recipe config
prime rl run cookbook/recipes/math_rl/config.toml
```

You define:
- a dataset (`question` / `prompt`, `answer`, `info`),
- an environment (`SingleTurnEnv`, `ToolEnv`, `StatefulToolEnv`, `PythonEnv`, `EnvGroup`),
- a rubric (deterministic and/or judge-based rewards).

## Recipes

All recipes live in `cookbook/recipes/`:

1. **[Math RL](cookbook/recipes/math_rl/README.md)**
2. **[Tool Use](cookbook/recipes/tool_use/README.md)**
3. **[Word Game](cookbook/recipes/word_game/README.md)**
4. **[Sandbox Code](cookbook/recipes/sandbox_code/README.md)**
5. **[Document Search](cookbook/recipes/document_search/README.md)**
6. **[Multi Env](cookbook/recipes/multi_env/README.md)**

Each recipe folder contains:
- `README.md`
- environment implementation (`*.py`)
- `config.toml`
- `pyproject.toml`

## Docs

- Main docs: [`docs/`](docs/)
- Contribution guide: [`CONTRIBUTING.md`](CONTRIBUTING.md)
- Agent standards: [`AGENTS.md`](AGENTS.md)

Prime Lab: [primeintellect.ai/blog/lab](https://www.primeintellect.ai/blog/lab)

## Contributing

PRs are welcome for:
- new recipe environments,
- better reward/rubric design,
- evaluation/training workflows,
- docs and reproducibility improvements.

## License

MIT
