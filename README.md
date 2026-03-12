<p align="center">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/40c36e38-c5bd-4c5a-9cb3-f7b902cd155d">
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/6414bc9b-126b-41ca-9307-9e982430cde8">
    <img alt="Prime Intellect" src="https://github.com/user-attachments/assets/6414bc9b-126b-41ca-9307-9e982430cde8" width="312" style="max-width: 100%;">
  </picture>
</p>

---

<h3 align="center">
Prime Cookbook
</h3>

<p align="center">A practical collection of RL environment recipes for Lab + verifiers.</p>

## Local development

For contributing to this repository:

```bash
uv sync
```

Optional developer dependencies:

```bash
uv sync --extra dev
```

Environment-specific usage and setup live in each environment folder under `cookbook/recipes/`.

## Evaluate and train

Typical workflow:

```bash
# Train from an environment config
prime rl run cookbook/recipes/swe_grep/config.toml
```

See each environment README for environment-specific evaluation and training commands.

You define:
- a dataset (`question` / `prompt`, `answer`, `info`),
- an environment (`SingleTurnEnv`, `ToolEnv`, `StatefulToolEnv`, `PythonEnv`, `EnvGroup`),
- a rubric (deterministic and/or judge-based rewards).

## Recipes

All recipes live in `cookbook/recipes/`:

1. **[Ethics Debate](cookbook/recipes/ethics_debate/README.md)**
2. **[Patent Search](cookbook/recipes/patent_search/README.md)**
3. **[SWE Grep](cookbook/recipes/swe_grep/README.md)**

Each environment keeps its documentation in its own folder, alongside its implementation and config files.

## Project Docs

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
