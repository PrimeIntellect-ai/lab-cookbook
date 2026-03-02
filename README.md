# prime-cookbook

> **Ready-to-run training environments built with the verifiers library**

`prime-cookbook` is a collection of RL environment recipes for Prime Intellect Lab — similar to [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) but purpose-built for our ecosystem. Each recipe is a standalone, runnable training environment that demonstrates a specific pattern or capability.

---

## Quick Start

```bash
# Install the cookbook and its utilities
pip install -e .

# Or install from PyPI
pip install prime-cookbook

# Install a specific recipe environment
prime env install environments/wiki-search

# Run evaluation
prime eval run environments/wiki-search

# Run RL training
prime rl run environments/wiki-search/config.toml
```

**Requirements:**
- Python 3.10+
- `verifiers >= 0.1.10`
- `prime` CLI (from `prime-rl`)

---

## Recipes

| Recipe | Level | Verifier | Description |
|--------|-------|----------|-------------|
| [wiki-search](environments/wiki-search/) | L1 | Exact match | Retrieve Wikipedia facts with a search tool |
| [patent-search](environments/patent-search/) | L2 | Deterministic | Multi-hop patent analysis |
| [qa-synthesis](environments/qa-synthesis/) | L3 | JudgeRubric | Open-ended synthesis from a document corpus |
| [code-runner](environments/code-runner/) | L2 | Code execution | Generate and verify Python solutions |
| [math-solve](environments/math-solve/) | L2 | Math verify | Symbolic math with `\boxed{}` answers |

---

## Skills (Reusable Building Blocks)

Two families of skills live in `prime_cookbook/skills/`:

### Verifier Skills
Ready-made reward functions for plugging into any `vf.Rubric`:

| Skill | Module | Use When |
|-------|--------|----------|
| `exact_match_reward` | `skills.verifiers.exact_match` | Single-word / label answers |
| `contains_reward` | `skills.verifiers.exact_match` | Answer is a substring of completion |
| `set_match_reward` | `skills.verifiers.exact_match` | Multiple valid answers (search tasks) |
| `judge_reward` | `skills.verifiers.judge_rubric` | Simple LLM judge comparison |
| `universal_rubric_reward` | `skills.verifiers.judge_rubric` | Open-ended with key_points + source_quotes |
| `math_reward` | `skills.verifiers.math_verify` | `\boxed{}` math answers |
| `code_reward` | `skills.verifiers.code_verify` | Execute-and-check Python |
| `xml_parser_reward` | `skills.verifiers.parsers` | Structured `<answer>` tag output |
| `last_line_reward` | `skills.verifiers.parsers` | Final line exact match |

### Lab Skills
Utilities for building datasets and search indexes:

| Skill | Module | Use When |
|-------|--------|----------|
| `TFIDFSearchIndex` | `skills.lab.semantic_search` | Fast keyword search, no GPU needed |
| `SimpleSearchIndex` | `skills.lab.semantic_search` | Alias for TFIDFSearchIndex |
| `DatasetBuilder` | `skills.lab.dataset_builder` | Build HF datasets from Q&A pairs |
| `load_jsonl` / `save_jsonl` | `skills.lab.dataset_builder` | JSONL I/O helpers |
| `generate_ground_truth` | `skills.lab.ground_truth` | GPT-4.1 structured ground truth for L3 |
| `GroundTruth` | `skills.lab.ground_truth` | Dataclass: answer + key_points + source_quotes |

---

## Docs

- [Reward Design Guide](docs/reward-design.md)
- [Dataset Format](docs/dataset-format.md)
- [Training Config](docs/training-config.md)
- [Adding a New Recipe](CONTRIBUTING.md)

---

## Project Structure

```
prime-cookbook/
├── environments/          # Recipe environments (each is a pip-installable package)
│   ├── wiki-search/
│   ├── patent-search/
│   ├── qa-synthesis/
│   ├── code-runner/
│   └── math-solve/
├── prime_cookbook/        # Shared utility library
│   └── skills/
│       ├── verifiers/     # Reward functions
│       └── lab/           # Dataset + search utilities
├── docs/                  # Extended documentation
├── AGENTS.md              # Coding standards for LLM agents
├── CONTRIBUTING.md        # How to add a new recipe
└── pyproject.toml
```

---

## License

MIT
