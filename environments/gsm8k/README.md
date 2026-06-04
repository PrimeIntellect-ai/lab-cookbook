# gsm8k

<a href="https://github.com/PrimeIntellect-ai/verifiers/tree/main/environments/gsm8k">
<img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="Source Code">
</a>

### Overview
- **Environment ID**: `gsm8k`
- **Short description**: Single-turn GSM8K math word problems with boxed numeric answers and CoT.
- **Tags**: math, gsm8k, single-turn, think, boxed-answer

### Datasets
- **Primary dataset(s)**: `gsm8k` train (train) and test (eval) via `load_example_dataset`
- **Source links**: Uses the example loader in `verifiers.utils.data_utils`
- **Split sizes**: Full GSM8K train (source) and test (eval) splits
- **Task identity**: assigned by Verifiers when task records are normalized

### Task
- **Type**: single-turn
- **Scoring**: Exact match on parsed `\boxed{}` answer

### Quickstart
Run an evaluation with default settings:

```bash
prime eval run prime/gsm8k
```

Configure model and sampling:

```bash
prime eval run prime/gsm8k \
  -m gpt-4.1-mini \
  -n 20 -r 3 -t 1024 -T 0.7
```

### Metrics
| Metric | Meaning |
| ------ | ------- |
| `reward` | 1.0 if parsed boxed answer equals target, else 0.0 |
