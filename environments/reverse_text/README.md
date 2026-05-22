# reverse-text

Single-turn environment that asks the model to reverse a string character by
character. Partial credit comes from a longest-common-subsequence ratio against
the true reversal.

### Overview
- **Environment ID**: `reverse-text`
- **Short description**: Reverse input text character-by-character with tagged output.
- **Tags**: single-turn, text, train, eval

### Datasets
- **Primary dataset(s)**: [PrimeIntellect/Reverse-Text-RL](https://huggingface.co/datasets/PrimeIntellect/Reverse-Text-RL) train split
- **Split sizes**: full train split for task rows

### Task
- **Type**: single-turn
- **Output format expectations**: answer inside `<reversed_text>...</reversed_text>` tags
- **Scoring**: LCS ratio between parsed answer and reversed input

### Quickstart

```bash
prime eval run reverse-text
```

Configure model and sampling:

```bash
prime eval run reverse-text \
  -m openai/gpt-4.1-mini \
  -n 20 -r 3 -t 1024 -T 0.7
```

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `lcs_reward` | LCS ratio between parsed `<reversed_text>` answer and target reversal |
