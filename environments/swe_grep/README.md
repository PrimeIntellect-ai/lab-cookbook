# swe-grep

A v1 tool-use taskset for repository search questions.

- Task: natural-language query plus expected answer and file path labels
- Toolset: grep, list files, and read file over a cloned repository
- Reward: LLM answer judge, file-path metrics, and a group efficiency bonus
- Tool placement: shared by default so repository setup happens once

Run the training example config:

```bash
uv run train environments/swe_grep/swe_grep_rl.toml
```

For eval-style smoke runs:

```bash
uv run eval swe-grep -n 3 -r 2 --max-turns 4
```
