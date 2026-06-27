# ethics-debate

A v1 synthetic debate taskset.

- Task: ethics question and number of debate rounds
- User simulator: manages argument/critique turns
- State: debate role, handoff history, and final argument
- Reward: LLM judge over final argument quality

Run directly with the taskset id after installing the workspace:

```bash
uv run eval ethics-debate -n 5 -r 1 --max-turns 6
```
