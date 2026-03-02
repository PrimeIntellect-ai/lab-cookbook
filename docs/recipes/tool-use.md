# Tool Use

Stateless tool-calling environment. Model is given a question and a set of tools; it must call tools strategically and synthesize a final answer.

**Environment type:** `ToolEnv`  
**Reward:** Exact match on final answer + tool-use efficiency bonus  
**Dataset:** Mixed (calculator, unit conversion, lookup tasks)

---

## Quick Start

```bash
prime env install prime_cookbook/recipes/tool_use

prime eval run recipe-tool-use --model gpt-4.1-mini
# Expected: reward mean ~0.72

prime rl run prime_cookbook/recipes/tool_use/config.toml
```

---

## Environment Overview

```python
import verifiers as vf
from prime_cookbook.skills.verifiers import exact_match_reward

def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Returns the numeric result."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(round(result, 6))
    except Exception as e:
        return f"Error: {e}"

def unit_convert(value: float, from_unit: str, to_unit: str) -> str:
    """Convert a value between units (e.g., km to miles, kg to lbs)."""
    return convert(value, from_unit, to_unit)  # see recipe source

def lookup(term: str) -> str:
    """Look up a factual definition or property."""
    return lookup_index.search(term, top_k=1)[0]["text"]

env = vf.ToolEnv(
    dataset=dataset,
    rubric=vf.Rubric(funcs=[answer_reward]),
    tools=[calculator, unit_convert, lookup],
    max_turns=8,
    system_prompt=(
        "Use the available tools to answer questions accurately. "
        "State your final answer clearly on the last line."
    ),
)
```

---

## Expected Metrics

| Model | Reward Mean | Avg Tool Calls | Notes |
|-------|-------------|---------------|-------|
| gpt-4.1-mini | ~0.72 | 2.1 | Solid baseline |
| Qwen2.5-7B-Instruct | ~0.58 | 1.8 | Trained well on tool use |
| Qwen2.5-1.5B-Instruct | ~0.28 | 1.2 | Good training difficulty |

---

## Reward Design

```python
from prime_cookbook.skills.verifiers import exact_match_reward, last_line_reward

def answer_reward(completion: str, state: dict, **kwargs) -> float:
    answer = state["info"]["answer"]
    # Primary: exact match on last line
    match = last_line_reward(completion, answer, normalize=True, strip_prefix="Answer:")
    return match

def efficiency_bonus(completion: str, state: dict, **kwargs) -> float:
    # Small bonus for using fewer tool calls (encourage conciseness)
    tool_calls = state.get("total_tool_calls", 0)
    max_calls = 8
    if tool_calls == 0:
        return 0.0  # no bonus if no tools used
    return max(0.0, 1 - (tool_calls - 1) / max_calls) * 0.1

rubric = vf.Rubric(
    funcs=[answer_reward, efficiency_bonus],
    weights=[0.9, 0.1],
    combine="sum",
)
```

---

## Training Config

```toml
[model]
name = "Qwen/Qwen2.5-7B-Instruct"

[training]
max_steps = 1500
batch_size = 64
rollouts_per_example = 8

[sampling]
max_tokens = 1024
temperature = 1.0

[[env]]
id = "recipe-tool-use"
weight = 1.0
```

---

## Extension Ideas

**Add web search:**
```python
import httpx

async def web_search(query: str) -> str:
    """Search the web and return a snippet."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://api.search.example.com/search",
            params={"q": query, "n": 3},
        )
    return "\n".join(r["snippet"] for r in resp.json()["results"])
```

**Add error recovery:**
```python
env = vf.ToolEnv(
    ...
    stop_errors=[ValueError],       # ValueError terminates rollout
    # other errors returned as tool response (model can retry)
)
```

**Combine with math tasks (multi-env):**
See [multi-env recipe](multi-env.md).

---

## Related

- [Environment Types](../environment-types.md) — ToolEnv details
- [Verifier Skills](../verifiers-skills.md)
- [Multi-Env Recipe](multi-env.md)
