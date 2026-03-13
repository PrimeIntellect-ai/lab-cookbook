# AGENTS.md — Coding Standards for LLM Agents

This file defines the coding standards for any LLM agent contributing to `prime-cookbook`. Read this before writing any code.

---

## Core Principles

### 1. Follow verifiers library patterns

Every environment must expose a `load_environment()` function that returns a `vf.Environment`:

```python
import verifiers as vf

def load_environment() -> vf.Environment:
    dataset = build_dataset()
    rubric = build_rubric()
    return vf.ToolEnv(
        dataset=dataset,
        rubric=rubric,
        tools=[...],
        max_turns=10,
    )
```

The `pyproject.toml` of each recipe must include the entrypoint:

```toml
[tool.verifiers.environment]
entrypoint = "my_recipe.my_module:load_environment"
```

### 2. NO REGEX — ever

Do not use `re.compile`, `re.search`, `re.match`, `re.findall`, or any `re.*` function anywhere in this codebase.

**Why**: Regex introduces fragile string-matching logic that fails silently on edge cases and is impossible to unit test exhaustively.

**Instead, use**:
- `str.find()` — locate a substring, returns -1 on miss
- `str.split()` — split on a delimiter
- `str.strip()`, `str.lower()`, `str.upper()` — normalize
- `json.loads()` — parse structured JSON output from models
- String slicing `text[start:end]` — extract substrings

**Parsing LLM output example**:
```python
# ❌ WRONG
import re
match = re.search(r'"score":\s*([\d.]+)', response)
score = float(match.group(1)) if match else 0.0

# ✅ CORRECT
try:
    result = json.loads(response)
    score = float(result.get("score", 0.0))
except (json.JSONDecodeError, ValueError):
    idx = response.find('"score"')
    if idx == -1:
        score = 0.0
    else:
        chunk = response[idx + 8:idx + 30]
        for part in chunk.split():
            try:
                score = float(part.strip(',:}"'))
                break
            except ValueError:
                continue
```

### 3. Tools must be STATELESS

A tool function must always return the same output for the same input. No global state, no mutation of shared objects, no side effects between calls.

```python
# ❌ WRONG — mutates shared state
_call_count = 0

async def search_tool(query: str) -> str:
    global _call_count
    _call_count += 1
    ...

# ✅ CORRECT — pure function
async def search_tool(query: str, state: dict) -> str:
    # If you need to track calls, use state (injected by verifiers)
    state["search_calls"] = state.get("search_calls", 0) + 1
    ...
```

### 4. API key management

Use `vf.ensure_keys()` at module import time for any required API keys:

```python
import verifiers as vf

vf.ensure_keys(["OPENAI_API_KEY"])          # Single key
vf.ensure_keys(["OPENAI_API_KEY", "PRIME_API_KEY"])  # Multiple keys
```

This raises a clear error with instructions if a key is missing, rather than failing at runtime.

### 5. Reward function signatures

All reward functions must be `async def` and return `float` in range `[0.0, 1.0]`:

```python
async def my_reward(
    completion: str,    # Model's final response text
    answer: str,        # Ground truth answer from dataset
    info: dict,         # Metadata from dataset row
    prompt: list,       # Full conversation prompt
    state: dict,        # Mutable per-rollout state
    parser: ...,        # Parser object (if set on rubric)
    judge: ...,         # Judge callable (if JudgeRubric used)
    **kwargs,           # Always accept **kwargs for forward compatibility
) -> float:
    ...
```

Arguments are injected by name — only include the ones you need. Always include `**kwargs`.

### 6. Dataset column schema

Every dataset must have these columns:

| Column | Type | Description |
|--------|------|-------------|
| `question` | `str` | The question (for single-turn) |
| `prompt` | `list` | Full conversation prompt (for multi-turn; overrides `question`) |
| `answer` | `str` | Ground truth answer |
| `info` | `dict` | Task-specific metadata (task type, source IDs, valid answers, etc.) |

### 7. pyproject.toml must include eval defaults

Every recipe's `pyproject.toml` must include:

```toml
[tool.verifiers.eval]
num_examples = 100
rollouts_per_example = 4
```

Adjust `num_examples` based on dataset size. `rollouts_per_example` should be 4-8 for eval.

### 8. Training config: TOML format

Training configs use TOML for `prime rl run`:

```toml
[train]
model = "meta-llama/Llama-3.2-3B-Instruct"
environment = "my_recipe.my_module:load_environment"
rollouts_per_example = 8
learning_rate = 1e-5
max_steps = 1000

[eval]
judge_model = "gpt-4.1-mini"
num_examples = 100
```

### 9. Reward calibration

Before merging a new recipe, verify the starting reward is in the sweet spot:

- **Target**: `0.15 – 0.35` on the base model (untrained)
- **Too easy** (> 0.35): Model already solves it — no learning signal. Make the task harder.
- **Too hard** (< 0.15): Model never gets reward — no learning signal. Add easier examples or simplify.

Run: `prime eval run recipe-<name>` (or the recipe's local config) and check mean reward in the output.

### 10. Reward level hierarchy

| Level | Verifier type | When to use |
|-------|--------------|-------------|
| **L1** | Binary exact match | Single correct answer, deterministic check |
| **L2** | Deterministic check | Multi-step verification (code execution, math, structured output) |
| **L3** | `JudgeRubric` | Open-ended synthesis, analysis, explanation |

For L3, use a strict `vf.JudgeRubric` (or equivalent) and explicitly include hallucination/factuality penalization in post-processing.

### 11. Hallucination penalty

For L3 judge rewards, always apply these penalties:

```python
if not result.get("no_hallucination", True):
    score *= 0.2   # Severe penalty — better to undertrain than train on bad signal
elif not result.get("factually_accurate", True):
    score *= 0.5   # Moderate penalty for factual errors
```

---

## File Checklist for New Recipes

- [ ] `cookbook/recipes/<name>/<name>.py` (or `recipe_<name>.py`) — exposes `load_environment()`
- [ ] `cookbook/recipes/<name>/pyproject.toml` — entrypoint + `[tool.verifiers.eval]`
- [ ] `cookbook/recipes/<name>/config.toml` — training config for `prime rl run`
- [ ] `cookbook/recipes/<name>/README.md` — complete documentation (quick start, metrics, design notes)
- [ ] Entry added to the recipes table in root `README.md`

---

## What NOT to Do

- Do not use `re` module anywhere
- Do not store mutable state in module-level variables accessed by tools
- Do not hard-code API keys — use `vf.ensure_keys()` and env vars
- Do not use `subprocess` in tools without a timeout
- Do not use synchronous HTTP calls in async reward functions (use `httpx.AsyncClient` or `openai.AsyncOpenAI`)
- Do not assume the judge always returns valid JSON — always have a fallback parser
- Do not merge recipes with starting reward outside `[0.05, 0.45]` without a note explaining why
