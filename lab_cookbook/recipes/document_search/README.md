# Document Search — 3-Level Curriculum Recipe

The flagship recipe in the cookbook. Mimics the patent search blog post methodology:  
start with simple retrieval (L1), progress through multi-step reasoning (L2), and finish  
with open-ended qualitative analysis (L3).

## Architecture

```
Corpus: 100 synthetic company profiles
        ┌───────────────────────────────┐
        │  id, name, founded, industry  │
        │  employees, revenue, hq, public│
        └───────────────────────────────┘
              ↕ TF-IDF search index

Tools:
  search_docs(query)      → [{id, title, score}, ...]
  get_metadata(doc_id)    → full profile dict
```

## Level 1 — Metadata Retrieval

**Questions**: Single-fact lookup ("What year was Acme Corp founded?")  
**Reward**: Exact match — binary 0.0 / 1.0  
**Difficulty**: Starting reward ~0.15-0.35 with 1B model  
**Saturates**: ~15-20 training steps

```python
from prime_cookbook.recipes.document_search.document_search_l1 import load_environment
env = load_environment()
```

## Level 2 — Multi-Step Reasoning

**Questions**: Require reasoning across 2+ documents:
- "Which was founded more recently: X or Y?"
- "How many years ago was X founded?"  
- "How many companies are in the Biotech industry?"
- "Which has the highest revenue of A, B, C, D?"

**Reward**: Deterministic programmatic check (no judge needed)  
**Difficulty**: Starting reward ~0.10-0.20 with 1B model  
**Partial credit**: 0.5 if answer is contained in response

```python
from prime_cookbook.recipes.document_search.document_search_l2 import load_environment
env = load_environment()
```

## Level 3 — Open-Ended Analysis

**Questions**: Strategic synthesis across multiple companies.  
**Reward**: `vf.JudgeRubric` with GPT-4.1 judge (scores 0-10 on 4 criteria)  
**Requires**: `OPENAI_API_KEY` set in environment  
**Ceiling**: ~0.75-0.85 (judge rarely awards perfect scores)

```python
from prime_cookbook.recipes.document_search.document_search_l3 import load_environment
env = load_environment()
```

## Setup

```bash
pip install verifiers>=0.1.10
# For L3 only:
export OPENAI_API_KEY=sk-...
```

## Training (L1)

```bash
prime rl run config.toml
```

## Expected Curriculum Progress

| Level | Task | Starting Reward | Saturates At |
|-------|------|----------------|--------------|
| L1 | Single-fact retrieval | 0.15-0.35 | ~0.95 |
| L2 | Multi-step reasoning | 0.10-0.20 | ~0.80 |
| L3 | Open-ended analysis | 0.20-0.40 | ~0.75 |

## Adapting to a New Domain

This 3-level curriculum pattern generalises directly:

1. **Replace the corpus** — use real documents (patents, research papers, manuals)
2. **Keep the tools** — `search_docs` and `get_metadata` are domain-agnostic
3. **L1**: Single-field lookup questions (generated from document metadata)
4. **L2**: Cross-document reasoning (comparisons, aggregations, ranking)
5. **L3**: Open synthesis (JudgeRubric with domain-specific criteria)

The `TFIDFSearchIndex` in `prime_cookbook/skills/lab.py` is a drop-in search backend.  
For larger corpora, replace it with an embedding-based index (e.g., FAISS + sentence-transformers).

## Key Design Decisions

- **No regex anywhere** — tool outputs use `str()` serialization, reward uses `str.find()` and `str.split()`
- **Shared corpus** — L1/L2/L3 all import from `document_search_l1.py` to avoid duplication
- **Deterministic rewards for L1+L2** — avoids judge variance during early training when signal is critical
- **Judge only at L3** — once the model can retrieve and reason, qualitative evaluation is appropriate
