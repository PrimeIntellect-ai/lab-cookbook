# Document Search

Multi-turn document search with a 3-level difficulty curriculum. The model must search a corpus, synthesize retrieved evidence, and produce a cited answer. This is the most complex recipe in the cookbook and the canonical example of curriculum RL.

**Environment type:** `StatefulToolEnv` (levels 1–2), `StatefulToolEnv` + `JudgeRubric` (level 3)  
**Reward:** L1: exact match, L2: set match with partial credit, L3: LLM judge  
**Dataset:** 5,000 questions over a 50K-document Wikipedia subset

---

## Quick Start

```bash
prime env install prime_cookbook/recipes/document_search

# Evaluate each level
prime eval run recipe-document-search-l1 --model gpt-4.1-mini   # ~0.71
prime eval run recipe-document-search-l2 --model gpt-4.1-mini   # ~0.48
prime eval run recipe-document-search-l3 --model gpt-4.1-mini   # ~0.61

prime rl run prime_cookbook/recipes/document_search/config.toml
```

---

## The 3-Level Curriculum

### Why Curriculum?

RL with sparse rewards (L3 only) stalls — the model rarely gets positive signal early in training. The curriculum bootstraps learning:

1. **L1 teaches search mechanics** — model learns to call `search()`, read snippets, formulate answers
2. **L2 teaches evidence synthesis** — model learns to combine multiple retrieved documents  
3. **L3 teaches open-ended quality** — model learns conciseness, citation quality, completeness

**Key insight:** L1 reward is cheap and fast (exact match). Use it to bootstrap, then shift training budget toward L2 and L3.

### Level 1 — Exact Match

Simple factoid questions with a single correct answer.

```
Q: "What year was the Eiffel Tower completed?"
A: "1889"
```

```python
from prime_cookbook.skills.verifiers import exact_match_reward
from prime_cookbook.skills.lab import TFIDFSearchIndex

class DocSearchL1(vf.StatefulToolEnv):
    def __init__(self, index: TFIDFSearchIndex, **kwargs):
        self.index = index
        super().__init__(**kwargs)

    def setup_state(self, state):
        state["searches"] = []
        return state

    def search(self, query: str, _state: dict) -> str:
        """Search the document corpus. Returns top-3 relevant passages."""
        results = self.index.search(query, top_k=3)
        _state["searches"].append(query)
        return "\n\n".join(
            f"[{i+1}] {r['text']}" for i, r in enumerate(results)
        )

    def get_tools(self):
        return [self.search]

    def get_rubric(self):
        def reward(completion, state, **kwargs):
            answer = state["info"]["answer"]
            return exact_match_reward(completion, answer, normalize=True)
        return vf.Rubric(funcs=[reward])
```

### Level 2 — Set Match with Partial Credit

Multi-hop questions requiring ≥2 searches and multiple facts.

```
Q: "Name the two founding members of Apple Computer."
A: ["Steve Jobs", "Steve Wozniak"]   (order doesn't matter)
```

```python
from prime_cookbook.skills.verifiers import set_match_reward

def l2_reward(completion: str, state: dict, **kwargs) -> float:
    valid_answers = state["info"]["valid_answers"]  # list of strings
    # Full credit: both answers found. Partial: one answer found.
    return set_match_reward(
        completion,
        valid_answers,
        threshold=0.0,    # allow partial credit
        normalize=True,
    )
```

### Level 3 — LLM Judge

Open-ended questions where quality of reasoning and citation matters.

```
Q: "How did the Industrial Revolution change urban living conditions in England?"
A: (long, multi-paragraph, judge-evaluated)
```

```python
import verifiers as vf

rubric_l3 = vf.JudgeRubric(
    judge_model="gpt-4.1-mini",
    criteria=[
        "accuracy",          # factually correct
        "no_hallucination",  # doesn't fabricate
        "covers_key_points", # addresses all parts
        "cites_evidence",    # uses retrieved passages
        "concise",           # avoids unnecessary padding
    ],
    weights=[3, 3, 2, 1, 1],
    system_prompt=(
        "Evaluate the response to the question. "
        "Score 0–1 per criterion. Be strict. "
        "Award 'cites_evidence' only if the response explicitly references "
        "information from the retrieved passages."
    ),
)
```

---

## Building the Index

```bash
# One-time setup — builds TF-IDF index from Wikipedia subset
python prime_cookbook/recipes/document_search/build_index.py \
  --output data/wiki_index.pkl \
  --n-docs 50000
```

```python
# build_index.py (abbreviated)
from datasets import load_dataset
from prime_cookbook.skills.lab import TFIDFSearchIndex

wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
wiki = wiki.select(range(50_000))

docs = [
    {"id": f"wiki_{i}", "text": row["text"][:2000], "title": row["title"]}
    for i, row in enumerate(wiki)
]

index = TFIDFSearchIndex(max_features=100_000, ngram_range=(1, 2))
index.build(docs, text_field="text", id_field="id")
index.save("data/wiki_index.pkl")
print(f"Built index: {len(docs):,} documents")
```

---

## Ground Truth Generation (L3)

Generate reference answers offline before training:

```bash
python prime_cookbook/recipes/document_search/generate_gt.py \
  --dataset data/questions_l3.jsonl \
  --output data/ground_truth_l3.jsonl \
  --model gpt-4.1
```

```python
# generate_gt.py (abbreviated)
from prime_cookbook.skills.lab import generate_ground_truth, TFIDFSearchIndex

index = TFIDFSearchIndex.load("data/wiki_index.pkl")

def augmented_prompt(question: str) -> str:
    """Include retrieved context in GT generation prompt."""
    results = index.search(question, top_k=5)
    context = "\n\n".join(r["text"] for r in results)
    return f"Context:\n{context}\n\nQuestion: {question}"

ground_truths = generate_ground_truth(
    questions=[augmented_prompt(q) for q in questions],
    model="gpt-4.1",
    system_prompt="Answer the question using ONLY the provided context. Be precise.",
    batch_size=20,
)
```

---

## Curriculum Training Config

```toml
[model]
name = "Qwen/Qwen2.5-7B-Instruct"

[training]
max_steps = 3000
batch_size = 128
rollouts_per_example = 8
learning_rate = 5e-6

[sampling]
max_tokens = 2048
temperature = 0.9

# L1 — high weight early; model bootstraps search mechanics
[[env]]
id = "recipe-document-search-l1"
weight = 3.0

# L2 — medium weight; teaches multi-hop synthesis
[[env]]
id = "recipe-document-search-l2"
weight = 2.0

# L3 — lower weight; expensive judge but teaches quality
[[env]]
id = "recipe-document-search-l3"
weight = 1.0
```

**Curriculum schedule (manual, adjust based on eval):**

| Training Step | L1 Weight | L2 Weight | L3 Weight |
|--------------|-----------|-----------|-----------|
| 0–500 | 3.0 | 1.0 | 0.0 |
| 500–1500 | 2.0 | 2.0 | 0.5 |
| 1500–3000 | 1.0 | 2.0 | 2.0 |

---

## Expected Metrics

| Model @ Checkpoint | L1 | L2 | L3 |
|-------------------|----|----|-----|
| Qwen2.5-1.5B (base) | 0.18 | 0.12 | 0.22 |
| Qwen2.5-1.5B (step 500) | 0.45 | 0.28 | 0.35 |
| Qwen2.5-7B (base) | 0.51 | 0.38 | 0.55 |
| Qwen2.5-7B (step 1000) | 0.71 | 0.54 | 0.68 |
| gpt-4.1-mini (eval only) | 0.71 | 0.48 | 0.61 |

---

## Common Issues

**Model doesn't call search tool:**
- Add explicit instruction: "You MUST call the `search` tool at least once before answering."
- Check `avg_tool_calls` in eval — should be ≥1.5 for L1, ≥2.5 for L2

**L3 reward collapses:**
- Reduce L3 weight until L1/L2 rewards stabilize
- Ensure ground truths are generated with context (not just raw questions)

**Index miss-rate high:**
- Check query coverage: `index.search("test query", top_k=5)`
- Increase `max_features` if vocabulary is too small
- Consider switching to ChromaDB for better semantic coverage

---

## Related

- [Lab Skills](../lab-skills.md) — TFIDFSearchIndex, DatasetBuilder, generate_ground_truth
- [Reward Design](../reward-design.md) — 3-level curriculum theory
- [Environment Types](../environment-types.md) — StatefulToolEnv
