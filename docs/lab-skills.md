# Lab Skills

Data pipeline utilities in `prime_cookbook/skills/lab/`. Use these to build datasets, search indices, and ground truth for RL environments.

```python
from prime_cookbook.skills.lab import (
    TFIDFSearchIndex,
    SimpleSearchIndex,
    DatasetBuilder,
    generate_ground_truth,
    GroundTruth,
)
```

---

## TFIDFSearchIndex

Sparse retrieval over a document corpus. Fast, no embedding model needed, good for <100K documents.

### Build and Search

```python
from prime_cookbook.skills.lab import TFIDFSearchIndex

# Build from a list of documents
docs = [
    {"id": "doc_1", "text": "Paris is the capital of France.", "title": "France"},
    {"id": "doc_2", "text": "Berlin is the capital of Germany.", "title": "Germany"},
    # ...
]

index = TFIDFSearchIndex()
index.build(docs, text_field="text", id_field="id")

# Search
results = index.search("capital of France", top_k=5)
# returns: [{"id": "doc_1", "text": "...", "score": 0.87}, ...]
```

### Save and Load

```python
# Save to disk
index.save("data/tfidf_index.pkl")

# Load (fast, no rebuild needed)
index = TFIDFSearchIndex.load("data/tfidf_index.pkl")
```

### Parameters

```python
TFIDFSearchIndex(
    max_features=50_000,      # vocabulary size
    ngram_range=(1, 2),       # unigrams + bigrams
    sublinear_tf=True,        # log-scale term frequency
)
```

**Typical build time:** ~10s for 10K docs, ~60s for 100K docs.

**When to use TFIDFSearchIndex vs SimpleSearchIndex:**
- `TFIDFSearchIndex` — better relevance ranking, slightly slower search
- `SimpleSearchIndex` — keyword exact match, instant, no sklearn dependency

---

## SimpleSearchIndex

Inverted index with BM25-style keyword matching. Zero-dependency, good for prototyping.

```python
from prime_cookbook.skills.lab import SimpleSearchIndex

index = SimpleSearchIndex()
index.build(docs, text_field="text")

results = index.search("capital city France", top_k=3)
```

**When to use:** You need a fast, lightweight index and don't need relevance ranking.

---

## DatasetBuilder

Build, validate, split, and export RL training datasets.

### Building a Dataset

```python
from prime_cookbook.skills.lab import DatasetBuilder

builder = DatasetBuilder(
    required_columns=["question", "answer"],
    optional_columns=["context", "difficulty", "source"],
)

# Add examples one at a time
builder.add({
    "question": "What is the capital of France?",
    "answer": "Paris",
    "difficulty": "easy",
})

# Or add in bulk from a list
builder.add_batch(examples_list)
```

### Validate and Split

```python
# Check for duplicates, missing fields, empty values
builder.validate()

# 90/10 train-test split, reproducible
train_ds, eval_ds = builder.split(test_size=0.1, seed=42)

print(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")
```

### Export

```python
# Save as JSONL (recommended for large datasets)
builder.save_jsonl("data/train.jsonl", split="train")
builder.save_jsonl("data/eval.jsonl", split="eval")

# Or as HuggingFace dataset
hf_dataset = builder.to_hf_dataset()
hf_dataset.push_to_hub("my_org/my_dataset")
```

### Load Back

```python
from datasets import load_dataset

# From JSONL
dataset = load_dataset("json", data_files={"train": "data/train.jsonl"})

# From Hub
dataset = load_dataset("my_org/my_dataset")
```

---

## generate_ground_truth

Generate reference answers for L3 (LLM judge) environments. Run this **offline** before training — do not call during rollouts.

```python
from prime_cookbook.skills.lab import generate_ground_truth, GroundTruth

questions = dataset["question"]  # list of strings

ground_truths: list[GroundTruth] = generate_ground_truth(
    questions=questions,
    model="gpt-4.1",                           # use best model for GT
    system_prompt=(
        "You are an expert researcher. "
        "Answer concisely and accurately. "
        "Cite key facts."
    ),
    batch_size=50,                             # concurrent requests
    temperature=0.0,                           # deterministic GT
)

# Save
import json
with open("data/ground_truth.jsonl", "w") as f:
    for gt in ground_truths:
        f.write(gt.model_dump_json() + "\n")
```

### GroundTruth Dataclass

```python
from dataclasses import dataclass

@dataclass
class GroundTruth:
    question: str          # original question
    answer: str            # generated reference answer
    model: str             # model used for generation
    metadata: dict         # extra fields (source, difficulty, etc.)
    timestamp: str         # ISO 8601 generation time
```

### Loading Ground Truths

```python
import json
from prime_cookbook.skills.lab import GroundTruth

ground_truths = []
with open("data/ground_truth.jsonl") as f:
    for line in f:
        ground_truths.append(GroundTruth(**json.loads(line)))

# Build lookup dict
gt_by_question = {gt.question: gt.answer for gt in ground_truths}
```

### Using in an Environment

```python
class MyEnv(vf.StatefulToolEnv):
    def __init__(self, ground_truths: dict, **kwargs):
        self.ground_truths = ground_truths
        super().__init__(**kwargs)

    def get_reward(self, state) -> float:
        q = state["info"]["question"]
        reference = self.ground_truths.get(q, "")
        return judge_reward(state["completion"], reference)
```

---

## When to Use ChromaDB Instead

`TFIDFSearchIndex` is sufficient for most cookbook recipes. Switch to ChromaDB when:

| Condition | Use |
|-----------|-----|
| < 100K documents | `TFIDFSearchIndex` |
| > 100K documents | ChromaDB + embedding model |
| Need semantic similarity | ChromaDB + embedding model |
| Keyword match is sufficient | `TFIDFSearchIndex` or `SimpleSearchIndex` |
| Multi-tenant (multiple env instances sharing index) | ChromaDB (persistent, supports concurrent reads) |
| Offline / no GPU | `TFIDFSearchIndex` |

ChromaDB setup (not included in cookbook, install separately):
```bash
pip install chromadb sentence-transformers
```

```python
import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")  # ~90MB, CPU-friendly
client = chromadb.PersistentClient(path="data/chroma")
collection = client.create_collection("docs")

# Index
embeddings = model.encode([d["text"] for d in docs])
collection.add(
    documents=[d["text"] for d in docs],
    ids=[d["id"] for d in docs],
    embeddings=embeddings.tolist(),
)

# Search
results = collection.query(
    query_embeddings=model.encode(["capital of France"]).tolist(),
    n_results=5,
)
```

---

## Related Docs

- [Reward Design](reward-design.md) — how to use ground truths in rubrics
- [Document Search Recipe](recipes/document-search.md) — full example using TFIDFSearchIndex
- [Verifier Skills](verifiers-skills.md)
