# Lab Skills

Data utilities for building RL training environments. These handle the plumbing — corpus indexing, dataset construction, and ground truth generation — so your environment code stays focused on the domain logic.

## Available Skills

| File | Classes / Functions | Use When |
|------|---------------------|----------|
| `semantic_search.py` | `TFIDFSearchIndex` / `SimpleSearchIndex` | Search tool for ToolEnv — build a retrievable corpus from a list of texts |
| `dataset_builder.py` | `DatasetBuilder` | Incrementally build a `datasets.Dataset` from Q&A pairs |
| `dataset_builder.py` | `load_jsonl`, `save_jsonl` | Read/write JSONL for inspection or caching |
| `ground_truth.py` | `generate_ground_truth` | Generate L3 structured ground truth via GPT-4.1 |
| `ground_truth.py` | `GroundTruth` | Dataclass: `answer`, `key_points`, `source_quotes` |

## Quick Start

### Build a search index for tools

```python
from prime_cookbook.skills.lab import TFIDFSearchIndex

# At environment init time — build once, use in tools
index = TFIDFSearchIndex()
index.build(
    texts=["Company profile text...", "Another doc..."],
    doc_ids=["doc_001", "doc_002"],
)

# In your tool function
def search_docs(query: str) -> str:
    results = index.search(query, k=5)
    return str([{"id": r["id"], "score": round(r["score"], 3)} for r in results])
```

### Build a dataset

```python
from prime_cookbook.skills.lab import DatasetBuilder

builder = DatasetBuilder()
builder.add(
    question="What year was Acme Corp founded?",
    answer="1987",
    info={"doc_id": "acme_corp", "field": "founded"},
)
dataset = builder.build()
train_ds, test_ds = builder.split(train_frac=0.8)
```

### Generate L3 ground truth

```python
import asyncio
from prime_cookbook.skills.lab import generate_ground_truth

gt = asyncio.run(generate_ground_truth(
    question="What is Acme Corp's core business model?",
    context="[full document text here]",
))
# gt.answer — prose answer
# gt.key_points — ["Acme focuses on...", "Revenue comes from..."]
# gt.source_quotes — ["direct quote from doc", ...]
```

## Search Index Details

`TFIDFSearchIndex` uses scikit-learn TF-IDF with bigrams. No GPU, no API calls, works offline.

- **Good for**: <100K documents, fast iteration, no external dependencies
- **Upgrade path**: Replace with ChromaDB + OpenAI embeddings for production (see below)

```python
# ChromaDB upgrade (when you need semantic search)
import chromadb

client = chromadb.Client()
collection = client.create_collection("docs")
collection.add(documents=texts, ids=doc_ids)

def search_docs(query: str) -> str:
    results = collection.query(query_texts=[query], n_results=5)
    return str(results["ids"][0])
```

### Save / load index

```python
index.save("corpus.pkl")
index = TFIDFSearchIndex.load("corpus.pkl")
```

## Dataset Schema

The verifiers library expects these columns:

| Column | Type | Required | Notes |
|--------|------|----------|-------|
| `question` | `str` | Yes (or `prompt`) | Auto-wrapped in user message |
| `answer` | `str` | Yes | Ground truth for reward functions |
| `info` | `dict` | Optional | Extra metadata (valid_answers, key_points, etc.) |
| `prompt` | `list[dict]` | Alt to `question` | Pre-formatted chat messages |

## Ground Truth Schema (L3)

```python
@dataclass
class GroundTruth:
    answer: str          # Full prose answer
    key_points: list     # ["Specific fact 1", "Specific fact 2", ...]
    source_quotes: list  # ["Direct quote from corpus", ...]
```

Pass these into your dataset `info` field:

```python
builder.add(
    question="...",
    answer=gt.answer,
    info={
        "reference_answer": gt.answer,
        "key_points": gt.key_points,
        "source_quotes": gt.source_quotes,
    },
)
```

## Adding a New Lab Skill

1. Add your module to this directory
2. Export it from `__init__.py`  
3. Document it in this SKILL.md table
4. Add usage to `docs/lab-skills.md`
