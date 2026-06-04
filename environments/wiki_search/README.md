# wiki-search

Agentic Wikipedia QA on a curated corpus, built on the verifiers v1 `Taskset` + `Harness` pattern.

The agent is given a trivia question and answers it by navigating a small Wikipedia corpus through three tools: embedding-based title search, section listing, and section reading. The corpus is loaded into a local ChromaDB index — **no live Wikipedia API calls**.

This is a v1 port of the canonical `wiki_search` environment in the `verifiers` repo.

## Environment overview

The stack:

- `WikiSearchTasksetConfig`: dataset, judge, embedding, and corpus defaults
- `vf.Taskset`: tasks from `willcb/wiki-trivia-questions-v4`, plus the toolset and judge reward
- `vf.Toolset`: ships the three Wikipedia tools and builds the shared ChromaDB index when the taskset loads
- `vf.Harness`: runs the default endpoint-backed tool loop

The shared corpus (`WikiIndex`, containing the Chroma collection and `page_id -> title` / `page_id -> content` maps) is closed over by the taskset-owned tools. The only process-level constant is the default Chroma path.

## Tools exposed to the model

- `search_pages(query)` — top-10 article candidates via title-embedding similarity over the ChromaDB index. Returns `[{page_id, title}, ...]`.
- `view_sections(page_id)` — parses Markdown-style `#` headings in the page content and returns the available `{section_id, section_name}` entries. Falls back to a single `:full` section if the page has no headings.
- `read_section(section_id)` — returns the slice of the page content for `section_id` (or the full page if `section_id` ends in `:full`).

## Datasets

- **Questions**: `willcb/wiki-trivia-questions-v4` (HF, `train` split)
- **Corpus**: `willcb/rare-wiki-pages` (HF, `train` split), indexed into a persistent ChromaDB collection (`wiki_titles`) under `.chroma_db/wiki-search` on first run.

The index is built lazily — the corpus + collection load runs the first time a rollout needs the tools, which allows multiple env instances to share work without colliding at construction time.

## Reward design

A single judge reward (weight `1.0`): `openai/gpt-4.1-mini` through Prime Inference returns yes/no on whether the final response is correct and coherent given the ground-truth answer. Incoherent responses score 0 even if the answer is buried inside them.

The judge call lives directly in `judge_reward` and uses its own `AsyncOpenAI`
client built from `WikiSearchTasksetConfig`. It does not use the rollout
endpoint proxy.

## Required environment variables

- `PRIME_API_KEY` — default key for Prime Inference judge and embedding calls.

Keys are validated by the Taskset component that owns each dependency:
embedding keys in `load_toolsets(config)`, judge keys in `judge_reward`.

## Quickstart

Run an eval with defaults:

```bash
prime eval run prime/wiki-search
```

Configure model and sampling:

```bash
prime eval run prime/wiki-search \
  -m openai/gpt-4.1-mini \
  -n 20 -r 3
```

The first run downloads the corpus and builds the Chroma index; subsequent runs reuse `.chroma_db/wiki-search`.

## Configuration

Dataset, corpus, embedding, and judge fields live on `WikiSearchTasksetConfig`.
The rollout budget belongs to the base harness:

```toml
[eval.taskset]
max_examples = 250
chroma_db_dir = ".chroma_db/wiki-search-small"

[eval.harness]
max_turns = 8
```

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `dataset_name` | str | `"willcb/wiki-trivia-questions-v4"` | HF dataset of trivia Q&A tasks |
| `train_split` | str | `"train"` | Source split for training tasks |
| `eval_split` | str | `"train"` | Source split for eval tasks |
| `max_examples` | int? | `None` | Optional cap on tasks yielded |
| `judge_model` | str | `"openai/gpt-4.1-mini"` | Judge model id |
| `judge_base_url` | str | Prime Inference v1 | Judge endpoint base URL |
| `judge_api_key_var` | str | `"PRIME_API_KEY"` | Env var holding the judge API key |
| `embed_model` | str | `"text-embedding-3-small"` | Title-embedding model |
| `embed_base_url` | str | Prime Inference v1 | Embedding provider base URL |
| `embed_api_key_var` | str | `"PRIME_API_KEY"` | Env var holding the embedding API key |
| `corpus_dataset` | str | `"willcb/rare-wiki-pages"` | HF dataset of Wikipedia pages |
| `corpus_split` | str | `"train"` | Corpus split |
| `chroma_db_dir` | str | `.chroma_db/wiki-search` | Path to the persistent ChromaDB store |

## Files

- `wiki_search.py` — environment, tools, prompt, dataset and corpus loading, judge reward
- `pyproject.toml` — package metadata and dependencies
- `README.md` — this file

## Notes and limitations

- Judge stability dominates reward noise — point `judge_model` at a stronger judge for cleaner scores.
- The Chroma index only embeds page **titles**, not bodies, so `search_pages` returns candidate titles by name similarity. The agent is expected to drill in with `view_sections` / `read_section` to actually verify content.
- The first run pays the cost of downloading the corpus and embedding every title; subsequent runs reuse the persistent store at `chroma_db_dir`.
