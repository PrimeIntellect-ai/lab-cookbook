# wiki-search

Agentic Wikipedia QA on a curated corpus, built on the verifiers v1 `Taskset` + `Harness` pattern.

The agent is given a trivia question and answers it by navigating a small Wikipedia corpus through three tools: embedding-based title search, section listing, and section reading. The corpus is loaded into a local ChromaDB index — **no live Wikipedia API calls**.

This is a v1 port of the canonical `wiki_search` environment in the `verifiers` repo.

## Environment overview

The stack:

- `WikiSearchTasksetConfig`: dataset, judge, embedding, and corpus defaults
- `vf.Taskset`: prompt rows from `willcb/wiki-trivia-questions-v4`, plus the toolset and judge reward
- `vf.Toolset`: ships the three Wikipedia tools and lazily builds the shared ChromaDB index via `objects`/`bindings`
- `vf.Harness`: runs the default endpoint-backed tool loop

The shared corpus (`wiki` dict containing the Chroma collection and `page_id → title` / `page_id → content` maps) is injected into each tool through `Toolset.bindings` rather than module-level globals or closures.

## Tools exposed to the model

- `search_pages(query)` — top-10 article candidates via title-embedding similarity over the ChromaDB index. Returns `[{page_id, title}, ...]`.
- `view_sections(page_id)` — parses Markdown-style `#` headings in the page content and returns the available `{section_id, section_name}` entries. Falls back to a single `:full` section if the page has no headings.
- `read_section(section_id)` — returns the slice of the page content for `section_id` (or the full page if `section_id` ends in `:full`).

## Datasets

- **Questions**: `willcb/wiki-trivia-questions-v4` (HF, `train` split)
- **Corpus**: `willcb/rare-wiki-pages` (HF, `train` split), indexed into a persistent ChromaDB collection (`wiki_titles`) under `.chroma_db` on first run.

The index is built lazily — the corpus + collection load runs the first time a rollout needs the tools, which allows multiple env instances to share work without colliding at construction time.

## Reward design

A single judge reward (weight `1.0`): a `gpt-4.1-mini` yes/no on whether the final response is correct and coherent given the ground-truth answer. Incoherent responses score 0 even if the answer is buried inside them.

The judge call lives in a `@vf.update` handler (`score_with_judge`) that receives the `AsyncOpenAI` client and model name through the same `Toolset.bindings` mechanism the tools use. The reward function (`judge_reward`) just reads `state["judge_score"]` — no factory, no closure-captured client.

## Required environment variables

- `OPENAI_API_KEY` — used by both the judge and the embedding model. Override with `judge_api_key_var` / `embed_api_key_var` if you point either component at another provider.

Keys are validated by a Pydantic `model_validator` on `WikiSearchTasksetConfig`, so missing env vars fail fast at config construction time.

## Quickstart

Run an eval with defaults:

```bash
prime eval run wiki-search
```

Configure model and sampling:

```bash
prime eval run wiki-search \
  -m openai/gpt-4.1-mini \
  -n 20 -r 3
```

The first run downloads the corpus and builds the Chroma index; subsequent runs reuse `.chroma_db`.

## Environment arguments

All fields live on `WikiSearchTasksetConfig` and can be overridden through the v1 config pipeline (TOML or `-a` JSON for fields under `taskset.*`).

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `dataset_name` | str | `"willcb/wiki-trivia-questions-v4"` | HF dataset of trivia Q&A rows |
| `dataset_split` | str | `"train"` | Split used as the prompt source |
| `max_examples` | int? | `None` | Optional cap on rows yielded |
| `max_turns` | int | `10` | Per-rollout turn cap |
| `judge_model` | str | `"gpt-4.1-mini"` | Judge model id |
| `judge_base_url` | str | OpenAI v1 | Judge endpoint base URL |
| `judge_api_key_var` | str | `"OPENAI_API_KEY"` | Env var holding the judge API key |
| `embed_model` | str | `"text-embedding-3-small"` | Title-embedding model |
| `embed_base_url` | str | OpenAI v1 | Embedding provider base URL |
| `embed_api_key_var` | str | `"OPENAI_API_KEY"` | Env var holding the embedding API key |
| `corpus_dataset` | str | `"willcb/rare-wiki-pages"` | HF dataset of Wikipedia pages |
| `corpus_split` | str | `"train"` | Corpus split |
| `chroma_db_dir` | str | `.chroma_db` | Path to the persistent ChromaDB store |

## Files

- `wiki_search.py` — environment, tools, prompt, dataset and corpus loading, judge reward
- `pyproject.toml` — package metadata and dependencies
- `README.md` — this file

## Notes and limitations

- Judge stability dominates reward noise — point `judge_model` at a stronger judge for cleaner scores.
- The Chroma index only embeds page **titles**, not bodies, so `search_pages` returns candidate titles by name similarity. The agent is expected to drill in with `view_sections` / `read_section` to actually verify content.
- The first run pays the cost of downloading the corpus and embedding every title; subsequent runs reuse the persistent store at `chroma_db_dir`.
