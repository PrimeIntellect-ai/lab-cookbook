# Custom Data Pipelines

Status: TODO

## Goal

Build a search environment whose corpus, tasks, and tools are all yours.

The earlier search guide ([Tool Use and Search](../07-tool-use-and-search/README.md)) uses `primeintellect/wiki-search`, where the corpus and tools are already wired up. This guide steps one layer down: you bring the raw documents, build the index, define the tools that read from it, and write the tasks the model is scored on.

## Reader Outcome

TODO: The reader can build an environment from a private or custom dataset — loading documents, indexing them for retrieval, exposing search tools to the model, and scoring rollouts with a mix of rule-based and judge-based rewards.

## Worked Example: patent_search

This guide walks through the `patent_search` family of environments in this repo, which trains search and reasoning over patent-style documents:

- **`basic_patent_q_and_a`** — straightforward Q&A with agentic retrieval over a patent corpus.
- **`advanced_patent_q_and_a`** — harder multi-step retrieval and comparison-style questions.
- **`patent_technical_analysis`** — analysis of claims, innovations, and differentiators.

All three share the same data pipeline pattern: load documents → build a Chroma index → expose retrieval tools → score with an LLM judge. The three envs differ in task difficulty and rubric, not in the pipeline itself.

See [`environments/patent_search/`](../../environments/patent_search/) for the full source.

## The Pipeline

TODO: each section below should land as a fleshed-out walkthrough.

### Source the Documents

- Where the raw data comes from (Hugging Face dataset, scraped corpus, vendor export, internal store).
- Cleaning and normalizing before indexing.
- Splitting documents into chunks the retriever can return.

### Build the Index

- Why Chroma is the default for cookbook-scale retrieval.
- Building the index once vs. building lazily inside `load_environment`.
- Where the index lives (local cache, sandbox, env-relative path) and what survives between rollouts.
- When to swap in BM25, hybrid retrieval, or an external vector DB.

### Expose Retrieval Tools

- Tool shape: `search(query)`, `view(doc_id)`, `view_section(doc_id, section)`.
- Keeping tool surface area small so the model learns the loop quickly.
- Stateful vs. stateless<a href="../../reference/glossary.md#stateful-vs-stateless">¹</a>: when the retriever can be shared and when it needs per-rollout state.
- Hiding implementation details (e.g. embedding clients, session ids) from the model's tool schema.

### Write the Tasks

- Where the question/answer pairs come from when the dataset doesn't ship with them.
- Generating tasks with a stronger model, validating them, and storing them as a static dataset.
- What goes in `prompt`, `answer`, and `info` for retrieval tasks.

### Score with Judges

- When a deterministic check is enough vs. when you need an LLM judge.
- Combining a cheap rule-based reward with a judged reward via `RubricGroup`.
- Caching judge calls during eval iteration.

Cross-link forward: judge design is covered in detail under [Designing Rewards](../02-building-your-first-environment/README.md#designing-rewards).

## Evaluate

TODO: include the standard smoke-eval commands for each of the three envs and note where the index gets built on first run.

```bash
prime eval run primeintellect/basic-patent-q-and-a -m openai/gpt-5-nano
prime eval run primeintellect/advanced-patent-q-and-a -m openai/gpt-5-nano
prime eval run primeintellect/patent-technical-analysis -m openai/gpt-5-nano
```

## Iterate

TODO: how to inspect retrieval-stage failures separately from reasoning-stage failures, and which knobs (chunk size, top-k, judge prompt) move the needle in practice.

## Next

In [Multimodal Environments](../10-multimodal-environments/README.md), you will work with environments that include image inputs and multimodal scoring.
