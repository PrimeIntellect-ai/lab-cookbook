# wiki-search

A multi-hop Wikipedia retrieval QA environment built on the verifiers v1 `Taskset` + `Harness` pattern.

The agent is given a HotpotQA-style question that usually requires combining facts from **multiple** Wikipedia articles. It can issue parallel `search_wikipedia` and `read_page` calls, then must return a short final answer citing the source pages it used.

## Why this environment exists

Wikipedia QA is a classic open-domain retrieval task. Multi-hop questions like _"Which film directed by the screenwriter of `Memento` was released in 2001?"_ require the agent to:

1. identify the entities involved,
2. search for them in parallel,
3. read enough of each page to verify the claim,
4. synthesize the answer.

This environment rewards correctness, source attribution, and efficient search.

## Environment overview

The environment is implemented in `wiki_search.py` as a v1 `Taskset` plus the default endpoint-backed `Harness`.

The stack is:

- `WikiSearchTasksetConfig`: dataset, judge, prompt, and Wikipedia API defaults
- `vf.Taskset`: exposes train/eval rows, the toolset, and reward signals
- `vf.Toolset`: ships rollout-scoped Wikipedia search/read tools (no sandbox)
- `vf.Harness`: runs the default endpoint-backed tool loop

See the verifiers docs for more on tasksets and harnesses:
https://docs.primeintellect.ai/verifiers/byo-harness

## Tools exposed to the model

Two tools, both async HTTP calls against `https://en.wikipedia.org/w/api.php`:

- `search_wikipedia(query, limit=5)` — Wikipedia search; returns a numbered list of titles with snippets
- `read_page(title, max_chars=4000)` — plain-text extract of a single page, redirects followed, truncated to `max_chars`

The toolset is `scope="rollout"`, but it owns no sandbox state — the tools simply hit Wikipedia directly through a module-level `httpx.AsyncClient`.

## Dataset

The default dataset is `hotpot_qa` (`distractor` config). Each row provides:

- `question` — the multi-hop question (kept verbatim as the user prompt)
- `answer` — the reference answer used by the judge
- `supporting_facts.title` — the Wikipedia article titles required to answer

Defaults cap to `max_train_examples=1000` and `max_eval_examples=200` so evals stay quick. Swap dataset or limits via env args.

## Reward design

Three rewards combine into the rollout score:

- **`correct_answer`** (weight `0.6`) — `gpt-4.1-mini` judge yes/no on whether the response answers the question
- **`cited_titles`** (weight `0.3`) — fraction of the gold supporting titles that appear in the response text
- **`parallel_tool_calls`** (weight `0.1`) — encourages issuing multiple tool calls per turn, capped at ~3

The system prompt asks for a fixed `Sources:` / `Answer:` shape so `cited_titles` can string-match against the titles the agent claims it used.

## Required environment variables

- `OPENAI_API_KEY` — used by the judge model (`gpt-4.1-mini` by default)

Validate via `vf.ensure_keys(...)` is run at taskset construction time.

## Quickstart

Install from the worktree:

```bash
prime env install wiki-search
```

Run an eval with the defaults:

```bash
prime eval run wiki-search
```

Configure model and sampling:

```bash
prime eval run wiki-search \
  -m openai/gpt-4.1-mini \
  -n 20 -r 3
```

## Environment arguments

All fields are top-level on `WikiSearchTasksetConfig` and can be overridden through the v1 config pipeline (TOML or `-a` JSON for fields under `taskset.*`).

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `dataset_name` | str | `"hotpot_qa"` | HF dataset id |
| `dataset_config` | str | `"distractor"` | HF dataset config |
| `train_split` | str | `"train"` | Split used as training source |
| `eval_split` | str | `"validation"` | Split used as eval source |
| `max_train_examples` | int | `1000` | Cap on rows yielded from train split |
| `max_eval_examples` | int | `200` | Cap on rows yielded from eval split |
| `max_turns` | int | `4` | Per-rollout turn cap |
| `judge_model` | str | `"gpt-4.1-mini"` | Judge model id |
| `judge_base_url` | str | OpenAI v1 | Judge endpoint base URL |
| `judge_api_key_var` | str | `"OPENAI_API_KEY"` | Env var holding the judge API key |
| `wiki_api_url` | str | `https://en.wikipedia.org/w/api.php` | MediaWiki API endpoint |
| `wiki_user_agent` | str | (cookbook UA) | Sent on Wikipedia requests |
| `wiki_request_timeout_seconds` | float | `30.0` | httpx timeout per request |

## Files

- `wiki_search.py` — environment, tools, prompt, dataset loading, rewards
- `pyproject.toml` — package metadata and dependencies
- `README.md` — this file

## Notes and limitations

- The `cited_titles` reward is a cheap substring match against gold titles. It rewards listing titles in the response, which the system prompt prescribes, but does not verify the page was actually consulted.
- Reward stability depends on judge stability — switch `judge_model` to a stronger judge for less noisy scores.
- The tools call the live Wikipedia API, so evals are subject to its rate limits. Wikipedia asks for a meaningful `User-Agent`; the default identifies the lab-cookbook recipe.
- The default Hugging Face `hotpot_qa` dataset is gated/heavy; the first load downloads the full split before the cap is applied.
