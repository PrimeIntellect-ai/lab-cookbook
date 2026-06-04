# Tool Use and Search

Build an environment where the model has to search before it answers.

In the earlier guides, each task gives the model everything it needs in the prompt. Search environments add tools for exploring a corpus before answering. Final-answer scoring uses the judge wiring from [Judges and Instruction Following](../07-judges-and-instruction-following/README.md).

Tools add a new kind of turn to the [rollout loop from guide 04](../04-prompt-optimization/README.md#how-a-multi-turn-rollout-runs). In wordle the environment replied with a user message; with tools, the model's response can instead be a *tool call*. When it is, the harness runs the tool and feeds its result back as the next turn, then the loop continues. The model decides each turn whether to call a tool or answer; the rollout ends when it answers (or hits a stop condition). You write the tools; the harness handles the call/result plumbing.

This guide uses [prime/wiki-search](https://app.primeintellect.ai/dashboard/environments/prime/wiki-search), a Wikipedia search environment on the Environments Hub. The model gets a trivia question, searches a small Wikipedia corpus, reads relevant sections, and answers from the evidence it finds.

## Evaluate the Hub Environment

Run a small eval:

```bash
prime eval run prime/wiki-search \
  -m openai/gpt-5.4-nano \
  -n 5 \
  -r 2 \
  -t 2048
```

Or run with a config file:

```toml
# [configs/08/wiki-search-eval.toml](../../configs/08/wiki-search-eval.toml)
model = "openai/gpt-5.4-nano"
save_results = true

[[eval]]
env_id = "prime/wiki-search"
num_examples = 5
rollouts_per_example = 2
sampling_args = { max_tokens = 2048 }

[eval.taskset]
max_examples = 250
chroma_db_dir = ".chroma_db/wiki-search-small"

[eval.harness]
max_turns = 8
```

```bash
prime eval run configs/08/wiki-search-eval.toml
```

The first run may spend extra time building or loading the search index before rollouts begin.

Use taskset and harness overrides to change the local corpus and rollout budget in eval TOML:

```toml
[[eval]]
env_id = "prime/wiki-search"

[eval.taskset]
max_examples = 250
chroma_db_dir = ".chroma_db/wiki-search-small"

[eval.harness]
max_turns = 8
```

## How Wiki-Search Is Built

[environments/wiki_search/wiki_search.py](../../environments/wiki_search/wiki_search.py) has four pieces: a config with all endpoint and corpus knobs, a small dataclass that holds the loaded search index, a taskset that exposes three tools and one reward, and the standard loader trio.

**Config.** Every external dependency — corpus, embedding model, judge model — is a typed field with a default:

```python
class WikiSearchTasksetConfig(vf.TasksetConfig):
    dataset_name: str = "willcb/wiki-trivia-questions-v4"
    train_split: str = "train"
    eval_split: str = "train"
    max_examples: int | None = None
    judge_model: str = "openai/gpt-4.1-mini"
    judge_base_url: str = "https://api.pinference.ai/api/v1"
    judge_api_key_var: str = "PRIME_API_KEY"
    embed_model: str = "text-embedding-3-small"
    embed_base_url: str = "https://api.pinference.ai/api/v1"
    embed_api_key_var: str = "PRIME_API_KEY"
    corpus_dataset: str = "willcb/rare-wiki-pages"
    corpus_split: str = "train"
    chroma_db_dir: str = ".chroma_db/wiki-search"
    system_prompt: vf.SystemPrompt = (
        "Use the provided Wikipedia search tools to help answer questions."
    )
```

Per-run overrides — pointing at a different corpus, smaller index dir, alternate judge — go under `[eval.taskset]` in TOML without touching code.

**Loaded index.** A frozen dataclass holds the artifacts produced once when the toolset is built:

```python
@dataclass(frozen=True)
class WikiIndex:
    collection: chromadb.Collection
    page_id_to_title: dict[str, str]
    page_id_to_content: dict[str, str]
```

`collection` is the title-embedding store the search tool queries; the two dicts hold raw page content for the read/inspect tools. Module-level `load_wiki(config)` walks the corpus dataset, opens a `PersistentClient` at `chroma_db_dir`, embeds any titles that aren't already indexed, and returns one `WikiIndex`. It runs once per env load — not per rollout, not per tool call — so it earns module-level placement.

**Taskset, tools, and reward.** The taskset class owns everything else. `load_toolsets` is where the index materializes and a process-wide semaphore caps in-flight embedding calls:

```python
class WikiSearchTaskset(vf.Taskset[WikiSearchTasksetConfig]):
    wiki: WikiIndex
    chroma_semaphore: asyncio.Semaphore

    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        source_split = self.config.train_split if split == "train" else self.config.eval_split
        dataset = load_dataset(self.config.dataset_name, split=source_split)
        if self.config.max_examples is not None:
            dataset = dataset.select(range(min(self.config.max_examples, len(dataset))))
        return dataset

    def load_toolsets(self, config: WikiSearchTasksetConfig) -> vf.Toolsets:
        vf.ensure_keys([config.embed_api_key_var])
        self.wiki = load_wiki(config)
        self.chroma_semaphore = asyncio.Semaphore(100)
        return {
            "wiki": vf.Toolset(tools=[self.search_pages, self.view_sections, self.read_section])
        }
```

`load_tasks` returns a `Dataset` whose columns already match the task contract (`question`, `answer`); the framework derives the model prompt from `question`. The `max_examples` knob trims the iterator without forcing a full materialization.

`load_toolsets` is the right place to validate the embedding key (the corpus pipeline can't run without it) and to bind the shared chroma semaphore. The semaphore is a *process-wide* rate limit, not per-rollout state — a single `Toolset` instance is shared across all concurrent rollouts, so any mutable per-rollout data must come from injected `task`/`state` instead.

The three tools are async methods on the class. Each one's docstring becomes the model-visible description; type hints become the parameter schema:

```python
async def search_pages(self, query: str) -> vf.JsonData:
    """Search for top 10 relevant articles using title embedding similarity."""
    async with self.chroma_semaphore:
        results = await asyncio.to_thread(
            self.wiki.collection.query, query_texts=[query], n_results=10
        )
    ...
    return {"pages": pages}

async def view_sections(self, page_id: str) -> vf.JsonData:
    """View the sections of a page."""
    ...

async def read_section(self, section_id: str) -> str:
    """Read a section of a page."""
    ...
```

Three details worth copying:

- **Wrap sync calls with `asyncio.to_thread`.** Chromadb's `collection.query` is synchronous; calling it directly would block the event loop and serialize every concurrent rollout. `asyncio.to_thread` releases the loop for the duration of the call.
- **Gate sync-blocking calls with a semaphore.** Even off-loop, a thousand concurrent chroma queries will saturate the embedding endpoint. `asyncio.Semaphore(100)` caps the in-flight count without changing the surface tools expose.
- **No injected args on the tool signatures.** `search_pages(query)` — that's it. The framework injects `task`/`state`/`sandbox`/`runtime` only when a tool declares them as parameters; this taskset doesn't need them, so they don't appear in the schema or the call.

The reward is the same direct-judge pattern from guide 07 — a `@vf.reward` method that builds its own `AsyncOpenAI`, prompts a yes/no judge, and closes the client.

## Train on Search

Once the eval shows that the environment is healthy, train against the same Hub environment.

Use [configs/08/wiki-search-rl.toml](../../configs/08/wiki-search-rl.toml):

```toml
# [configs/08/wiki-search-rl.toml](../../configs/08/wiki-search-rl.toml)
model = "openai/gpt-oss-20b"

max_steps = 100
batch_size = 256
rollouts_per_example = 8

[sampling]
max_tokens = 2048
reasoning_effort = "low"

[[env]]
id = "prime/wiki-search"
```

```bash
prime train configs/08/wiki-search-rl.toml
```

## Next

In [Multimodal Environments](../09-multimodal-environments/README.md), you will work with environments that include image inputs and multimodal scoring.
