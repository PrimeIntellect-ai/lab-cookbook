# Tool Use and Search

Build an environment where the model has to search before it answers.

In the earlier guides, each task gives the model everything it needs in the prompt. Search environments add tools for exploring a corpus before answering. Final-answer scoring uses the judge wiring from [Judges and Instruction Following](../07-judges-and-instruction-following/README.md).

This guide uses [primeintellect/wiki-search](https://app.primeintellect.ai/dashboard/environments/primeintellect/wiki-search), a Wikipedia search environment on the Environments Hub. The model gets a trivia question, searches a small Wikipedia corpus, reads relevant sections, and answers from the evidence it finds.

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
# [configs/07/wiki-search-eval.toml](../../configs/07/wiki-search-eval.toml)
model = "openai/gpt-5.4-nano"
save_results = true

[[eval]]
env_id = "prime/wiki-search"
num_examples = 5
rollouts_per_example = 2
sampling_args = { max_tokens = 2048 }
taskset = { max_examples = 250, max_turns = 8, chroma_db_dir = ".chroma_db/wiki-search-small" }
```

```bash
prime eval run configs/07/wiki-search-eval.toml
```

The first run may spend extra time building or loading the search index before rollouts begin.

Use taskset overrides to change the local corpus/cache and rollout budget
without editing the environment package:

```bash
prime eval run prime/wiki-search \
  -m openai/gpt-5.4-nano \
  -a '{"taskset": {"max_examples": 250, "max_turns": 8, "chroma_db_dir": ".chroma_db/wiki-search-small"}}'
```

## The Search Pattern

- **Tasks** — questions, answers, metadata
- **Corpus** — documents the tools search over
- **Tools** — search, inspect, read
- **Rewards** — scoring on the completed rollout

In `wiki-search`:

- `search_pages(query)` — page IDs and titles
- `view_sections(page_id)` — section list
- `read_section(section_id)` — section text

## The Taskset Shape

The [wiki-search](../../environments/wiki_search/wiki_search.py) implementation subclasses `vf.Taskset`. Tasks, tools, prompts, and rewards live on the class:

```python
class WikiSearchTaskset(vf.Taskset[WikiSearchTasksetConfig]):
    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        ...

    def load_system_prompt(
        self, config: WikiSearchTasksetConfig
    ) -> vf.SystemPrompt:
        return "Use the provided Wikipedia search tools to help answer questions."

    def load_toolsets(self, config: WikiSearchTasksetConfig) -> vf.Toolsets:
        wiki = load_wiki(self.config)
        ...
        return {"wiki": vf.Toolset(tools=[search_pages, view_sections, read_section])}

    @vf.reward(weight=1.0)
    async def judge_reward(self, task: vf.Task, state: vf.State) -> float:
        ...


def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(
        taskset=vf.load_taskset(config=config.taskset),
        harness=vf.Harness(config=config.harness),
    )
```

`load_tasks(split)` yields tasks with prompt, answer, example ID, and turn limit. `load_toolsets(config)` builds the Wikipedia index. `judge_reward` uses a dedicated judge client — see [Judges and Instruction Following](../07-judges-and-instruction-following/README.md).

## Train on Search

Once the eval shows that the environment is healthy, train against the same Hub environment.

Use [configs/07/wiki-search-rl.toml](../../configs/07/wiki-search-rl.toml):

```toml
# [configs/07/wiki-search-rl.toml](../../configs/07/wiki-search-rl.toml)
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
prime train configs/07/wiki-search-rl.toml
```

## Next

In [Multimodal Environments](../09-multimodal-environments/README.md), you will work with environments that include image inputs and multimodal scoring.
