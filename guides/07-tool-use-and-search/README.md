# Tool Use and Search

Build an environment where the model has to search before it answers.

In the earlier guides, each task gives the model everything it needs in the prompt. Search environments add another layer: the task asks a question, and the environment gives the model tools for finding evidence in a corpus.

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
```

```bash
prime eval run configs/07/wiki-search-eval.toml
```

The first run may spend extra time building or loading the search index before rollouts begin.

## The Search Pattern

A search environment usually has four pieces:

- **Tasks**: questions and answers, plus any metadata needed for scoring.
- **Corpus**: the documents, pages, files, or records the model can search.
- **Tools**: the interface the model uses to explore the corpus.
- **Metrics**: the scoring signal computed from the completed rollout.

In `wiki-search`, the tools are intentionally small:

- `search_pages(query)` returns matching Wikipedia page IDs and titles.
- `view_sections(page_id)` lists sections inside a selected page.
- `read_section(section_id)` returns the text for one section.

This gives the model a natural retrieval path: search broadly, choose a page, inspect its structure, read the useful section, then answer.

## The Taskset Shape

The `wiki-search` implementation follows the Taskset pattern. The Taskset is where tasks, tools, prompts, and rewards come together.

Conceptually, `load_environment` looks like this:

```python
import verifiers.v1 as vf


def load_environment(config: vf.EnvConfig) -> vf.Env:
    cfg = WikiSearchTasksetConfig(config.taskset)
    toolset = vf.Toolset(tools=[search_pages, view_sections, read_section])
    return vf.Env(
        taskset=vf.Taskset(
            source=build_source(cfg),
            system_prompt=SYSTEM_PROMPT,
            toolsets=[toolset],
            rewards=[judge_reward],
            config=cfg,
        ),
        harness=vf.Harness(config=config.harness),
    )
```

The source yields task rows with a user prompt, answer, example ID, and turn limit. The toolset exposes the search tools that look up content in a Wikipedia index built once at process start. `judge_reward` is a module-level `@vf.reward` function that pulls the env's primary endpoint via `state.get_endpoint_config(api="chat")` to call a judge, then scores the model's final answer against the reference.

You do not need a custom harness for this pattern. The default tool-use loop is enough: the model calls tools, receives tool results, and eventually responds with a final answer.

## Design the Tools

Good search tools should give the model enough control to solve the task without giving away the answer.

For a document search environment, start with a small tool surface:

- one tool to search or filter the corpus
- one tool to inspect candidate results
- one tool to read the selected content

Avoid tools that do the reasoning for the model. In `wiki-search`, the environment can retrieve pages and sections, but the model still has to decide what evidence matters and compose the answer.

## Design the Reward

Search rewards need to separate retrieval quality from answer quality.

For a first version, score the completed rollout by final-answer correctness. That is what `wiki-search` does with a judge reward: it gives the judge the question, reference answer, and model response, then asks whether the response is correct and coherent.

As the environment matures, add diagnostics around the trajectory:

- Did the model call search?
- Did it read any sections before answering?
- Did it cite or use information from the retrieved text?
- Did it stop too early or hit the turn limit?

Keep those as metrics until you are confident they should affect training. A metric that is useful for debugging is not always a good reward.

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

The command starts a Hosted Training run and prints a run id plus the command for streaming logs.

During training, watch both reward and trajectories. A good run should show the model searching more directly, reading fewer irrelevant sections, and answering from evidence more consistently.

## Next

In [Synthetic Agent Environments](../09-synthetic-agent-environments/README.md), you will simulate a small world in memory and have an agent interact with it through tools.
