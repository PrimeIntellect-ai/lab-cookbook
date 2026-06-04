# Synthetic Agent Environments

Simulate a small world in memory and let an agent interact with it through tools.

The environments in earlier guides either pulled tasks from a dataset (`reverse-text`, `gsm8k`) or wrapped an external corpus (`wiki-search`). A synthetic agent environment generates the world itself: a Python program builds the task state, exposes tools that read and mutate that state, and scores the rollout against a deterministic specification. There is no dataset to download and no external service to call.

This pattern is a good fit when you want:

- programmatic task generation with fine-grained difficulty controls
- deterministic validation that a solution exists (and what the best possible score is)
- a stateful tool surface that mirrors a real product without the operational cost
- training-scale task variety from a single environment package

The same recipe extends well beyond calendars. The world being simulated can just as easily be a database the agent queries and updates, a spreadsheet it edits cell-by-cell, a website backend it drives through HTTP-like tools, a filesystem, a ticketing system, a CRM, or any other service whose state you can hold in memory. Once you can generate problems and grade solutions, the surface presented to the agent is a design choice.

This guide uses [prime/calendar-scheduling](https://app.primeintellect.ai/dashboard/environments/prime/calendar-scheduling) as the worked example. The model is asked to schedule a meeting across a set of attendees with busy calendars, hard constraints (required attendees, hard local-time bounds, room availability, exact duration), and soft penalties (early/late local hours, day preferences, back-to-back blocks, missing optional attendees). Difficulty is controlled by `easy` / `medium` / `hard` knobs that map to ranges over attendee count, window length, constraint tightness, and constraint mix.

## Try the Hub Environment First

Run a small eval:

```bash
prime eval run prime/calendar-scheduling \
  -m openai/gpt-5.4-nano \
  -n 5 \
  -r 2 \
  -t 2048 \
  --harness.max-turns 16
```

Or run with a config file:

```toml
# [configs/11/calendar-scheduling-eval.toml](../../configs/11/calendar-scheduling-eval.toml)
model = "openai/gpt-5.4-nano"
save_results = true

[[eval]]
env_id = "prime/calendar-scheduling"
num_examples = 5
rollouts_per_example = 2

[eval.sampling]
max_tokens = 2048

[eval.taskset]
difficulty = "medium"

[eval.harness]
max_turns = 16
```

```bash
prime eval run configs/11/calendar-scheduling-eval.toml
```

The environment ships a standalone visualizer that renders the generated problem as a TUI similar to a meeting app. Use it to develop intuition for what the agent sees:

```bash
uv run --project environments/calendar_scheduling \
  calendar-scheduling-tui --show-oracle --difficulty medium --seed 5
```

The visualizer shows attendees, time zones, busy blocks, hard and soft constraints, oracle best windows, and the random-baseline score. If the oracle score is high and the random baseline is low, the task has signal: there are valid solutions and they are not trivially common.

## The Synthetic Pattern

A synthetic agent environment usually has five pieces:

- **Generator**: a function that, given a difficulty and seed, builds a fully specified problem (attendees, calendars, constraints, candidate windows).
- **World state**: in-memory data structures the rollout reads and writes; created fresh per task.
- **Tools**: a small surface the agent uses to inspect the world and submit answers.
- **Oracle**: a deterministic procedure that finds the best achievable score for the generated problem.
- **Reward**: the rollout's achieved score, usually normalized against the oracle.

Generation and oracle scoring run on the same problem object, so the environment can pre-filter unsolvable tasks before the agent ever sees them. This is the key invariant: every task in the taskset is known-solvable, and the best possible score is known up front.

## Designing the Generator

Pick a small set of difficulty knobs that compose cleanly. For calendar-scheduling:

- number of attendees
- consideration window in days
- meeting duration
- which constraint types are active
- tightness of soft constraints

Group those into `easy` / `medium` / `hard` presets that map to ranges over the underlying knobs. Sample within the range using the task seed so the same `(difficulty, seed)` always produces the same problem.

Two properties matter for synthetic generation:

- **Solvable by construction.** Generate a candidate problem, run the oracle, and reject if no valid solution exists or if the oracle score is below a floor. Resample with the same difficulty until acceptance. This keeps the taskset clean without per-task hand-curation.
- **Not trivially solvable.** A random proposal should score poorly on average. If random baseline reward is close to oracle reward, tighten constraints or expand the search window.

The visualizer is the fastest way to confirm both properties. Generate a handful of tasks at each difficulty and check that valid windows exist but are scarce.

## Designing the Tools

Keep the tool surface small and intention-revealing. For calendar-scheduling:

- `check_attendee_calendar(attendee_id, day_index)` — busy blocks for one attendee on one day, or all days with `day_index=-1`
- `view_attendee_constraints(attendee_id)` — preferred hours, hard hours, day preferences, and utility penalties
- `check_proposal(day_index, start_time_utc, duration_minutes, room_id)` — score a proposed window without committing
- `submit_window(day_index, start_time_utc, duration_minutes, room_id)` — submit the final answer

Two design choices keep the environment honest:

- **Budget the oracle.** `check_proposal` is the agent's view of the oracle. Bound the number of calls per rollout (the TUI shows the budget as `Score-check budget`) so the agent cannot brute-force the search space. The remaining budget should be visible in every tool result.
- **Surface remaining turns.** Tool results include the remaining turn count. The agent learns to plan instead of exploring exhaustively.

This is the same `vf.Toolset` pattern from [Tool Use and Search](../08-tool-use-and-search/README.md), with one extra requirement: the generated world belongs on the serializable `vf.Task`, not in a module global or rollout state. Put task metadata under `task["info"]`, initialize rollout-only progress with `@vf.setup`, and let tools declare hidden `task: vf.Task` and `state: vf.State` arguments; the framework injects both while keeping the model-visible schema clean.

## Designing the Reward

The reward should reflect achieved utility against what was achievable on this specific task:

- if no valid window is submitted, score is 0
- if a valid window is submitted, score is the weighted average of attendee utilities for that window
- attendee weights are normalized to 1 per task, so scores are comparable across tasks

Normalizing against the oracle is optional but useful: it gives `score / oracle_best`, which is bounded in `[0, 1]` regardless of how generous the underlying utilities are. Either form works as an RL reward; the unnormalized form is easier to compare to a random baseline.

Avoid composite rewards with many small bonuses. They invite reward hacking and obscure what the agent learned. One clean reward, computed at submission, is usually enough — diagnostic signals (did the agent call `check_proposal`, did it submit before the turn limit, did it ever view constraints) belong as metrics, not as reward terms.

## How Calendar-Scheduling Is Built

[environments/calendar_scheduling/calendar_scheduling.py](../../environments/calendar_scheduling/calendar_scheduling.py) is the calendar example wired up. It imports the world model (`CalendarTask`, generator, oracle, evaluator) from a sibling `calendar_problem` module and exposes one taskset with four tools, one reward, and a handful of metrics.

**Config.** Difficulty and dataset sizing are typed Pydantic fields; generator-level overrides ride along as a nested config:

```python
class CalendarSchedulingTasksetConfig(vf.TasksetConfig):
    difficulty: str = "medium"
    seed: int = 7
    num_train: int = 512
    num_eval: int = 128
    generator_overrides: GenerationOverrides = Field(default_factory=GenerationOverrides)
    system_prompt: vf.SystemPrompt = SYSTEM_PROMPT
```

`generator_overrides` is a structured config object the user can override field-by-field from TOML (`[eval.taskset.generator_overrides]`) without exposing the generator internals.

**Per-rollout setup.** A `@vf.setup` hook fires at the start of every rollout to initialize bookkeeping state — the score-check budget, the running list of proposals examined, and the submission slot:

```python
@vf.setup
async def setup_calendar(self, task: vf.Task, state: vf.State) -> None:
    calendar_task = CalendarTask.from_task(task)
    state["score_checks_remaining"] = int(calendar_task.score_check_budget)
    state["score_checks_used"] = 0
    state["proposal_checks"] = []
    state["submitted"] = False
    state["submitted_valid"] = False
    state["submitted_score"] = 0.0
    state["submitted_payload"] = None
```

`@vf.setup` attaches to the **start** of the [rollout loop](../04-prompt-optimization/README.md#how-a-multi-turn-rollout-runs) — step 1, before the first model turn — and is the right place for per-rollout bookkeeping that tools mutate later. With [guide 10's](../10-coding-agents-and-sandboxes/README.md) `@vf.cleanup` (loop end) and the `@vf.stop` hook below (loop exit), that completes the lifecycle: `@vf.setup` opens a rollout, `@vf.stop` decides when it ends, `@vf.cleanup` closes it. These hooks are how a taskset changes state at loop boundaries; tools change state mid-loop, but only through the `task`/`state` the framework injects.

**Task generation.** `load_tasks(split)` is deterministic in `seed` and `task_index`. Each task is generated and validated up front (the oracle proves a valid solution exists) before being packaged with `build_example`:

```python
def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
    if split == "train":
        num_examples, seed = self.config.num_train, self.config.seed
    else:
        num_examples, seed = self.config.num_eval, self.config.seed + 1_000_003
    tasks: list[vf.Task] = []
    for index in range(num_examples):
        task_seed = seed + (index * 1009)
        task, summary, config = generate_validated_task(
            seed=task_seed,
            difficulty=self.config.difficulty,
            overrides=self.config.generator_overrides,
        )
        tasks.append(build_example(task, summary, config))
    return tasks
```

Train and eval use disjoint seeds so eval examples never appear in training, even by coincidence. `1_000_003` is just a large prime offset; `1009` between tasks keeps each example seed far from its neighbors.

**Tools.** Four `@staticmethod` tools on the taskset class. The model-visible signature names are `attendee_id`, `day_index`, `start_time_utc`, `duration_minutes`, `room_id`. `task` and `state` are added to the signatures but stripped from the schema — the framework injects both by parameter name and the model never sees them:

```python
def load_toolsets(self, config: CalendarSchedulingTasksetConfig) -> vf.Toolsets:
    _ = config
    return {
        "calendar": vf.Toolset(
            tools=[
                self.check_attendee_calendar,
                self.view_attendee_constraints,
                self.check_proposal,
                self.submit_window,
            ]
        )
    }

@staticmethod
async def check_proposal(
    day_index: int,
    start_time_utc: str,
    duration_minutes: int,
    room_id: str,
    task: vf.Task,
    state: vf.State,
) -> str:
    """Check score and hard-constraint status for a candidate window.

    Args:
        day_index: Day index in the planning window.
        start_time_utc: UTC start time in HH:MM format.
        duration_minutes: Proposed duration in minutes.
        room_id: Room identifier.

    Returns:
        JSON with validity, score, attendee utility details, and budgets.
    """
    calendar_task = CalendarTask.from_task(task)
    ...
    if remaining_checks <= 0:
        payload["error"] = "score-check budget exhausted"
        ...
    evaluation = evaluate_proposal(calendar_task, proposal)
    state["score_checks_remaining"] = remaining_checks - 1
    state["score_checks_used"] = int(state.get("score_checks_used", 0)) + 1
    ...
    return json.dumps(payload, indent=2, sort_keys=True)
```

`check_proposal` shows the full pattern: take model arguments, reconstruct the world from `task`, read and decrement the score-check budget on `state`, return a JSON-shaped string the model can re-parse. The score-check budget is the throttle that keeps this from being brute-force search — every probe costs one of a fixed number of checks, and the budget is reported in every tool response.

`submit_window` is the only tool that ends the rollout. Submission flips `state["submitted"] = True` and calls `state.stop("submitted")` so the `@vf.stop` hook below sees it on the next check:

```python
@vf.stop(priority=50)
async def has_submission(self, state: vf.State) -> bool:
    return bool(state.get("submitted", False))
```

`@vf.stop` is the loop's exit check — it runs after each model turn (step 2 of the loop), and returning `True` ends the rollout. `state.stop("submitted")` inside `submit_window` is the imperative form of the same thing: a tool can signal "we're done" directly. `priority=50` runs this stop before the built-in cheap checks (turn limit, error) so a successful submission terminates immediately without burning extra turns. The framework always provides those built-in stops; you add `@vf.stop` only for task-specific exit conditions.

**Reward and metrics.** One reward, several metrics. The reward gates on validity so an invalid submission scores 0 even if the underlying utility computation succeeded:

```python
@vf.reward(weight=1.0)
async def final_score_from_submission(self, state: vf.State) -> float:
    if not bool(state.get("submitted_valid", False)):
        return 0.0
    return float(state.get("submitted_score", 0.0))

@vf.metric
async def submission_valid(self, state: vf.State) -> float:
    return 1.0 if bool(state.get("submitted_valid", False)) else 0.0

@vf.metric
async def submitted_to_optimal_ratio(self, state: vf.State, answer: object) -> float:
    ...  # 0–1 ratio of achieved to oracle score, useful for cross-task comparison
```

`@vf.metric` tracks signals that should appear in rollout outputs but never affect the gradient. Use it for diagnostics like "did the agent submit at all?", "how close to the oracle?", or "how much of the score-check budget did the agent use?". These show up alongside the reward in eval output and in W&B during training, which makes reward hacking easy to spot — if reward climbs but `submission_valid` stays flat, something is broken.

Both reward and metric methods accept any subset of the injected names (`task`, `state`, `answer`, `info`, `prompt`, `completion`) — request only what you read. `submitted_to_optimal_ratio` reads both `state` and `answer`; `submission_valid` reads only `state`.

## One-Shotting It with a Coding Agent

For this class of environment, the design doc *is* most of the work. A capable coding agent given a clear specification can produce a working first version in one pass. The prompt that produced `prime/calendar-scheduling` is reproduced below as a template. Adapt the constraint list, tool surface, and difficulty knobs to your domain.

```md prompt.md wrap
Make an environment for a calendar scheduling agent.
In each task, there should be a set of people with busy calendars, and individual + global constraints for scheduling the meeting.
Some constraints can be "hard" (not allowed to violate), others can be "soft", where violating a constraint incurs some utility cost for certain attendees.
Each attendee has a utility for the proposed meeting time between 0 and 1, and the task score will be the weighted average of attendee scores if an acceptable meeting time is found, and 0 otherwise.
Attendee importance weights should be normalized to 1 for each task.

We should be able to programmatically generate task problems, and deterministically validate that satisfying solutions exist (and what their best possible score would be).
We should have fine-grained controls for key degrees of freedom in task generation, with higher-level parameters ("easy" / "medium" / "hard") for the full task set, which then map into setting ranges for the more fine-grained controls.
Be creative, and use your judgment to design clean composition rules for converting meeting choices and conflicts into scores. Avoid complex branching/conditional logic where possible.
Think carefully about designing your system in a way which discourages "backdoor" strategies or reward hacks.
The best approach for an agent should be to make a good-faith effort to satisfy constraints as best as possible.
Experiment with sampling strategies to ensure that tasks are solvable most of the time (so that we can pre-filter any unsolvable tasks cheaply), and that they aren't too easy -- there shouldn't be an abundance of valid solutions, random proposal times should be a bad strategy.

Types of constraints we want to potentially account for:

- Conflicting schedules
- Time zones + early/late/day preferences
- Meeting length
- Room availability
- Back-to-back meeting preferences
- Desired-but-optional attendees
- Other related constraints which reflect real-world calendar challenges

Degrees of freedom:

- Number of attendees
- Window of consideration
- Types of constraints
- Tightness of constraints

Use the `vf.Taskset` + `vf.Toolset` pattern with per-rollout state for the calendar + attendee information. The agent should have tools for things like:

- Checking attendee calendars
- Viewing attendee constraints
- Checking score of a proposed window
- Submitting a window

Set `max_turns` on the harness config, and make tool results show the remaining turns to the agent.
Default limit should be enough to allow reasonable exploration, but not so high that the agent can brute-force search all times.

We should also have a nice standalone script in the environment which creates a TUI to visualize a "calendar problem" similar to typical meeting apps, including attendees, timeblocks, and constraints, but fully in the terminal, using Rich styling, similar design language to the `prime eval tui` viewer implemented within the `verifiers` library (inspect verifiers source for reference).

Create a detailed design doc and plan for testing (PLAN.md), implement in full, revise PLAN.md after major milestones to reflect accomplishments and updated TODOs, and run basic small evals throughout as needed.
You are welcome to use the PRIME_API_KEY set in my environment for inference tests (see configs/endpoints.toml for models).
Let me know when you're happy with your implementation.
```

The prompt is doing a few specific things worth copying when you write your own:

- it names the env pattern (`vf.Taskset` + `vf.Toolset` with per-rollout state) so the agent does not invent its own
- it lists hard vs. soft constraints separately, which forces the scoring design to follow
- it asks for an oracle and a random-baseline check, which yields the solvability guarantees
- it requests a visualizer, which doubles as a debugging tool during development
- it asks for a `PLAN.md` and incremental evals, which keeps the run honest

## Adapting the Pattern to Other Worlds

The calendar example is the most legible because everyone has scheduled a meeting, but the recipe transfers directly:

- **A simulated database.** Generate a schema and a set of rows. Tools are `list_tables`, `describe_table`, `query`, `insert`, `update`, `delete`. Tasks ask the agent to answer a question or apply a migration. The oracle is the correct query result or the post-migration state.
- **A simulated spreadsheet.** Generate a grid with formulas and values. Tools are `read_cell`, `read_range`, `write_cell`, `write_formula`. Tasks ask for a derived value or a transformation. The oracle is the expected grid state.
- **A simulated website backend.** Generate users, items, orders, carts. Tools are HTTP-shaped: `list_orders(user_id)`, `update_inventory(item_id, qty)`, `process_refund(order_id)`. Tasks are customer-support tickets with a known correct resolution.
- **A simulated filesystem.** Generate a directory tree with file contents. Tools mirror a shell: `ls`, `cat`, `grep`, `write`. Tasks ask for a refactor or a fix. The oracle is the expected final tree.
- **A simulated ticketing system / CRM.** Generate tickets, customers, SLAs, prior interactions. Tools query and mutate ticket state. Tasks ask the agent to triage, route, or resolve.

In every case the structure is the same: a generator produces a self-contained world plus an oracle, the agent interacts through a small tool surface, and a deterministic reward compares achieved state to expected state. The difficulty knobs and constraint vocabulary are domain-specific; the scaffold is not.

## Before Training

Before launching RL on a synthetic environment, check:

- the generator produces only solvable tasks (oracle finds a valid solution every time)
- random or naive baselines score well below the oracle
- a capable model (gpt-5 or similar) scores meaningfully above random
- failures look like genuine reasoning mistakes, not environment bugs or formatting issues
- the score-check budget and turn limit are tight enough to discourage brute-force search

When those conditions hold, training works the same as in earlier guides. Use [configs/11/calendar-scheduling-rl.toml](../../configs/11/calendar-scheduling-rl.toml):

```toml
# [configs/11/calendar-scheduling-rl.toml](../../configs/11/calendar-scheduling-rl.toml)
model = "Qwen/Qwen3-30B-A3B-Instruct-2507"

max_steps = 100
batch_size = 128
rollouts_per_example = 8

[sampling]
max_tokens = 768

[[env]]
id = "prime/calendar-scheduling"

[env.taskset]
difficulty = "medium"
num_train = 512
num_eval = 128

[env.harness]
max_turns = 18
```

```bash
prime train configs/11/calendar-scheduling-rl.toml
```

## Next

In [Custom Harnesses](../12-custom-harnesses/README.md), you will run third-party agent libraries through the program pattern.
