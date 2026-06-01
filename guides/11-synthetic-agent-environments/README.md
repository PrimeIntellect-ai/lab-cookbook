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
  -m openai/gpt-oss-20b \
  -n 5 \
  -r 2 \
  -t 2048 \
  -a '{"taskset": {"difficulty": "medium"}}'
```

Or run with a config file:

```toml
# [configs/11/calendar-scheduling-eval.toml](../../configs/11/calendar-scheduling-eval.toml)
model = "openai/gpt-oss-20b"
save_results = true

[[eval]]
env_id = "prime/calendar-scheduling"
num_examples = 5
rollouts_per_example = 2
sampling_args = { max_tokens = 2048 }
taskset = { difficulty = "medium" }
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

- `list_attendees()` — names, time zones, importance weights, required vs. optional
- `view_calendar(attendee_id)` — busy blocks for one attendee
- `view_constraints(attendee_id)` — preferred hours, hard hours, day preferences
- `check_window(start, end, room)` — score a proposed window without committing
- `submit_window(start, end, room)` — submit the final answer

Two design choices keep the environment honest:

- **Budget the oracle.** `check_window` is the agent's view of the oracle. Bound the number of calls per rollout (the TUI shows the budget as `Score-check budget`) so the agent cannot brute-force the search space. The remaining budget should be visible in every tool result.
- **Surface remaining turns.** Tool results include the remaining turn count. The agent learns to plan instead of exploring exhaustively.

This is the same Toolset pattern from [Tool Use and Search](../08-tool-use-and-search/README.md), with one extra requirement: the per-rollout state owns the generated world, not just a session handle into an external one. Wire tools that need per-rollout context through `state` rather than module globals; `vf.Toolset` runs each tool with `task` and `state` in scope.

## Designing the Reward

The reward should reflect achieved utility against what was achievable on this specific task:

- if no valid window is submitted, score is 0
- if a valid window is submitted, score is the weighted average of attendee utilities for that window
- attendee weights are normalized to 1 per task, so scores are comparable across tasks

Normalizing against the oracle is optional but useful: it gives `score / oracle_best`, which is bounded in `[0, 1]` regardless of how generous the underlying utilities are. Either form works as an RL reward; the unnormalized form is easier to compare to a random baseline.

Avoid composite rewards with many small bonuses. They invite reward hacking and obscure what the agent learned. One clean reward, computed at submission, is usually enough — diagnostic signals (did the agent call `check_window`, did it submit before the turn limit, did it ever view constraints) belong as metrics, not as reward terms.

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

The environment should have a max_turns parameter, and tool results should show the remaining turns to the agent.
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
- a capable model (gpt-5.5 or similar) scores meaningfully above random
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
args = { difficulty = "medium", num_train = 512, num_eval = 128, max_turns = 18 }
```

```bash
prime train configs/11/calendar-scheduling-rl.toml
```

## Next

In [Custom Harnesses](../12-custom-harnesses/README.md), you will run third-party agent libraries through the program pattern.
