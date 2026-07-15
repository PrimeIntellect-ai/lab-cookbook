# Recipe — Infinite Tasksets

Every dataset-backed environment has the same three problems: it runs out (train long enough and the model has seen everything), it leaks (public benchmarks end up in pretraining data, silently inflating scores), and its difficulty is whatever it happens to be. There's a clean escape for any task whose instances can be *constructed*: don't load tasks — **generate** them.

In this recipe you'll build a generator-backed taskset: unlimited fresh tasks from a seed, a difficulty dial, and a train/test split that's leakage-proof by construction. It's maybe 60 lines of code, and the pattern powers real environments — this repo's `calendar-scheduling` generates its meeting-scheduling worlds this way, and Hub environments like `primeintellect/count-char` advertise "synthetic difficulty tiers" doing exactly the same thing.

**You need:** tutorials [1](1_setup.md)–[2](2_first_eval.md), plus [Build Your First Environment](5_build_first_environment.md) if you've never built an environment before.

## The idea

A normal taskset's `load_tasks` reads rows from a dataset. A generator taskset's `load_tasks` *computes* rows from a random-number generator. Three ingredients make it work:

1. **A seeded RNG** — same seed, same tasks, forever. Determinism is what keeps "generated" from meaning "irreproducible": a run is fully described by `(seed, num_tasks, difficulty)`.
2. **A difficulty knob** — because you're constructing instances, you control exactly how hard they are, parametrically.
3. **Verifiable-by-construction answers** — you built the instance, so you know its answer. The reward is exact, free, and unhackable.

## Build it

The task: mental arithmetic chains — evaluate `((7 + 12) * 3) - 5`-style expressions, where difficulty is the number of operations. Scaffold a package (`prime env init arith-chains`) and make the taskset:

```python
import random
import verifiers.v1 as vf

SYSTEM = "Compute the value of the expression. Reply with only the final integer."


class ArithTask(vf.Task):
    answer: int


class ArithConfig(vf.TasksetConfig):
    num_tasks: int = 200
    seed: int = 0
    difficulty: int = 3          # number of chained operations


class ArithChainsTaskset(vf.Taskset[ArithTask, ArithConfig]):
    def _generate_one(self, rng: random.Random) -> tuple[str, int]:
        expr, value = str(rng.randint(1, 20)), 0
        value = int(expr)
        for _ in range(self.config.difficulty):
            op, operand = rng.choice("+-*"), rng.randint(2, 12)
            expr = f"({expr} {op} {operand})"
            value = value + operand if op == "+" else value - operand if op == "-" else value * operand
        return expr, value

    def load_tasks(self) -> list[ArithTask]:
        tasks = []
        for i in range(self.config.num_tasks):
            # decorrelate per-task seeds so task i is stable even if num_tasks changes
            rng = random.Random(self.config.seed + i * 1009)
            expr, value = self._generate_one(rng)
            tasks.append(
                ArithTask(idx=i, prompt=f"Compute: {expr}", system_prompt=SYSTEM, answer=value)
            )
        return tasks

    @vf.stop
    async def single_turn(self, trace: vf.Trace) -> bool:
        return trace.num_turns >= 1

    @vf.reward(weight=1.0)
    async def exact(self, task: ArithTask, trace: vf.Trace) -> float:
        reply = (trace.last_reply or "").strip().replace(",", "")
        try:
            return float(int(reply) == task.answer)
        except ValueError:
            return 0.0


__all__ = ["ArithChainsTaskset"]
```

Two lines deserve a second look:

- **`random.Random(self.config.seed + i * 1009)`** — a *fresh RNG per task*, offset by the index (the same trick `calendar-scheduling` uses). Task 17 is the same task whether you generate 20 or 20,000, which means results at different scales stay comparable.
- **The config carries `seed`, `num_tasks`, `difficulty`** — the entire dataset is those three numbers. There is no data file to version, ship, or leak.

For harder domains the shape is identical, plus one extra step: **validate what you generate**. If instances can be degenerate (unsatisfiable, ambiguous, trivially guessable), check each candidate and re-roll until it passes — `calendar-scheduling`'s generator does exactly this (`generate_validated_task`) to guarantee every meeting-scheduling instance actually has a valid answer.

## Use the dials

Everything interesting now happens from the command line, no code changes:

```bash
# a quick look at difficulty 3
prime eval run arith-chains -n 20 --model openai/gpt-5.4-nano

# same model, harder tasks
prime eval run arith-chains -n 20 --model openai/gpt-5.4-nano --taskset.difficulty 6

# a *completely disjoint* set of tasks — just move the seed
prime eval run arith-chains -n 20 --model openai/gpt-5.4-nano --taskset.seed 10000
```

That last one is the leakage-proofing. Your **train/test split is a seed convention**: train on `seed = 0`, evaluate on `seed = 10_000` (far enough that index offsets can't collide), and the eval set is guaranteed unseen — not "probably not in the training data", but *did not exist* until you evaluated. When a taskset is public, you can even keep a private eval seed nobody has ever trained against.

And difficulty becomes an *instrument*: sweep it and you get a capability curve, not a single score —

```bash
for d in 2 4 6 8 10; do
  prime eval run arith-chains -n 50 --taskset.difficulty $d --taskset.seed 10000
done
```

Plot mean reward against difficulty and you can see exactly where a model's arithmetic breaks down — and compare *where the cliff is* across models, which is far more informative than comparing two points.

## Why trainers love this

For RL ([tutorial 3](3_first_rl.md)), generator tasksets solve the two problems that actually kill runs:

- **No exhaustion:** need 512 more tasks at step 400? `num_tasks = 4096` and they exist. The model can't memorize its way through an unbounded set.
- **Difficulty is the curriculum knob.** RL learns fastest on *sometimes-solvable* tasks — all-fail and all-pass groups teach nothing. With a difficulty dial you can park training right at the model's frontier, and embed *multiple tiers at once*:

```toml
[[env]]
name = "arith-easy"
taskset = { id = "arith-chains", difficulty = 3, seed = 0, num_tasks = 1024 }
harness = { id = "default" }

[[env]]
name = "arith-hard"
taskset = { id = "arith-chains", difficulty = 6, seed = 0, num_tasks = 1024 }
harness = { id = "default" }
```

Same environment, two entries, two difficulty tiers — the model gets signal from the easy tier while the hard tier is still mostly out of reach, and keeps getting signal from the hard tier after the easy one saturates. (Weighting the tiers adaptively as the model improves is the natural next step — see the mixing machinery in [Generalist Training](19_generalist.md).)

## Things to try

- Add a `min_value`/`max_value` override to the config and watch how operand size changes the difficulty curve independently of chain length — you now have a two-axis difficulty space.
- Make the eval-seed convention explicit in your env's README ("seeds ≥ 10,000 are reserved for evaluation") — cheap institutional leakage-proofing.
- Study `environments/calendar_scheduling/` in this repo as the grown-up version: procedural generation + validation + tools + an oracle (`optimal_score`) computed *at generation time*, which then powers an `optimality_gap` metric no dataset could provide.
- Browse the Hub for `synthetic-data`-tagged environments to see the pattern in the wild.

## Recap

Generate, don't load: a per-task seeded RNG makes unlimited tasks reproducible; a difficulty field turns one environment into a capability curve and a curriculum; seed ranges give you train/test splits that cannot leak. Any task you can construct and verify — arithmetic, sorting, scheduling, string puzzles, synthetic codebases — can be infinite.
