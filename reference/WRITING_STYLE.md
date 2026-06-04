# Writing Style

> Early draft, iterating.

This document describes best practices for writing and maintaining technical documentation, both for humans and agents, for projects in the Prime Intellect ecosystem. It is written for an agent that takes on local technical writing tasks: rewriting a README, drafting a skill, maintaining documentation, writing a section of a guide, revising a docstring, or polishing a runbook.

Most of this work is editing in context. The agent inherits a project with established documents, conventions, voice, and reader expectations. New sections sit next to existing ones, and new documents live next to existing documents. The first move on any writing task is reading what is already there.

The reader is usually one of four kinds: a visiting user evaluating the project or referencing documentation, a contributor preparing to make a change, or a coding agent absorbing the doc into its own behavior. Each has different priors, different goals, and different tolerance for friction. Writing for one audience often means writing differently for another, and most documents have more than one reader.

The work depends on several considerations: who reads the document and in what state, what kind of document it is, what surrounds it, what gets included, what order the included material goes in, what voice the prose takes, how confident the claims are, and what survives revision. None of these is a master frame. They are the topics this document develops in turn.

## Reading what is already there

Before drafting anything, read the project's existing documents. They have already shaped what the reader expects when they arrive at yours.

The voice of nearby docs is the most consequential thing to read. If the surrounding documentation is terse and declarative, a chatty new section will feel wrong to readers even when its content is correct. The same is true in the other direction. Voice drift between sections is the most common failure when agents revise existing documentation, and it almost always traces back to skipping this step.

Look for content overlap as well. If the project already has a tutorial that covers basic installation, the new doc should link to it rather than duplicate it. Duplicated documentation drifts over time and produces contradictory instructions. Linking and writing only the delta is almost always the right move.

Notice the project's terminology. If other docs use a specific term consistently — `environment` rather than `env` rather than `task suite` — use the same term. Inconsistent terminology produces the same kind of stumbling that voice drift does: the reader feels the inconsistency without being able to articulate what is wrong.

The rest of this document is easier to apply once the surroundings have been read.

## Who reads, and what they need

A document is written for someone specific. The first job is to identify that person and what they need. Most documents in this ecosystem have one of three primary audiences (users, maintainers, agents), plus a fourth case where multiple audiences overlap. The right writing is very different for each.

### End users

End users come to documentation with a specific goal and limited tolerance for friction. They are usually in one of two states. They are evaluating: deciding whether this project is worth engaging with at all. Or they are acting: trying to accomplish something concrete and currently blocked or unblocked by the docs.

For evaluators, the most expensive thing the document can do is fail to answer the evaluation question quickly. A README that opens with two paragraphs about the project's design philosophy is asking for attention the reader has not yet decided to invest. The relevant questions for evaluation are: what is this, who is it for, what does it look like in use, and where do I go next? The first paragraph of a README should answer the first three. Everything else can wait, and most of it should live in linked docs rather than the README itself.

For actors, the most expensive thing is writing that explains when the reader needs to act. A user who hits an error and pastes it into search is not looking for context; they are looking for what to do next. An error message that reads "Operation failed (please consult logs)" sends a blocked user on a side quest. An error message that reads "Upload failed: file size 41 MB exceeds limit 25 MB; compress or split, then retry" lets them act immediately, without leaving the message.

The voice that end-user docs work best in is plain, direct, second-person, and imperative. "Run `cmd`" rather than "the user can run `cmd`." Imperatives carry better than descriptions because the reader is going to do something with the words. Hedging on instructions ("you may want to consider running...") is hostile when the reader has decided to commit to following them, and it is almost always reducible without loss.

Examples in end-user docs should be runnable as written. Pseudo-code with `your_value_here` placeholders forces the reader to translate, which is friction; concrete examples with real values do not. The cost of using the project's actual values is usually small, and the benefit is large.

End-user documentation also fails gracefully. The reader will deviate from the happy path more often than not. Cover the obvious failure modes — wrong input format, missing config, common environment issues — inline at the point they would occur, not in a separate "troubleshooting" section that the reader has to find while blocked.

### Maintainers and contributors

Maintainers and contributors come with very different priors. They have time, they expect to read code alongside the documentation, and they care about understanding why things are the way they are.

For this audience, rationale carries more weight than instructions. A `CONTRIBUTING.md` that lists style rules without explaining what the rules protect against will be followed inconsistently and revised inconsistently. The same document that explains the failure modes the rules prevent will be understood and maintained.

Conventions are documentation in this audience. A maintainer who notices that test files always import the module they test using `from module import _internal_helper` rather than the public API has learned a convention even when no document said so. The convention doc's job is to catch the contributor who missed the implicit pattern, not to repeat what the code already shows.

Edge cases and trade-offs earn space here in a way they do not in end-user docs. A doc for end users hides edge cases because they distract from the happy path. A doc for maintainers surfaces them because the maintainer needs to know what is brittle. The same fact deserves different treatment in the two docs, which is one reason mixed-audience documents are hard.

The voice for maintainer-targeted writing is denser, more technical, more willing to assume domain knowledge. "Configure the retry policy" is fine for a tutorial; "the retry policy is implemented as a `RetryPolicy(jitter=True, max=5)` instance passed to `RequestQueue.__init__` and is shared across all requests in that queue" is appropriate for a contributor doc.

Maintainer docs benefit from links to source. The reader will read the code anyway; surface the right entry points. A doc that says "see `submit_request()` in `client/request.py:142`" is more useful than one that paraphrases what the function does, because the paraphrase will drift out of sync with the code while the link will not.

### Coding agents

Coding agents are the most distinctive audience, and the one that comes up most often in this ecosystem. They do not consult a document and act independently; they absorb the document into their own context and continue producing text. The doc becomes part of the agent's behavior.

This changes what good writing looks like in two important ways.

First, the form of the prose conditions the agent's output as much as the content. An agent that absorbs a document full of hedging clauses will hedge more in its own writing. An agent that absorbs declarative sentences will write more declaratively. A skill that gestures at thirty abstract principles produces an agent that gestures at principles; a skill that gives five concrete behaviors with discriminating conditions produces an agent that does the right concrete thing. Style is not decoration in agent-targeted writing. It is the policy.

Second, abstract claims often fail to ground. "Be helpful" is undefined; the agent has no operational referent for it. "When the user asks a follow-up question, answer the new question rather than restating context they already have" gives a discriminator the agent can apply at decision time. The unit of agent guidance is a concrete behavior with a triggering condition, not a maxim.

Discriminating examples matter more than illustrative ones. An agent that reads "Use grep when searching for code" will hit the boundary cases the doc did not cover. An agent that reads "Use grep when you know the exact symbol or string; use semantic search when you only know what the code does" has a useful discriminator. The example that earns its place is the one that distinguishes the correct action from a plausible-wrong one. Examples drawn from the canonical center confirm what the agent would have done anyway and teach little.

Contradictions hurt agent-targeted documents more than human-targeted ones. A human can read two contradictory passages and reconcile them mentally. An agent reading both into context experiences the contradiction as policy underspecification, and downstream behavior becomes unpredictable. Skills and system prompts should be edited for self-consistency more aggressively than human-facing docs.

Length is conditional. An agent's context window is finite and shared across many concerns. A skill that condenses to a hundred well-chosen lines is more effective than a thousand-line one that says the same thing with more padding, because more of the agent's available attention is on the policy rather than on filler.

### When the audience is mixed

A README is the canonical mixed-audience document. A single page is read by visitors evaluating the project, contributors preparing to install or modify, and maintainers checking conventions or troubleshooting. Trying to serve all three in every paragraph produces text that serves none of them well.

The right move is structural, not stylistic. Decide which audience the document most exists to serve — usually first-time visitors, in the case of a README — and write the body for that audience. Serve the others through dedicated sections, callouts, or links to separate documents.

Most projects' README problems are visible at a glance. A first-time visitor lands and sees a paragraph about implementation details they have no slot for; a contributor scrolls past five paragraphs of project history to find the dev setup; a maintainer hunts for the build command among feature descriptions. The fix is rarely to write better paragraphs. It is to put each audience's content under a heading they would scan to.

## What kind of document this is

The audience is one axis. The document type is another, and a different one. A README and a tutorial are both written for end users in many cases, but they have different jobs and very different demands. A docstring and a reference page are both consulted by people looking up facts, but the contexts in which they are read are different enough that the same content rarely belongs in both. The document type determines what belongs in the document, what order it goes in, and what kind of incompleteness is honest.

### READMEs and orientation pages

A README is a decision document. The reader has just landed and is deciding, in seconds, whether this project is what they need and where to go next. A README that opens with an architecture overview has answered the wrong question — the reader has not yet decided to invest the attention an architecture overview demands.

The order that respects this reader: what this is in one or two sentences, who it is for, what it looks like in use (the smallest possible runnable example), and where to go from here for each kind of next reader. Anything beyond that goes into linked docs. A README that tries to be the full documentation has confused itself with what it should be linking to.

The smallest possible runnable example matters more in a README than almost anywhere else. It is the fastest way to convey what the project does, and it makes the difference between a reader who engages further and one who closes the tab. The example should be honest — runnable as written, with real values — not a schematic.

### References (API, CLI, config)

A reference page is looked up, not read. A reader who reaches it knows what they are looking up, wants the fact stated exactly, and wants to leave. The structure that serves this is search-shaped: section titles are what someone with the question would type, not topic labels. The fact is the first thing on the page; context, if any, comes after.

What kills a reference is prose around the fact. A configuration option whose page opens with a paragraph about the broader system has failed its reader before they reach the spec. A flag described conceptually instead of by its type, default, and exact behavior forces every reader to translate.

What earns elaboration in a reference is not central behavior but boundaries: edge cases, behavior under error, defaults that surprise, interactions between options. "If both `--source` and `--source-file` are set, `--source` wins" is the kind of line that makes a reference worth opening — it answers a question the reader would otherwise have to discover by experiment.

### Tutorials and how-tos

A tutorial is read linearly, and the reader's state evolves as they read. The contract is not just "they reach a working result" but "they understand why each step works." A tutorial that produces a working artifact through commands the reader cannot reproduce in a new situation has failed: they have a result without a model.

The structure is not "fastest path to the result." It is "shortest path to the result while building the model the reader will need to extend it." Sometimes that means a longer path with explanatory detours; sometimes a shorter one with deferred concepts. The discriminator is whether the reader can do something new at the end, not just repeat what they followed.

The most common tutorial failure is explaining concepts before the reader has the context to receive them. A tutorial that opens with a section titled "Concepts" is almost always organized backward. Give them the experience first, then name what they did.

### Runbooks

A runbook is for action under pressure. The reader is already in the failure the runbook addresses; they have no time and no patience for context. The opening of a runbook is the first diagnostic command, not a description of what kind of failure this addresses.

Runbooks fail when they explain. A runbook for a database failover that pauses to explain the consensus protocol while the cluster is mid-failover is writing for a reader who has time, but the actual reader has none. If the rationale matters at a step, give it in one line; if it deserves more, link to a separate explanation rather than inlining it.

Lift the warnings. A step that can cause data loss earns its own line with the warning visible, not buried in a paragraph. Reversibility is the variable: irreversible steps deserve more visibility and more confirmation than reversible ones.

### Skills and system prompts

A skill or system prompt is a special kind of document because the reader is a model that absorbs the doc into its disposition. The test of the document is behavioral: does the agent, after reading it, do the thing? Style is not measured by how it reads but by what it produces.

What works in skill writing: concrete behaviors paired with the conditions that trigger them, discriminating examples that distinguish correct from plausible-wrong actions, and ruthless attention to length. Every token competes for the agent's attention with the rest of its context.

What fails: abstract principles ("be precise," "be helpful") that the agent cannot ground in action, contradictions that leave the policy underdefined, and padding that signals the wrong style to the agent absorbing it. A skill written loosely teaches looseness. A skill written tightly teaches tightness. The form is half the lesson.

### Inline writing

Inline writing — docstrings, error messages, commit messages, comments — lives next to code. The signature, the error context, the diff, the surrounding code have already done most of the conditioning by the time the reader's eyes reach the words. Inline writing adds what those did not.

A docstring on `parse_iso_timestamp(s: str) -> datetime` should not say it parses ISO timestamps; the signature said it. The docstring should say what the signature cannot: that the parser raises on missing time components, that the returned `datetime` is naive rather than UTC-aware, that the parser accepts the `Z` suffix but not numeric offsets. Restating the signature is one of the most common docstring failures.

An error message is read by someone who is blocked. They have a goal, an action that just failed, and need to know what to do next. The discriminator: a great error message is a next action, not a description. "Invalid input" is an indictment without a remedy. "Field 'email' is required and was missing from the request body" tells them exactly what to fix.

A commit message is read by a reviewer or someone bisecting later. The diff already shows what changed; the message says why. "Update parser" wastes the line. "Reject trailing commas in the JSON parser to match RFC 8259 — clients sending Postel-style payloads now error explicitly instead of silently dropping fields" gives the reviewer the reason and the user-visible impact in one breath.

A code comment that restates the code is noise. A comment that explains why the code is the way it is — the constraint, the bug, the historical contract that justifies an otherwise odd shape — earns its space. `# increment counter` is condescension; `# Counter must be incremented before publish() to preserve at-least-once semantics if publish() raises` is necessary.

## Choosing what to include

Choosing what goes in the document and what stays out is the highest-leverage decision in technical writing. Most other decisions follow from it.

The criterion is the reader's prior knowledge. Words about things the reader already knows are wasted. Words about things they don't know but wouldn't notice missing produce gaps that may or may not matter. The most valuable words are the ones that cover places where the reader has a confident default that happens to be wrong, because those are the gaps the reader will not feel and will not check on their own. They will pass through the writing acting on the wrong belief.

A common writer-side failure here is treating your own prior as the reader's. Most things that seem obvious to the writer are obvious because the writer has the context, not because the reader does. The high-leverage elaboration is at the points where the reader's default would fail; finding those points requires actively imagining the reader's state, not just listing what feels important about the topic.

The surroundings have already done part of the work. The function signature has named the parameters. The page that linked here has set the frame. The error has been thrown by an action the reader just took. None of that needs to be repeated in the sentences you add. Inline writing fails most often by restating what the surroundings already supplied.

Linking is an alternative to writing. If the project already documents installation in another file, link to that doc rather than duplicate it. Duplicated documentation drifts over time, and the drift produces contradictory instructions.

Length should scale with consequence, not topic importance. A reversible action earns one line even when it is a marquee feature. A destructive action earns repetition: a warning lifted onto its own line, the same warning in the help text and in the confirmation prompt. "This permanently deletes data" stated three times in three places is the right amount of writing for what the reader stands to lose.

## Specificity

The fluent default is smooth and general. The information that documentation actually provides is in what smooth, general writing skips: the specific number, the exact name, the precise condition under which something happens. Specificity is harder to write than fluency, and the gap between the two is where most technical writing fails.

"Configure the timeout appropriately" is invisible. "Set `timeout_seconds` to 30 for interactive use and 120 for batch jobs" is the documentation. The first sounds correct without conveying anything. The second tells the reader what to do. Where you do not have specific values, say so plainly. Marking a number as unknown is more useful than gesturing at it.

Smooth-but-vague writing is the most dangerous register because it propagates. It sounds polished, so it gets repeated, and the imprecision survives across copies and versions. Imprecise writing at least signals its imprecision and invites correction. Smooth-but-vague writing does neither, and over time it accumulates into documentation that sounds correct and conveys nothing.

A precise name carries its documentation at every use site. A vague name forces re-explanation everywhere it appears. `--workers` is ambiguous: workers of what kind? `--worker-count` describes the mechanism but not the effect. `--max-concurrent-uploads` describes the effect with no ambiguity. The right name removes a paragraph of explanation that a worse name would have required.

If you find yourself writing a long docstring or comment to explain what something does, check first whether the name should have done that work. Documentation is what you fall back on when naming has failed.

## Order and conditioning

Each sentence depends on what came before it. The reader can only understand what comes next if the prior text has set it up. When a doc says "we'll come back to X later," that is a signal that X should have been introduced earlier. The fix is to reorder.

What can be left unsaid grows as the document accumulates context. Early in a document, the reader has no slot for the terms the writer has defined or for the assumptions the writer is making, so the writer has to be explicit. Later, the same words can carry more meaning because the reader has built up. A document that begins terse has skipped setup, and one that ends verbose has wasted the setup it did.

The structure of a document should follow how the reader uses it, not the order in which the writer figured things out. A reader scanning a reference for a flag name needs the reference to be search-shaped. A reader following a tutorial reads top to bottom and needs sequential shape. A reader trying to understand a system needs an argument that builds the model. Match the structure to how the reader will arrive and move.

The choice between prose, bullets, tables, and code is part of structure. Use bullets when items are parallel and don't depend on each other. Use prose when ideas need to flow and build. Use tables when the reader is comparing values across a stable set of dimensions. Use code blocks when running code is the most direct way to show what prose would have explained. The wrong format is friction.

## Voice

The voice of a new section is set by the surroundings, not chosen freshly. A casual paragraph dropped into a terse reference reads as wrong even when the content is correct, and the same is true in the other direction. Read the local voice before writing and match it.

Voice drift is the most common failure when an agent revises existing documentation. Within a section, the writer re-decides register sentence by sentence: an opening that mirrors the surrounding style, a middle that grows more verbose, a closing that becomes terser. The reader feels the inconsistency without being able to point at what is wrong.

Voice drift is usually a sign that other things were also missed. A writer who did not read the surrounding voice probably also missed the conventions in use, the existing docs that cover related territory, and the audience the surrounding pages were written for.

Breaking voice on purpose is fine when the break is visible: a callout, a distinct artifact type, a clearly marked aside. The failure is drift, not deliberate change.

## Calibration

Match the language of each claim to what you actually know. Reflexive hedging on things you are sure of dilutes the claims that needed your weight, because readers cannot tell calibrated hedging from reflexive hedging. After enough of either, they stop trusting the document. State what you know directly. Mark uncertainty where it exists. Do not mix the two in the same claim.

Hedging is for genuine uncertainty, not for safety. "May," "might," "generally," "could possibly" sprinkled over claims the writer is sure of make the surrounding writing read as uncertain too, including the parts that needed to be confident.

Match the document's surface to its lifespan. A shipped reference should resolve its open questions or cut them. A visible "TBD" in something authoritative misleads the reader, who assumes that authority means completeness. A working draft or design doc, on the other hand, often does its best work by surfacing unknowns. Forcing closure on a draft hides exactly what readers came to see.

Faking the surface in either direction costs the document's trust. A polished page that quietly hedges every claim is dishonest one way; a working draft that pretends to be settled to look professional is dishonest the other way. The reader who notices either starts wondering what else is wrong.

## Revision

First drafts almost always overdo. They include padding, restated points, hedged claims, marketing language, throat-clearing introductions, and sentences that exist to make the writer feel thorough. Revision is part of the work, not an optional polish step.

The first pass commits to substance: claims, structure, examples. The second pass cuts what does not serve. Skipping the second pass produces output that careful readers can recognize as first-draft within a paragraph or two, even when they cannot articulate why the prose feels off.

The hardest part of revision is cutting your own good work: the elegant sentence, the clever aside, the careful explanation you were proud of. The test is whether the sentence helps the reader do or understand the thing they came for. If it does not, it comes out, regardless of what it cost to write.

A particular failure mode of agent revision is self-referential narration: sentences that describe what the document is doing rather than doing it. "This section will cover..." "We will now discuss..." "Below, we explain..." Cut these and let the section do its work.

## Worked examples

The examples below show the principles in this document operating together on writing tasks that come up often in this ecosystem.

### One change, three artifacts

Suppose a function `get_user(id)` is being changed to `get_user(id, *, include_deleted=False)`. Previously it returned soft-deleted users by default. From v3.0, soft-deleted users are excluded unless `include_deleted=True` is passed.

The same fact takes a very different shape in each artifact that has to mention it.

In the **CHANGELOG**, the reader is scanning for what affects them and will read further only if something does. The entry is a single bullet:

> **Breaking changes**
>
> - `get_user(id)` no longer returns soft-deleted users by default. Pass `include_deleted=True` to restore previous behavior.

That is the entry. Anything more — rationale, migration steps, the full call-site impact — would be wasted in the CHANGELOG and is delegated to the migration guide.

In the **migration guide**, the reader is preparing to update their code and needs enough information to do so:

> ## Migrating from `get_user(id)`
>
> Before v3.0, `get_user(id)` returned both active and soft-deleted users. From v3.0, it returns only active users. To restore the previous behavior at a specific call site, pass `include_deleted=True`.
>
> Find affected call sites:
>
> ```
> rg 'get_user\(' --type py
> ```
>
> For most call sites — login, profile rendering, permission checks — the new default is correct, and no change is needed. Audit code, admin tools, and any code that reads soft-deleted records intentionally need `include_deleted=True` added explicitly.

The find-call-sites command belongs in the migration guide because a CHANGELOG reader does not yet know they need it. The migration guide owns the structure of the reader's task.

In an **inline deprecation warning**, the reader is staring at a stack trace and has the least surrounding context of the three:

> ```
> DeprecationWarning: get_user(id) returned a soft-deleted user. From v3.0, soft-deleted users will be excluded by default. Pass include_deleted=True to opt in, or filter on the .deleted_at field if you want to handle deleted users explicitly.
> ```

The inline warning is the longest of the three even though the surrounding text is the sparsest, because the reader has the least ambient context. The same content takes a different correct shape in each artifact. Substituting any one for another would fail its reader.

### A README opening rewritten

A common README failure mode:

> # Workflows
>
> Welcome to Workflows, a powerful and flexible workflow engine for modern data pipelines. Built from the ground up with developer experience in mind, Workflows lets you define, schedule, and monitor complex DAGs with minimal boilerplate. Whether you're orchestrating ETL jobs, ML training pipelines, or business logic, Workflows scales with your needs and integrates seamlessly with your existing infrastructure.

This opening fails on every dimension that matters for a README. It does not tell the reader what the project actually is (what runtime, what language, what kind of scheduling), who it is for, or what it looks like in use. The claims it does make — powerful, flexible, modern, minimal boilerplate, scales, integrates seamlessly — are marketing language the reader cannot act on. A first-time visitor reading this still does not know whether to try the project.

Rewritten:

> # Workflows
>
> Workflows is a Python workflow engine. You define a DAG of tasks in a single file, run it locally with `workflows run pipeline.py`, and monitor it in a web UI at `localhost:8080`. Tasks are pure Python functions; the runtime handles scheduling, retries, and persistence.
>
> ```python
> from workflows import dag, task
>
> @task
> def fetch_data(date: str) -> list[dict]:
>     ...
>
> @task
> def transform(rows: list[dict]) -> list[dict]:
>     ...
>
> @dag(schedule="@daily")
> def pipeline(date: str):
>     transform(fetch_data(date))
> ```
>
> See [Getting Started](docs/getting-started.md) for a full walkthrough, [Concepts](docs/concepts.md) for the underlying model, or [Deployment](docs/deployment.md) to run in production.

The rewrite tells the visitor what the project is, what it looks like in use, and where to go for each next reader. The marketing language is gone. The example is concrete and runnable, not a schematic. The links delegate the rest of the documentation to dedicated pages instead of trying to be all of it. A visitor reading this in five seconds knows whether to keep reading.

### A skill rewritten

A common skill failure mode is gesturing at principles instead of stating concrete behaviors with conditions:

> # Code Search Skill
>
> When searching for code, choose the right tool for the job. Use grep when appropriate. Use semantic search when appropriate. Be thoughtful about which approach is best given the situation. Always be precise and avoid wasting time on the wrong tool.

This skill teaches the agent very little. "Choose the right tool" is undefined. "Be precise" and "avoid wasting time" are abstract; the agent has no operational referent for either. An agent that absorbs this will still face the same decision at runtime with no useful policy.

Rewritten:

> # Code Search Skill
>
> Use **grep** when you know the exact symbol, string, or substring you are looking for. Examples: finding all call sites of `submit_request()`, finding every place a particular error message is constructed, finding which file defines a constant.
>
> Use **semantic search** when you only know what the code does, not what it is called. Examples: finding the code that handles user authentication when you don't know the function or class name, finding code related to a concept that may be implemented across multiple files.
>
> When unsure, try grep first. Grep is cheaper and faster, and most code search questions have a known symbol or string somewhere in them. If grep returns nothing relevant within a few queries, switch to semantic search.
>
> Avoid:
>
> - Using semantic search for known symbols (`submit_request`, `MAX_RETRIES`). Grep is faster and more precise.
> - Using grep for fuzzy concept queries ("authentication code"). It will miss anything not containing those literal words.

The rewrite gives the agent discriminators it can apply at runtime: known-symbol versus concept-only as the criterion for choosing, an order to try them in, and explicit failure cases on each side. A skill written this way produces an agent that does the right concrete thing when faced with a search question. The original produces an agent that gestures at "choosing the right tool" and proceeds without a policy.
