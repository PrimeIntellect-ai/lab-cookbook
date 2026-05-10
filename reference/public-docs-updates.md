# Public Docs Updates

Status: TODO

Use this page to track docs-site changes implied by the cookbook structure.

## Proposed Direction

- Make this repo the source of truth for Lab educational guides.
- Use `public-docs` for navigation, frontmatter, product reference, and docs-site presentation.
- Replace the current `public-docs/guides/*` pages where the cookbook guide is stronger.
- Keep Hosted Training model tables, pricing, advanced config reference, API docs, and CLI reference in `public-docs`.

## Pages to Rework or Remove

- `public-docs/guides/rl-training.mdx`
  - TODO: split into setup, first eval, first RL run, deployment.
- `public-docs/guides/search-agents.mdx`
  - TODO: rework into `guides/07-tool-use-and-search`.
- `public-docs/guides/recipes.mdx`
  - TODO: convert into a recipes index or training-patterns reference.
- `public-docs/guides/index.mdx`
  - TODO: rebuild around the new Lab guide sequence.
- `public-docs/verifiers/evaluation.mdx`
  - TODO: update eval persistence docs. Results are saved by default, so examples and option tables should not imply `-s` is required.
- `public-docs/tutorials-environments/evaluating.mdx`
  - TODO: update `--save-results` parameter wording for the current default behavior.
- `public-docs/verifiers/faqs.mdx`
  - TODO: fix stale `-s` language. Results are saved automatically.

## Navigation TODO

- Decide whether docs should expose a top-level `Lab` tab or keep Lab under `Guides`.
- Add ordered guide navigation once the first guides are filled in.
- Add redirects from old guide URLs.
- Link docs reference pages back to relevant cookbook guides where useful.

## Branding TODO

- Lean into `Lab` as the product name.
- Use the Lab loop as the conceptual frame: specify tasks, evaluate, train, inspect, deploy, repeat.
- Keep copy concrete and operational.
- Avoid burying Lab behind generic "training recipes" language.
