# Repository Structure

Status: TODO

## Proposed Shape

```text
guides/      # ordered Lab walkthroughs
configs/     # starter configs aligned with prime lab setup
recipes/     # runnable educational examples
reference/   # reusable concept and API-adjacent reference
tracks/      # thin indexes by use case, no duplicate content
skills/      # agent guidance used by Lab workflows
```

## Notes

- `guides/` should be the first path for new users.
- [configs/](../configs/) should remain close to the actual Lab setup output.
- `recipes/` should contain runnable examples that guides can reference.
- `reference/` should avoid duplicating product docs that change often.
- `tracks/` should only link into guides and recipes.

## TODO

- Decide whether existing `cookbook/recipes` should move to top-level `recipes`.
- Decide whether the repo name should stay `lab-cookbook`.
- Decide the public docs sync workflow.
- Decide how to mark planned features in guides.
