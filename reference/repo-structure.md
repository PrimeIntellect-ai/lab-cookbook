# Repository Structure

This is a local map of the cookbook workspace.

- [tutorials](../tutorials/) - the sequential on-ramp.
- [recipes](../recipes/) - use-case-driven, end-to-end walkthroughs.
- [configs](../configs/) - runnable Prime eval, GEPA, and training configs.
- [environments](../environments/) - local environment packages used by tutorials and recipes.
- [reference](./) - thin pointers only; do not duplicate managed guidance here.
- [`.prime/skills`](../.prime/skills/) - managed Lab skills.
- [`.prime/lab/templates`](../.prime/lab/templates/) - managed config templates.

Generated guidance files are refreshed by `prime lab sync`. Cookbook-owned
examples should point to managed guidance instead of restating it.
