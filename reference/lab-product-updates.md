# Lab Product Updates

Status: TODO

Use this page to track Lab product and CLI changes assumed by the cookbook.

## CLI TODO

- Add `prime lab view --evals` to open Lab directly on the eval results view.
- Add `prime lab view --training` to open Lab directly on Hosted Training runs.
- Add a GEPA `save_to_environment` boolean config flag. When true, GEPA should save optimized prompts into the local environment's `prompts/` folder if that environment is available in the workspace; otherwise it should warn and leave the artifact in the GEPA results directory.

## Environment TODO

- Publish or verify the custom-harness guide environments under the intended Hub IDs:
  - `primeintellect/langchain-deep-agents-env`
  - `primeintellect/dspy-rlm`
  - `primeintellect/dspy-flights`
