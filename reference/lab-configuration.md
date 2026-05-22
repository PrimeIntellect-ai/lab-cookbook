# Lab Configuration

A reference tour of the platform machinery that surrounds environments, evals, and training: accounts, teams, secrets, Hub workflows, hosted runs, and inference deployments.

The guides keep the surface small so the learning loop stays fast. This page is the opposite: short sections covering each piece of Lab plumbing you eventually need, with pointers into the public docs for full reference.

## Accounts, Teams, and Billing

Most Lab actions run against either your personal account or a team account. The CLI keeps one active context at a time.

```bash
prime config view                    # current api key, team, base url
prime teams list                     # team ids you belong to
prime config set-team-id <team-id>   # switch the CLI to a team
prime config set-team-id ""          # clear, back to personal
```

The active team is used by `prime env push`, `prime eval run`, `prime train`, deployments, and any credit charges. Switch before launching a hosted run so the work and the bill land in the right place.

For team account billing, credit pools, and member roles, see [Inference: Team Accounts](https://docs.primeintellect.ai/inference/team-accounts) and the platform Team Profile page.

## API Keys and CLI Auth

`prime login` opens a browser flow and writes credentials the CLI uses automatically. Most workflows just need this — `prime eval run`, `prime env push`, and `prime train` all reuse the logged-in session for Prime Inference.

`PRIME_API_KEY` is for code that talks to the Prime Inference API directly (OpenAI client, scripts, third-party tools). Create the key in the platform under **API Keys**, scope it to **Inference**, and export:

```bash
export PRIME_API_KEY=sk-...
```

You do not need to export `PRIME_API_KEY` just to run `prime eval run` against a Prime model — that path uses the login session. Reference: [CLI config](https://docs.primeintellect.ai/cli-reference/config-cli).

## Workspace Endpoint Aliases

[configs/endpoints.toml](../configs/endpoints.toml) in a Lab workspace maps short aliases (`gpt-5-mini`, `qwen3-30b-i`, `sonnet`) to provider, model id, base URL, and the env var that holds the key. Aliases are how evals and GEPA refer to models without repeating provider URLs.

```toml
[[endpoint]]
endpoint_id = "qwen3-30b-i"
model = "qwen/qwen3-30b-a3b-instruct-2507"
url = "https://api.pinference.ai/api/v1"
key = "PRIME_API_KEY"
type = "openai_chat_completions"
```

Add new aliases here when bringing in a third-party provider, and export the corresponding `*_API_KEY` in your shell before running anything that resolves to it. The CLI reads [endpoints.toml](../configs/endpoints.toml) directly — no `prime lab sync` step is needed.

## Environment Variables and Secrets

Three knobs control runtime config for an environment on the Hub. Pick by sensitivity and reuse:


| Knob                 | Sensitive? | Scope     | Use for                            |
| -------------------- | ---------- | --------- | ---------------------------------- |
| Environment variable | No         | one env   | model id, difficulty, feature flag |
| Direct secret        | Yes        | one env   | env-specific API key               |
| Linked global secret | Yes        | many envs | shared key reused across envs      |


Names are `[A-Z][A-Z0-9_]*`; the platform blocks collisions across all three types in a single environment.

```bash
prime env var create owner/env --name TASK_DIFFICULTY --value hard
prime secret create OPENAI_API_KEY                   # global, reusable
prime env secret link owner/env --name OPENAI_API_KEY
prime env secret create owner/env --name HF_TOKEN ...  # direct, env-only
```

Precedence at runtime: env variable < linked secret < direct secret. All three are auto-injected into Hub Actions, Hosted Evaluations, and Hosted Training — you do not need to pass them as `--env-args`. Validate required keys in `load_environment()` with `vf.ensure_keys(...)` so the env fails loudly when a secret is missing.

References: [Secrets](https://docs.primeintellect.ai/tutorials-environments/secrets), [Environment Variables](https://docs.primeintellect.ai/tutorials-environments/environment-variables).

## Environments Hub Workflows

`prime env push` builds the wheel, uploads metadata, and triggers a build/install/test Action in a fresh container. The Action result is visible in the Hub under the env's **Actions** tab.

```bash
prime env push --path environments/my-env
prime env push --path environments/my-env --team my-team
prime env push --path environments/my-env --visibility PRIVATE
prime env push --path environments/my-env --auto-bump            # bump patch in pyproject
```

Versions are immutable once pushed. Install or pull a specific tag:

```bash
prime env install owner/my-env@0.1.3
prime env pull owner/my-env                # source for local modification
prime env info owner/my-env                # metadata + direct wheel url
```

Git-based dependencies must use PEP 440 form in `pyproject.toml` (`dependencies = ["mypkg @ git+https://..."]`); `[tool.uv.sources]` is stripped from wheel metadata and breaks remote installs.

Collaborators are added on the env's **Collaborators** tab (read-only today; writer access is on the roadmap). Actions surface the same build/install/test logs the Hub uses to validate every push — check these first when a hosted run can't load your env.

References: [Create](https://docs.primeintellect.ai/tutorials-environments/create), [Install](https://docs.primeintellect.ai/tutorials-environments/install), [Actions](https://docs.primeintellect.ai/tutorials-environments/environment-actions), [Manage Collaborators](https://docs.primeintellect.ai/tutorials-environments/manage-collaborators).

## Hosted Evaluations

Hub-driven evals, also called hosted evals, run on Prime infrastructure instead of your machine. Open the env in the Hub, click **Run Hosted Evaluation**, pick a model, set `num_examples` and `rollouts_per_example`, and optionally add `env_args`. Credits cover model tokens; secrets and variables auto-inject. Results show up in the env's **Evaluations** tab alongside CLI-launched runs.

Reach for hosted evals when the model is large enough that local concurrency is the bottleneck, or when you want a reproducible run pinned to a specific Hub version of the env. For everything else, `prime eval run` from your workspace is still faster to iterate on.

Reference: [Hosted Evaluations](https://docs.primeintellect.ai/tutorials-environments/hosted-evaluations).

## Training Config Reference

[Training with RL](../guides/03-training-with-rl/README.md) covers the minimal TOML. The pieces below are the ones you reach for once a basic run learns:

```toml
[buffer]
env_ratios = [0.6, 0.4]              # weight multi-env mixes
online_difficulty_filtering = true
easy_threshold = 0.8
hard_threshold = 0.2
easy_fraction = 0.1                  # cap easy/hard exposure
hard_fraction = 0.1

[checkpoints]
interval = 100
keep_cloud = 5                       # retention; deployed adapters are exempt

[adapters]
interval = 100
keep_last = 3                        # LoRA uploads for inference deploy

[eval]
interval = 100
num_examples = -1                    # full eval split
eval_base_model = true               # baseline at step 0

[val]
interval = 5
num_examples = 64                    # quick val cadence

[wandb]
project = "my-project"
entity = "my-team"
```

Pass `checkpoint_id = "cp_..."` at the top level to resume. Secrets for training read from the same linked/direct sources as evals — only fall back to `env_file = ["secrets.env"]` when working off-platform.

Reference: [Hosted Training: Advanced Configs](https://docs.primeintellect.ai/hosted-training/advanced-configs), [Models and Pricing](https://docs.primeintellect.ai/hosted-training/models-and-pricing), [Troubleshooting](https://docs.primeintellect.ai/hosted-training/troubleshooting).

## Inference Deployments

Prime Inference exposes an OpenAI-compatible API at `https://api.pinference.ai/api/v1`. Base models are always-on; LoRA adapters from a training run must be deployed before they can be queried through inference deployments.

```bash
prime deployments list
prime deployments create <adapter-id>     # waits for DEPLOYED status
prime deployments delete <adapter-id>     # unload
```

Once deployed, query the adapter by combining base model id and adapter id with a colon:

```python
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["PRIME_API_KEY"],
    base_url="https://api.pinference.ai/api/v1",
)
resp = client.chat.completions.create(
    model="Qwen/Qwen3-4B-Instruct-2507:gw3zytpj9...",
    messages=[{"role": "user", "content": "..."}],
)
```

For team workloads, scope a request to a team without switching CLI context by sending `X-Prime-Team-ID` as a default header on the OpenAI client. Deployed adapters are excluded from the `keep_cloud` cleanup in `[checkpoints]`, so the adapter stays queryable even after newer checkpoints overwrite the source run.

References: [Inference Overview](https://docs.primeintellect.ai/inference/overview), [Usage](https://docs.primeintellect.ai/inference/usage), [Adapter Deployments](https://docs.primeintellect.ai/inference/adapter-deployments), [Team Accounts](https://docs.primeintellect.ai/inference/team-accounts).

## Workspace Health: `sync` and `doctor`

Two workspace-level commands cover the "is my setup still right?" loop:

```bash
prime lab sync       # pull upstream Lab skills + agent guidance into this workspace
prime lab doctor     # validate the workspace; print active account, team, CLI version
```

`sync` refreshes `.prime/skills/` and the local agent docs (`AGENTS.md`, `CLAUDE.md`, `environments/AGENTS.md`) from the version of Lab the CLI ships. Reach for it after `prime upgrade`, or when the bundled skills or guidance feel out of date — not after editing your own files in [configs/](../configs/). Use `--skip-docs` to refresh skills without touching agent docs, or `--skip-agent` to refresh shared Lab assets without configuring coding-agent skill roots.

Run `doctor` first when a hosted run misbehaves — it surfaces the most common causes (wrong team context, missing API key, outdated CLI) before you start digging into env code.

## Where to Look Next


| Topic                              | Public docs                                                                                                                                                                    |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| CLI commands and flags             | [/cli-reference/environments](https://docs.primeintellect.ai/cli-reference/environments), [/cli-reference/config-cli](https://docs.primeintellect.ai/cli-reference/config-cli) |
| Environments Hub end-to-end        | [/tutorials-environments/getting-started](https://docs.primeintellect.ai/tutorials-environments/getting-started)                                                               |
| Verifiers reference                | [/verifiers/reference](https://docs.primeintellect.ai/verifiers/reference)                                                                                                     |
| Training models, pricing, recovery | [/hosted-training](https://docs.primeintellect.ai/hosted-training/getting-started)                                                                                             |
| Inference API                      | [/inference](https://docs.primeintellect.ai/inference/overview)                                                                                                                |
| API surface                        | [/API/api-references](https://docs.primeintellect.ai/API/api-references)                                                                                                       |


