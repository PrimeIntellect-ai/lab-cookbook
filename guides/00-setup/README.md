# Setup

Create a local Lab workspace for environments, evals, training, and more.

## Install the CLI

Install `uv` if you do not already have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

The installer prints shell-specific setup instructions if `uv` is new on your machine. Restart your shell or source the profile file it names before continuing.

Install the Prime CLI:

```bash
uv tool install prime
```

Expected output:

Prime CLI install terminal output

Authenticate with Prime:

```bash
prime login
```

You can skip this step for now, but most Lab workflows require an authenticated Prime account.

## Create a Workspace

Lab development takes place inside a workspace folder, which can hold multiple environments and configs for your project.

First, create a folder to use as your workspace:

```bash
mkdir -p ~/lab
cd ~/lab
```

Then set up the workspace and choose which coding agents to configure:

```bash
# options: amp, claude, codex, cursor, droid, hermes, letta, opencode, pi
prime lab setup --agents codex,claude
```

The command initializes the workspace, installs the Verifiers dependency, prepares [configs/](../../configs/), refreshes Lab guidance, and prints a short "get started" panel with next commands.

This adds project structure, example configs, and agent guidance covering environment workflows and best practices.

You can also run `prime lab setup` without flags to configure coding agents interactively.
image.png
To refresh Lab-provided configs and guidance later, run:

```bash
prime lab sync
```

Sync refreshes the managed Lab skills, templates, and agent guidance in the current workspace.

## Workspace Layout

After setup, your workspace contains a few key pieces:

- `configs/` holds example configs for evaluation, training, and related workflows.
- `environments/` is where your local environment packages live.
- `AGENTS.md` (and/or `CLAUDE.md`) tells coding agents how to navigate Lab.
- `.prime/` stores Lab metadata and other local assets.

Most day-to-day work happens in `configs/` and `environments/`; the other files keep Lab in sync with the workspace and help your coding agents follow best practices.

## Check the CLI

Check that your workspace is set up correctly:

```bash
prime lab doctor
```

See the models available for Hosted Training:

```bash
prime train models
```

You should see a Hosted Training model table with model names, availability, inference pricing, and training pricing.

Then continue to [Environments and Evals](../01-environments-and-evals/README.md) to explore evaluating models using environments.