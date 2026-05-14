# Setup

Create a local Lab workspace for environments, evals, training, and more.

## Install the CLI

Install `uv` if you do not already have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```text
TODO: expected output
```

Install the Prime CLI:

```bash
uv tool install prime
```

```text
TODO: expected output
```

Authenticate with Prime:

```bash
prime login
```

```text
TODO: expected output
```

You can also continue without logging in immediately, but many Lab workflows require an authenticated Prime account.

## Create a Workspace

Lab development takes place inside a workspace folder, which can contain multiple environments and configs for your project.

First, select a folder as your workspace:

```bash
mkdir -p ~/lab
cd ~/lab
```

```text
TODO: expected output
```

Then, set up your workspace for development with agents:

```bash
# options: amp, claude, codex, cursor, droid, hermes, letta, opencode, pi
prime lab setup --agents codex,claude
```

```text
TODO: expected output
```

This prepares your workspace for Lab development by adding project structure, example configs, and agent guidance for environment workflows and best practices.

You can also run `prime lab setup` without flags to configure coding agents interactively.

To refresh Lab-provided configs and guidance later, run:

```bash
prime lab sync
```

```text
TODO: expected output
```

## Workspace Layout

After setup, your workspace is organized around a few recurring pieces:

- `configs/` contains example configs for evaluation, training, and related workflows.
- `environments/` is where your local environment packages live.
- `AGENTS.md` (and/or `CLAUDE.md`) gives coding agents guidance on how to best navigate Lab.
- `.prime/` stores Lab metadata and other local assets.

Most day-to-day work happens in `configs/` and `environments/`; the other files keep Lab and your coding agents aligned around the workspace.

## Check the CLI

Check that your workspace is set up correctly:

```bash
prime lab doctor
```

```text
TODO: expected output
```

See the models available for Hosted Training:

```bash
prime train models
```

```text
TODO: expected output
```

Then continue to [Environments and Evals](../01-environments-and-evals/README.md) to explore evaluating models using environments.
