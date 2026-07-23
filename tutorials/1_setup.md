# Setup

In this tutorial you will install the tools, connect them to your Prime Intellect account, and finish with a real (tiny) evaluation running from your terminal. Everything here works on macOS and Linux and will take you a couple of minutes to do.

## 1. Create an account and get an API key

1. Sign up at [app.primeintellect.ai](https://app.primeintellect.ai).
2. Add a small amount of credit — evaluations bill per model token through Prime Inference, and the tutorials cost cents, not dollars.
3. Create an API key from your account settings and store it somewhere safe. You'll hand it to the CLI in step 3.

## 2. Install the tools

First `[uv](https://docs.astral.sh/uv/)`, a fast Python package manager that everything else builds on (of course, skip if you already use uv):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then the `prime` CLI:

```bash
uv tool install -U prime
```

Check it worked:

```bash
prime --help
```

If your shell says `prime: command not found`, run `uv tool update-shell` and open a fresh terminal — this puts uv's tool directory (`~/.local/bin`) on your `PATH`.

## 3. Authenticate

The friendliest way is the interactive login, which opens your browser:

```bash
prime login
```

Alternatively, you can paste the API key from step 1 directly (interactive mode hides your input, so the key stays out of your shell history) by running:

```bash
prime config set-api-key
```

After logging in via either method, verify that the CLI knows who you are:

```bash
prime whoami
prime config view
```

> **For scripts and CI:** the CLI (and everything built on it) also reads the `PRIME_API_KEY` environment variable, so `export PRIME_API_KEY="..."` works anywhere an interactive login doesn't. If you belong to multiple teams, `prime teams list` and `prime switch` control which one you bill to.

This single authentication is all you need to access anything on the platform. So for example, you do not need any external API keys for running evaluations, since they run through through **Prime Inference**, the unified gateway we mentioned in Tutorial 0.

## 4. Get the cookbook

The tutorials use this repository's environments and configs, so clone it and set up its Python workspace:

```bash
git clone https://github.com/PrimeIntellect-ai/lab-cookbook.git
cd lab-cookbook
uv sync
```

`uv sync` installs the verifiers framework plus the small example environments that live under `environments/` that we will use throughout the cookbook. This is a one-time step.

> **Current status:** the released `prime` CLI does not support verifiers v1 environments yet, so these tutorials run evals with `uv run eval ...` from the cookbook workspace (it comes with `uv sync`, via the verifiers framework). Once the CLI gains v1 support, the same configs will run through `prime eval run` as well.



## 5. Optional but recommended: install the Lab skills

If you work with a coding agent (Claude Code, Codex, ...), this teaches it how environments and evals work, so you can delegate the boring parts later:

```bash
prime lab setup
```

An interactive installer walks you through it. You can skip this if you don't use a coding agent.

## 6. Smoke test

Time to prove the whole chain works — CLI, authentication, inference, and the workspace — with a two-task evaluation of a toy environment that asks a model to reverse strings of text:

```bash
uv run eval reverse_text_v1 -n 2 -r 1
```

Reading that command: evaluate the local environment named `reverse_text_v1`, on `-n 2` tasks, with `-r 1` attempt per task. You should see the run start up, two rollouts complete with a score each, and a small summary at the end. Don't worry yet about where those numbers come from, what got saved to disk, and how to read the transcripts -- that will be exactly the subject of the next tutorials.

If instead you see:

- **an authentication error** — re-run `prime login` (or check `prime config view` shows your key).
- **a "taskset not found" error** — make sure you are inside the `lab-cookbook` directory and `uv sync` succeeded.
- **a payment / quota error** — add credit to your account at [app.primeintellect.ai](https://app.primeintellect.ai).



## Recap

You now have an authenticated `prime` CLI, the cookbook workspace, and proof that evals run end-to-end from your machine. Now you are ready to use the full functionality of the platform.

Next → [2 — Your First Eval](2_first_eval.md): measure a real model on grade-school math, and learn to read what comes back.