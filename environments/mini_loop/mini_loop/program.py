# /// script
# requires-python = ">=3.10"
# dependencies = ["openai>=1.0"]
# ///
"""A minimal bash-loop coding agent.

One chat completion per step: the model replies with a single ```bash code block,
the program runs it in the task container and feeds the output back. The loop ends
when the model replies DONE (no code block) or the step budget runs out.
"""

import argparse
import os
import re
import subprocess

from openai import OpenAI

INSTRUCTIONS = """\

You are working inside a Linux container; the task files live in the current directory.
Respond with exactly one bash code block per turn, like:

```bash
ls
```

I will run it and show you exit code and output. Keep commands short and observable.
When the task is complete, reply with the single word DONE and no code block."""

BASH_BLOCK = re.compile(r"```bash\n(.*?)```", re.DOTALL)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--command-timeout", type=float, default=120.0)
    args = parser.parse_args()

    client = OpenAI(
        base_url=os.environ["OPENAI_BASE_URL"], api_key=os.environ["OPENAI_API_KEY"]
    )
    messages = [{"role": "user", "content": args.task + INSTRUCTIONS}]

    for _ in range(args.max_steps):
        reply = (
            client.chat.completions.create(model=args.model, messages=messages)
            .choices[0]
            .message.content
            or ""
        )
        messages.append({"role": "assistant", "content": reply})
        match = BASH_BLOCK.search(reply)
        if match is None:
            break  # DONE, or the model stopped producing commands
        try:
            result = subprocess.run(
                ["bash", "-lc", match.group(1)],
                capture_output=True,
                text=True,
                timeout=args.command_timeout,
            )
            feedback = f"exit={result.returncode}\n{(result.stdout + result.stderr)[-4000:]}"
        except subprocess.TimeoutExpired:
            feedback = f"command timed out after {args.command_timeout}s"
        messages.append({"role": "user", "content": feedback})


if __name__ == "__main__":
    main()
