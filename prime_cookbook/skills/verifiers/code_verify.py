"""Code execution verifier.

Runs model-generated code in a subprocess with a timeout and
checks the output against expected test results.

For sandboxed execution (recommended for production), use
vf.PythonEnv or vf.SandboxEnv instead of this module.

WARNING: This verifier executes untrusted code in a subprocess.
Only use in controlled environments. For production, use the
Prime Sandboxes-backed vf.SandboxEnv.
"""
import asyncio
import subprocess
from typing import Optional


async def code_reward(
    completion: str,
    info: dict,
    **kwargs,
) -> float:
    """Execute code and check output against expected test cases.

    Extracts Python code from the completion (```python ... ``` blocks
    or raw code) and runs it with each test case from `info`.

    Returns the fraction of test cases that pass (0.0 to 1.0).

    info dict must contain:
        test_inputs: list[str]   — stdin input for each test case
        test_outputs: list[str]  — expected stdout output for each test case

    Both lists must have the same length.

    WARNING: Executes untrusted code. Use vf.PythonEnv or vf.SandboxEnv
    for sandboxed execution in production.

    Example:
        info = {
            "test_inputs": ["3 5", "10 20"],
            "test_outputs": ["8", "30"],
        }
        rubric = vf.Rubric(funcs=[code_reward])
        # Rewards 0.5 if 1 of 2 test cases pass, 1.0 if both pass.
    """
    test_inputs: list[str] = info.get("test_inputs", [])
    test_outputs: list[str] = info.get("test_outputs", [])

    if not test_inputs or not test_outputs:
        return 0.0

    n_tests = min(len(test_inputs), len(test_outputs))
    if n_tests == 0:
        return 0.0

    code = _extract_code(completion)
    if not code:
        return 0.0

    loop = asyncio.get_event_loop()
    passed = 0

    for i in range(n_tests):
        stdin_input = test_inputs[i]
        expected = test_outputs[i]
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda inp=stdin_input: subprocess.run(
                        ["python3", "-c", code],
                        input=inp,
                        capture_output=True,
                        text=True,
                        timeout=5,
                    ),
                ),
                timeout=10.0,
            )
            if result.stdout.strip() == expected.strip():
                passed += 1
        except (asyncio.TimeoutError, subprocess.TimeoutExpired):
            # Timed out — count as failure
            continue
        except Exception:
            # Any other error (import error, syntax error, etc.) — count as failure
            continue

    return passed / n_tests


def _extract_code(text: str) -> str:
    """Extract Python code from markdown code blocks or raw text.

    Extraction order:
    1. ```python ... ``` block (preferred)
    2. ``` ... ``` block (generic)
    3. Assume entire text is raw Python code

    Returns empty string if text is empty.
    """
    if not text:
        return ""

    # Try ```python ... ``` block
    marker_python = "```python"
    start = text.find(marker_python)
    if start != -1:
        newline = text.find("\n", start)
        if newline != -1:
            code_start = newline + 1
            code_end = text.find("```", code_start)
            if code_end != -1:
                return text[code_start:code_end].strip()

    # Try ``` ... ``` block (language-agnostic)
    marker_generic = "```"
    start = text.find(marker_generic)
    if start != -1:
        newline = text.find("\n", start)
        if newline != -1:
            code_start = newline + 1
            code_end = text.find(marker_generic, code_start)
            if code_end != -1:
                return text[code_start:code_end].strip()

    # Assume raw code
    return text.strip()
