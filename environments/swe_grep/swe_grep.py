import asyncio
import logging
from typing import Any
import pandas as pd
from datasets import Dataset, load_dataset
from prime_sandboxes import AsyncSandboxClient
import tenacity as tc
import verifiers as vf
from src.sandbox_metrics import SandboxMetrics, execute_command, retry_with_metrics

#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SweGrepEnv")
metrics = SandboxMetrics()


class SweGrepEnv(vf.SandboxEnv):
    def __init__(
        self, 
        max_turns, 
        max_setup_retries, 
        system_prompt, 
        labels: list[str] = ["swe-grep"],
        max_retries: int = 10,
        #logger: Any = None,  # Custom logger (e.g. structlog)
        **kwargs
    ):
        super().__init__(max_turns=max_turns, system_prompt=system_prompt, **kwargs)
        self.client = AsyncSandboxClient()
        self.max_setup_retries = max_setup_retries
        self.labels = labels
        self.max_retries = max_retries

        self.remove_tool(self.bash)
        self.add_tool(self.grep_tool, args_to_skip=["sandbox_id"])
        self.add_tool(self.list_files, args_to_skip=["sandbox_id"])
        self.add_tool(self.read_file, args_to_skip=["sandbox_id"])

    async def init_state(self, input, client, model, sampling_args=None):
        state = await super().init_state(input, client, model, sampling_args)
        # Copy file_path(s) from input to state for the reward function
        if isinstance(input, dict):
            if "file_path" in input:
                state["file_path"] = input["file_path"]
            if "file_path_2" in input:
                state["file_path_2"] = input["file_path_2"]
        return state


    async def setup_state(self, state, **kwargs):
        state = await super().setup_state(state, **kwargs)
        sandbox_id = state.get("sandbox_id", "unknown")
        last_error = ""
        for attempt in range(self.max_setup_retries):
            if attempt > 0:
                metrics.setup_retries += 1
                try:
                    old_sandbox_id = state.get("sandbox_id")
                    if old_sandbox_id:
                        try:
                            await self.client.delete(old_sandbox_id)
                        except:
                            pass
                    new_sandbox = await self.client.create(self.sandbox_request)
                    state["sandbox_id"] = new_sandbox.id
                    sandbox_id = new_sandbox.id
                except Exception as e:
                    metrics.creation_failed += 1
                    metrics.track_error(e, "sandbox_create")
                    continue
            
            try:
                await self.client.wait_for_creation(sandbox_id)
                metrics.creation_success += 1
            except Exception as e:
                metrics.creation_failed += 1
                metrics.track_error(e, "wait_for_creation")
                continue

            success, output = await execute_command(
                sandbox_client=self.client,
                sandbox_id=sandbox_id, 
                command="apt-get update && apt-get install -y git ripgrep", 
                metrics=metrics,
                operation="apt_install",
                max_retries=2
            )
            if not success:
                last_error = output
                continue

            success, output = await execute_command(
                sandbox_client=self.client,
                sandbox_id=sandbox_id, 
                command="git clone --depth 1 https://github.com/microsoft/vscode.git", 
                metrics=metrics,
                operation="git_clone", 
                max_retries=2

            )
            if not success:
                metrics.clone_failed += 1
                last_error = output
                continue

            success, output = await execute_command(
                sandbox_client=self.client,
                sandbox_id=sandbox_id, 
                command="ls vscode", 
                metrics=metrics,
                operation="verify_clone",
                max_retries=2,
            )
            if not success or not output.strip():
                last_error = "clone verification failed"
                continue

            metrics.setup_success += 1
            metrics.maybe_log()
            return state
        
        metrics.setup_failed += 1
        metrics.maybe_log()
        raise RuntimeError(f"Sandbox setup failed after {self.max_setup_retries} attempts: {last_error}")


    def update_tool_args(self, tool_name: str, tool_args: dict[str, Any], messages, state, **kwargs):
        updated_args = dict(tool_args)
        if tool_name in ["grep_tool", "list_files", "read_file"]:
            updated_args["sandbox_id"] = state["sandbox_id"]
        return updated_args

    async def grep_tool(
        self, 
        pattern: str, 
        sandbox_id: str, 
        path: str = "vscode", 
        file_pattern: str = "", 
        context_lines: int = 2, 
        case_insensitive: bool = False
    ) -> str:
        """Search for a pattern in files using ripgrep.
        
        Args:
            pattern: Text or regex to search for inside files
            path: Directory to search in
            file_pattern: Only search files matching this glob (e.g., *.ts, *.py)
            context_lines: Lines of context around each match
            case_insensitive: Ignore case when matching
        """
        import shlex
        
        max_lines = 50
        flags = ["-n", "--max-filesize", "100K"]
        if context_lines > 0:
            flags.extend(["-C", str(min(context_lines, 5))])
        if case_insensitive:
            flags.append("-i")
        if file_pattern:
            if file_pattern.startswith(".") and not file_pattern.startswith("*"):
                file_pattern = "*" + file_pattern
            flags.extend(["-g", shlex.quote(file_pattern)])
        
        cmd = f"rg {' '.join(flags)} {shlex.quote(pattern)} {shlex.quote(path)} 2>&1 | head -{max_lines + 1}"

        success, output = await execute_command(
            self.client,
            sandbox_id,
            cmd,
            metrics,
            "grep",
            max_retries=2
        )
        if not success:
            return f"Error: {output[:100]}"
        if not output.strip():
            return "No mathches found"
        lines = output.split('\n')
        lines = [line[:300] + '...' if len(line) > 300 else line for line in lines]
        if len(lines) > max_lines:
            output = '\n'.join(lines[:max_lines])
            return f"{output}\n\n[TRUNCATED - results exceed {max_lines} lines. Narrow your search with a more specific pattern or file_pattern]"
        return output


    async def list_files(self, path: str, sandbox_id: str) -> str:
        """List files and directories at a path.
        
        Args:
            path: Directory path to list contents of
        """
        import shlex
        
        cmd = f"ls -la {shlex.quote(path)}"
        success, output = await execute_command(
            self.client,
            sandbox_id,
            cmd,
            metrics,
            "list_files",
            max_retries=2
        )
        if not success:
            return f"Error: {output[:100]}"
        return output.strip() or "Empty directory"

    async def read_file(self, file_path: str, sandbox_id: str, start_line: int = 1, num_lines: int = 100) -> str:
        """Read lines from a file.
        
        Args:
            file_path: Path to the file
            start_line: Line number to start from (1-indexed)
            num_lines: Number of lines to read (max 50)
        """
        import shlex
        
        num_lines = min(num_lines, 50)
        end_line = start_line + num_lines - 1
        # Get one extra line to detect if there's more
        cmd = f"sed -n '{start_line},{end_line + 1}p' {shlex.quote(file_path)}"
        success, output = await execute_command(
            self.client,
            sandbox_id,
            cmd,
            metrics,
            "read_file",
            max_retries=2
        )
        if not success:
            return f"Error: {output[:100]}"
        if not output.strip():
            return f"No content at lines {start_line}-{end_line}"
        lines = output.split('\n')
        has_more = len(lines) > num_lines
        if has_more:
            output = '\n'.join(lines[:num_lines])
            return f"Lines {start_line}-{end_line} of {file_path}:\n{output}\n\n[MORE CONTENT BELOW - use start_line={end_line + 1} to continue]"
        return f"Lines {start_line}-{end_line} of {file_path}:\n{output}"


def convert_dataset(train_ratio=0.9, dataset_name="cdreetz/swe-grep-v2"):
    dataset = load_dataset(dataset_name, split="train")
    dataset = dataset.filter(lambda x: x["check"] == "Yes")
    dataset = dataset.rename_columns({"user_query": "question", "ground_truth": "answer"})
    # Remove check column (file_chunk doesn't exist in new dataset)
    cols_to_remove = [c for c in ["file_chunk", "check"] if c in dataset.column_names]
    if cols_to_remove:
        dataset = dataset.remove_columns(cols_to_remove)
    # file_path and file_path_2 are kept for reward function
    split = dataset.train_test_split(test_size=1 - train_ratio, seed=42)
    return split["train"], split["test"]



JUDGE_PROMPT = """Given a ground truth answer and a response, determine if the answer is correct.

Question:
{question}

Ground truth answer:
{answer}

Response:
{response}

Respond either 'yes' or 'no' only.
"""


REQUIRE_TOOL_CALLS_FOR_REWARD = False  # set False to reward even without tool calls


def _made_tool_calls(state) -> bool:
    for step in state.get("trajectory", []):
        response = step.get("response")
        if response and hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "message") and choice.message and choice.message.tool_calls:
                return True
    return False


async def correct_answer_reward_func(judge, prompt, completion, answer, state, **kwargs):
    if REQUIRE_TOOL_CALLS_FOR_REWARD and not _made_tool_calls(state):
        state["_is_correct"] = False
        return 0.0

    judge_response = await judge(prompt, completion, answer, state)
    is_correct = "yes" in judge_response.lower()
    state["_is_correct"] = is_correct
    return 1.0 if is_correct else 0.0


async def correct_file_path_reward_func(judge, completion, state, **kwargs):
    """Reward for finding the correct file path using the judge."""
    file_path = state.get("file_path")
    if not file_path:
        return 0.0

    if REQUIRE_TOOL_CALLS_FOR_REWARD and not _made_tool_calls(state):
        return 0.0

    judge_response = await judge(
        "Does the response reference this file path?",
        completion,
        file_path,
        state
    )
    is_correct_path = "yes" in judge_response.lower()
    state["_correct_file_path"] = is_correct_path
    return 1.0 if is_correct_path else 0.0


async def correct_file_paths_reward_func(judge, completion, state, **kwargs):
    """Reward for finding correct file path(s). Handles both single and multi-file examples."""
    file_path_1 = state.get("file_path")
    file_path_2 = state.get("file_path_2")

    if not file_path_1:
        return 0.0

    if REQUIRE_TOOL_CALLS_FOR_REWARD and not _made_tool_calls(state):
        return 0.0

    # Check first file path
    resp1 = await judge(
        "Does the response reference this file path?",
        completion,
        file_path_1,
        state
    )
    found_1 = "yes" in resp1.lower()
    state["_found_file_1"] = found_1

    # Single-file example: only check file_path_1
    if not file_path_2:
        state["_found_file_2"] = True  # Mark as "found" for efficiency_bonus compatibility
        state["_is_single_file"] = True
        return 1.0 if found_1 else 0.0

    # Multi-file example: check both paths
    resp2 = await judge(
        "Does the response reference this file path?",
        completion,
        file_path_2,
        state
    )
    found_2 = "yes" in resp2.lower()
    state["_found_file_2"] = found_2
    state["_is_single_file"] = False

    if found_1 and found_2:
        return 1.0
    elif found_1 or found_2:
        return 0.3
    return 0.0


async def parallel_tool_calls_reward_func(completion, state, **kwargs):
    """Reward for making parallel tool calls per turn."""
    trajectory = state.get("trajectory", [])
    if not trajectory:
        return 0.0
    
    tool_calls_per_turn = []
    for step in trajectory:
        response = step.get("response")
        if response and hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "message") and choice.message and choice.message.tool_calls:
                tool_calls_per_turn.append(len(choice.message.tool_calls))
    
    if not tool_calls_per_turn:
        return 0.0
    
    avg_calls = sum(tool_calls_per_turn) / len(tool_calls_per_turn)
    return min(avg_calls / 8.0, 1.0)


async def efficiency_bonus_for_correct(states: list, **kwargs) -> list[float]:
    """Among fully correct rollouts, reward for fewer turns. No tool calls = 0."""
    rewards = [0.0] * len(states)

    def is_correct(s) -> bool:
        """Check if rollout found all required file paths."""
        if not s.get("_found_file_1", False):
            return False
        # Single-file: _found_file_2 is set to True by correct_file_paths_reward_func
        # Multi-file: _found_file_2 is set based on judge response
        if not s.get("_found_file_2", False):
            return False
        if REQUIRE_TOOL_CALLS_FOR_REWARD and not _made_tool_calls(s):
            return False
        return True

    correct_indices = [i for i, s in enumerate(states) if is_correct(s)]

    if not correct_indices:
        return rewards

    turn_counts = [len(s.get("trajectory", [])) for s in states]
    min_turns = min(turn_counts[i] for i in correct_indices)

    for i in correct_indices:
        rewards[i] = min_turns / turn_counts[i]

    return rewards


SYSTEM_PROMPT = """You are a helpful assistant that can answer questions and help with tasks.
Use the provided tools to search through the codebase to best answer user queries.
You will only have 2 turns to complete the task, but can make up to 8 tool calls per turn.
This means you MUST make multiple parallel tool calls to find information efficiently.

IMPORTANT: Questions may require finding information from MULTIPLE files. When providing your final answer, you MUST:
1. Include ALL file paths where you found relevant information
2. Provide the complete answer to the question

Format your response like this:
Files:
- <path/to/file1>
- <path/to/file2>
Answer: <your answer here>

With only 2 turns, you must parallelize your searches to find all relevant files.
"""

#SYSTEM_PROMPT = "You are a helpful assistant."

def load_environment(
    max_turns: int = 2,
    max_setup_retries: int = 3,
    system_prompt: str = SYSTEM_PROMPT,
    **kwargs
) -> vf.Environment:
    train_dataset, test_dataset = convert_dataset()
    rubric = vf.JudgeRubric(judge_prompt=JUDGE_PROMPT)
    rubric.add_reward_func(correct_answer_reward_func, weight=0.4)
    rubric.add_reward_func(correct_file_paths_reward_func, weight=0.4)
    rubric.add_reward_func(parallel_tool_calls_reward_func, weight=0.2)
    rubric.add_reward_func(efficiency_bonus_for_correct, weight=0.0)

    return SweGrepEnv(
        dataset=train_dataset,
        eval_dataset=test_dataset,
        rubric=rubric,
        max_turns=max_turns,
        max_setup_retries=max_setup_retries,
        system_prompt=system_prompt
    )

