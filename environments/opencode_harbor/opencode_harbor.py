import json
import shlex
from pathlib import PurePosixPath

import verifiers.v1 as vf
from verifiers.v1.tasksets.harbor import HarborConfig, HarborTask, HarborTaskset

DEFAULT_RELEASE_REPO = "PrimeIntellect-ai/opencode"
DEFAULT_RELEASE_VERSION = "1.1.63-rl2"

DEFAULT_SYSTEM_PROMPT = """You are an autonomous coding agent running inside the task container.

Edit files and run tests as needed. When you believe the task is complete, stop after a concise final message.
"""

DEFAULT_DISABLED_TOOLS = [
    "apply_patch",
    "write",
    "multiedit",
    "glob",
    "todowrite",
    "todoread",
    "websearch",
    "task",
    "batch",
    "list",
    "read",
    "question",
    "webfetch",
    "grep",
    "plan_exit",
    "plan_enter",
    "lsp",
    "codesearch",
    "skill",
]

JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


class OpenCodeHarborConfig(HarborConfig):
    dataset: str = "hello-world"
    ignore_dockerfile: bool = True


class OpenCodeHarborTaskset(
    HarborTaskset,
    vf.Taskset[HarborTask, OpenCodeHarborConfig],
):  # ty: ignore[invalid-generic-class]
    pass


class OpenCodeHarnessConfig(vf.HarnessConfig):
    version: str = DEFAULT_RELEASE_VERSION
    release_repo: str = DEFAULT_RELEASE_REPO
    install_ripgrep: bool = True
    allow_git: bool = False
    disable_compaction: bool = True
    provider_timeout_ms: int = 3_600_000
    prompt_path: str = "/opencode/prompt.txt"
    system_prompt_path: str = "/opencode/system.txt"
    log_path: str = "/opencode/logs.txt"
    agent_workdir: str = "/app"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    disabled_tools: list[str] | None = DEFAULT_DISABLED_TOOLS


def build_install_script(config: OpenCodeHarnessConfig) -> str:
    rg_install = (
        "apt-get -o Acquire::Retries=3 install -y -qq ripgrep > /dev/null 2>&1 || true"
        if config.install_ripgrep
        else ""
    )
    return f"""\
set -e
apt-get -o Acquire::Retries=3 update -qq && apt-get -o Acquire::Retries=3 install -y -qq curl tar > /dev/null 2>&1
{rg_install}

case "$(uname -m)" in
  x86_64) OPENCODE_ARCH=x64 ;;
  aarch64|arm64) OPENCODE_ARCH=arm64 ;;
  *) echo "Unsupported architecture: $(uname -m)"; exit 1 ;;
esac

OPENCODE_ASSET="opencode-linux-$OPENCODE_ARCH.tar.gz"
OPENCODE_RELEASE_URL="https://github.com/{config.release_repo}/releases/download/v{config.version}/$OPENCODE_ASSET"

mkdir -p "$HOME/.opencode/bin"
if [ -x "$HOME/.opencode/bin/opencode" ]; then
  echo "OpenCode already installed, skipping download"
else
  curl -fsSL "$OPENCODE_RELEASE_URL" -o /tmp/opencode.tar.gz
  tar -xzf /tmp/opencode.tar.gz -C /tmp
  install -m 755 /tmp/opencode "$HOME/.opencode/bin/opencode"
  rm -f /tmp/opencode.tar.gz /tmp/opencode
fi
"""


def build_opencode_config(config: OpenCodeHarnessConfig) -> str:
    agent_config: dict[str, JsonValue] = {"title": {"disable": True}}
    build_config: dict[str, JsonValue] = {}
    if config.system_prompt:
        build_config["prompt"] = "{file:" + config.system_prompt_path + "}"
    if config.disabled_tools:
        build_config["tools"] = {tool: False for tool in config.disabled_tools}
    if build_config:
        agent_config["build"] = build_config
    payload: dict[str, JsonValue] = {
        "${SCHEMA_DOLLAR}schema": "https://opencode.ai/config.json",
        "provider": {
            "${OPENAI_MODEL%%/*}": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "${OPENAI_MODEL%%/*}",
                "options": {
                    "baseURL": "$OPENAI_BASE_URL",
                    "apiKey": "${OPENAI_API_KEY:-intercepted}",
                    "timeout": config.provider_timeout_ms,
                },
                "models": {
                    "${OPENAI_MODEL##*/}": {
                        "name": "${OPENAI_MODEL##*/}",
                        "modalities": {"input": ["text", "image"], "output": ["text"]},
                        "interleaved": {"field": "reasoning_content"},
                    }
                },
            }
        },
        "model": "$OPENAI_MODEL",
        "small_model": "$OPENAI_MODEL",
        "agent": agent_config,
    }
    if config.disable_compaction:
        payload["compaction"] = {"auto": False, "prune": False}
    return json.dumps(payload, indent=2)


class OpenCodeHarness(vf.Harness[OpenCodeHarnessConfig]):
    SUPPORTS_MCP = False
    APPENDS_SYSTEM_PROMPT = True

    async def setup(self, runtime: vf.Runtime) -> None:
        install = build_install_script(self.config)
        guarded = (
            "mkdir -p /tmp/vf-opencode && "
            f"flock /tmp/vf-opencode/install.lock sh -c {shlex.quote(install)}"
        )
        result = await runtime.run(["sh", "-c", guarded], self.config.env)
        if result.exit_code != 0:
            detail = (result.stderr or result.stdout).strip()[-2000:]
            raise RuntimeError(f"OpenCode install failed: {detail}")

    async def launch(
        self,
        ctx: vf.RolloutContext,
        trace: vf.Trace,
        runtime: vf.Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> vf.ProgramResult:
        if mcp_urls:
            raise ValueError("OpenCodeHarness does not expose taskset MCP servers")
        task_system_prompt, prompt = self.resolve_prompt(trace.task)
        if not isinstance(prompt, str):
            raise ValueError("OpenCodeHarness requires a string task prompt")
        system_prompt = "\n\n".join(
            part for part in (self.config.system_prompt, task_system_prompt) if part
        )
        await runtime.write(self.config.prompt_path, prompt.encode())
        if system_prompt:
            await runtime.write(self.config.system_prompt_path, system_prompt.encode())
        log_dir = str(PurePosixPath(self.config.log_path).parent)
        config_json = build_opencode_config(self.config)
        script = f"""\
set -eo pipefail
export PATH="$HOME/.opencode/bin:$PATH"
export OPENCODE_DISABLE_FILETIME_CHECK=true
export ALLOW_GIT={"1" if self.config.allow_git else "0"}
if [[ "$OPENAI_MODEL" != *"/"* ]]; then
    export OPENAI_MODEL="vllm/$OPENAI_MODEL"
fi
OPENCODE_CONFIG_DIR="${{XDG_CONFIG_HOME:-$HOME/.config}}/opencode"
mkdir -p "$OPENCODE_CONFIG_DIR" {shlex.quote(log_dir)} {shlex.quote(self.config.agent_workdir)}
SCHEMA_DOLLAR='$'
cat > "$OPENCODE_CONFIG_DIR/opencode.json" << EOFCONFIG
{config_json}
EOFCONFIG
cd {shlex.quote(self.config.agent_workdir)}
cat {shlex.quote(self.config.prompt_path)} | opencode run 2>&1 | tee {shlex.quote(self.config.log_path)}
"""
        env = {
            **self.config.env,
            "OPENAI_BASE_URL": endpoint,
            "OPENAI_API_KEY": secret,
            "OPENAI_MODEL": ctx.model,
        }
        return await runtime.run_program(["bash", "-lc", script], env)


__all__ = ["OpenCodeHarborTaskset", "OpenCodeHarness"]
