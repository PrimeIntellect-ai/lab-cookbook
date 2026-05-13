from __future__ import annotations

import asyncio
import os
import re
from collections.abc import Iterable, Mapping
from typing import Any

import httpx
import verifiers.v1 as vf
from datasets import load_dataset
from openai import AsyncOpenAI
from verifiers import ensure_keys

DATASET_NAME = "hotpot_qa"
DATASET_CONFIG = "distractor"
WIKI_API_URL = "https://en.wikipedia.org/w/api.php"

SYSTEM_PROMPT = """You are a research assistant that answers questions using Wikipedia.

You have access to tools that search and read Wikipedia pages. Most questions need
evidence from MULTIPLE pages, so you should make several parallel tool calls per turn
to gather facts efficiently.

Work in this order:
1. Plan the entities and relations the question is about.
2. Issue parallel `search_wikipedia` calls for each entity.
3. Read the most promising pages with `read_page`.
4. Synthesize the evidence and produce a short final answer.

Format your final response exactly like this:
Sources:
- <page title 1>
- <page title 2>
Answer: <concise answer>
"""

JUDGE_PROMPT = """Given a ground truth answer and a response, determine if the response is correct.

Question:
{question}

Ground truth answer:
{answer}

Response:
{response}

Respond either 'yes' or 'no' only.
"""


class WikiSearchTasksetConfig(vf.TasksetConfig):
    dataset_name: str = DATASET_NAME
    dataset_config: str = DATASET_CONFIG
    train_split: str = "train"
    eval_split: str = "validation"
    max_train_examples: int = 1000
    max_eval_examples: int = 200
    max_turns: int = 4
    system_prompt: str | None = SYSTEM_PROMPT
    judge_model: str = "gpt-4.1-mini"
    judge_base_url: str = "https://api.openai.com/v1"
    judge_api_key_var: str = "OPENAI_API_KEY"
    wiki_api_url: str = WIKI_API_URL
    wiki_user_agent: str = (
        "lab-cookbook-wiki-search (https://github.com/PrimeIntellect-ai/lab-cookbook)"
    )
    wiki_request_timeout_seconds: float = 30.0


_http_client_lock = asyncio.Lock()
_http_clients: dict[tuple[str, float], httpx.AsyncClient] = {}


async def _get_http_client(user_agent: str, timeout: float) -> httpx.AsyncClient:
    key = (user_agent, timeout)
    async with _http_client_lock:
        client = _http_clients.get(key)
        if client is None or client.is_closed:
            client = httpx.AsyncClient(
                timeout=httpx.Timeout(timeout, connect=10.0),
                headers={"User-Agent": user_agent},
            )
            _http_clients[key] = client
        return client


def completion_text(state: Mapping[str, object]) -> str:
    completion = state.get("completion") or []
    if not isinstance(completion, list):
        return str(completion)
    for message in reversed(completion):
        if isinstance(message, Mapping) and message.get("role") == "assistant":
            return str(message.get("content") or "")
    return ""


def tool_calls_per_turn(state: Mapping[str, object]) -> list[int]:
    completion = state.get("completion") or []
    if not isinstance(completion, list):
        return []
    counts: list[int] = []
    for message in completion:
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls") or []
        if isinstance(tool_calls, list) and tool_calls:
            counts.append(len(tool_calls))
    return counts


def _supporting_titles(row: Mapping[str, Any]) -> list[str]:
    supporting = row.get("supporting_facts")
    titles: Iterable[Any]
    if isinstance(supporting, Mapping):
        titles = supporting.get("title") or []
    elif isinstance(supporting, list):
        titles = [entry.get("title") for entry in supporting if isinstance(entry, Mapping)]
    else:
        titles = []
    seen: set[str] = set()
    output: list[str] = []
    for title in titles:
        title_str = str(title).strip() if title is not None else ""
        if title_str and title_str not in seen:
            seen.add(title_str)
            output.append(title_str)
    return output


def _rows(dataset: Iterable[Mapping[str, Any]], max_turns: int, limit: int | None):
    count = 0
    for row in dataset:
        if limit is not None and count >= limit:
            break
        question = str(row.get("question") or "").strip()
        answer = str(row.get("answer") or "").strip()
        if not question or not answer:
            continue
        titles = _supporting_titles(row)
        yield {
            "example_id": row.get("id") or row.get("example_id") or count,
            "question": question,
            "answer": answer,
            "supporting_titles": titles,
            "prompt": [{"role": "user", "content": question}],
            "max_turns": max_turns,
        }
        count += 1


def _make_tools(config: WikiSearchTasksetConfig):
    user_agent = config.wiki_user_agent
    timeout = config.wiki_request_timeout_seconds
    api_url = config.wiki_api_url

    async def search_wikipedia(query: str, limit: int = 5) -> str:
        """Search Wikipedia for pages matching a query.

        Args:
            query: Search terms (entity names, phrases, etc.).
            limit: Maximum number of results to return, capped at 10.

        Returns:
            A numbered list of "Title — snippet" entries, or a no-results message.
        """
        limit = max(1, min(int(limit), 10))
        client = await _get_http_client(user_agent, timeout)
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "format": "json",
            "formatversion": "2",
        }
        response = await client.get(api_url, params=params)
        response.raise_for_status()
        hits = response.json().get("query", {}).get("search") or []
        if not hits:
            return f"No Wikipedia results for {query!r}."
        lines: list[str] = []
        for index, hit in enumerate(hits, start=1):
            title = str(hit.get("title") or "").strip()
            snippet = re.sub(r"<[^>]+>", "", str(hit.get("snippet") or "")).strip()
            if len(snippet) > 240:
                snippet = snippet[:240] + "..."
            lines.append(f"{index}. {title} — {snippet}")
        return "\n".join(lines)

    async def read_page(title: str, max_chars: int = 4000) -> str:
        """Read the plain-text extract of a Wikipedia page.

        Args:
            title: Page title as it appears on Wikipedia.
            max_chars: Maximum number of characters to return, capped at 8000.

        Returns:
            The page extract, truncated to max_chars, or an error message.
        """
        max_chars = max(200, min(int(max_chars), 8000))
        client = await _get_http_client(user_agent, timeout)
        params = {
            "action": "query",
            "prop": "extracts",
            "titles": title,
            "explaintext": "1",
            "redirects": "1",
            "format": "json",
            "formatversion": "2",
        }
        response = await client.get(api_url, params=params)
        response.raise_for_status()
        pages = response.json().get("query", {}).get("pages") or []
        if not pages:
            return f"No page found for {title!r}."
        page = pages[0]
        if page.get("missing"):
            return f"No page found for {title!r}."
        canonical_title = str(page.get("title") or title)
        extract = str(page.get("extract") or "").strip()
        if not extract:
            return f"Page {canonical_title!r} has no plain-text extract."
        if len(extract) > max_chars:
            return (
                f"{canonical_title}\n\n{extract[:max_chars]}\n\n"
                "[truncated — call read_page again with a larger max_chars "
                "or refine your search]"
            )
        return f"{canonical_title}\n\n{extract}"

    return [search_wikipedia, read_page]


def _judge_reward_factory(
    judge_model: str,
    judge_base_url: str,
    judge_api_key_var: str,
):
    ensure_keys([judge_api_key_var])
    judge_client = AsyncOpenAI(
        api_key=os.environ[judge_api_key_var],
        base_url=judge_base_url,
    )

    async def yes_no_judge(question: str, answer: str, response: str) -> bool:
        judge_response = await judge_client.chat.completions.create(
            model=judge_model,
            messages=[
                {
                    "role": "user",
                    "content": JUDGE_PROMPT.format(
                        question=question, answer=answer, response=response
                    ),
                }
            ],
        )
        text = judge_response.choices[0].message.content or ""
        return "yes" in text.lower()

    @vf.reward(weight=0.6)
    async def correct_answer(task: vf.Task, state: vf.State) -> float:
        is_correct = await yes_no_judge(
            str(task["question"]), str(task["answer"]), completion_text(state)
        )
        state["_is_correct"] = is_correct
        return 1.0 if is_correct else 0.0

    return [correct_answer]


@vf.reward(weight=0.3)
async def cited_titles(task: vf.Task, state: vf.State) -> float:
    titles = task.get("supporting_titles") or []
    if not titles:
        return 0.0
    response = completion_text(state).lower()
    matches = sum(1 for title in titles if str(title).strip().lower() in response)
    return matches / len(titles)


@vf.reward(weight=0.1)
async def parallel_tool_calls(task: vf.Task, state: vf.State) -> float:
    _ = task
    counts = tool_calls_per_turn(state)
    if not counts:
        return 0.0
    avg_per_turn = sum(counts) / len(counts)
    return min(avg_per_turn / 3.0, 1.0)


def load_taskset(config: vf.TasksetConfig) -> vf.Taskset:
    taskset_config = WikiSearchTasksetConfig.from_config(config)
    cache: dict[str, Any] = {}

    def _load() -> dict[str, Any]:
        if "train" not in cache:
            cache["train"] = load_dataset(
                taskset_config.dataset_name,
                taskset_config.dataset_config,
                split=taskset_config.train_split,
            )
            cache["eval"] = load_dataset(
                taskset_config.dataset_name,
                taskset_config.dataset_config,
                split=taskset_config.eval_split,
            )
        return cache

    toolset = vf.Toolset(
        tools=_make_tools(taskset_config),
        scope="rollout",
    )

    return vf.Taskset(
        source=lambda: _rows(
            _load()["train"],
            taskset_config.max_turns,
            taskset_config.max_train_examples,
        ),
        eval_source=lambda: _rows(
            _load()["eval"],
            taskset_config.max_turns,
            taskset_config.max_eval_examples,
        ),
        system_prompt=taskset_config.system_prompt,
        toolsets=[toolset],
        rewards=[
            *_judge_reward_factory(
                taskset_config.judge_model,
                taskset_config.judge_base_url,
                taskset_config.judge_api_key_var,
            ),
            cited_titles,
            parallel_tool_calls,
        ],
        config=taskset_config,
    )


def load_harness(config: vf.HarnessConfig) -> vf.Harness:
    return vf.Harness(config=config)


def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(
        taskset=load_taskset(config=config.taskset),
        harness=load_harness(config=config.harness),
    )
