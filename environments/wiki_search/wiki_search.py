from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from typing import cast

import chromadb
import verifiers.v1 as vf
from chromadb.api.types import Embeddable, EmbeddingFunction
from chromadb.utils import embedding_functions
from datasets import load_dataset
from openai import AsyncOpenAI
from verifiers import Parser, ensure_keys

CHROMA_DB_DIR = ".chroma_db"

SYSTEM_PROMPT = "Use the provided Wikipedia search tools to help answer questions."

JUDGE_PROMPT = """Given a ground truth answer and a response, determine if the response is both correct and coherent.

Question:
```
{question}
```

Ground truth answer:
```
{answer}
```

Response:
```
{response}
```

Respond either "yes" or "no" only.

If a response contains incoherent text, respond with "no" even if the correct answer is also present.
"""


class WikiSearchTasksetConfig(vf.TasksetConfig):
    dataset_name: str = "willcb/wiki-trivia-questions-v4"
    dataset_split: str = "train"
    max_examples: int | None = None
    max_turns: int = 10
    judge_model: str = "gpt-4.1-mini"
    judge_base_url: str = "https://api.openai.com/v1"
    judge_api_key_var: str = "OPENAI_API_KEY"
    embed_model: str = "text-embedding-3-small"
    embed_base_url: str = "https://api.openai.com/v1"
    embed_api_key_var: str = "OPENAI_API_KEY"
    corpus_dataset: str = "willcb/rare-wiki-pages"
    corpus_split: str = "train"
    chroma_db_dir: str = CHROMA_DB_DIR


parser = Parser()

_chroma_semaphore: asyncio.Semaphore | None = None


def get_chroma_semaphore() -> asyncio.Semaphore:
    global _chroma_semaphore
    if _chroma_semaphore is None:
        _chroma_semaphore = asyncio.Semaphore(100)
    return _chroma_semaphore


def normalize_id(text: str) -> str:
    return text.strip().lower().replace(" ", "_")


def init_chroma(collection, page_id_to_title: dict[str, str]) -> None:
    all_ids = list(page_id_to_title)
    existing: set[str] = set()
    for i in range(0, len(all_ids), 500):
        batch = all_ids[i : i + 500]
        got = collection.get(ids=batch)
        existing.update(got.get("ids", []))
    missing = [page_id for page_id in all_ids if page_id not in existing]
    if not missing:
        return
    documents: list[str] = []
    metadatas: list[dict[str, str]] = []
    for page_id in missing:
        title = str(page_id_to_title[page_id]).strip()
        if not title:
            raise ValueError(f"Empty title for page_id {page_id}")
        documents.append(title)
        metadatas.append({"title": title})
    for i in range(0, len(missing), 100):
        collection.upsert(
            ids=missing[i : i + 100],
            documents=documents[i : i + 100],
            metadatas=metadatas[i : i + 100],
        )


def load_wiki(config: WikiSearchTasksetConfig) -> dict[str, object]:
    ensure_keys([config.embed_api_key_var])
    page_id_to_title: dict[str, str] = {}
    page_id_to_content: dict[str, str] = {}
    corpus = load_dataset(config.corpus_dataset, split=config.corpus_split)
    for raw_row in corpus:
        row = cast(Mapping[str, object], raw_row)
        page_id = str(row["id"])
        page_id_to_title[page_id] = str(row["title"])
        page_id_to_content[page_id] = str(row["content"])

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        model_name=config.embed_model,
        api_base=config.embed_base_url,
        api_key=os.getenv(config.embed_api_key_var, "EMPTY"),
    )
    client = chromadb.PersistentClient(path=config.chroma_db_dir)
    collection = client.get_or_create_collection(
        name="wiki_titles",
        embedding_function=cast(EmbeddingFunction[Embeddable], openai_ef),
    )
    init_chroma(collection, page_id_to_title)
    return {
        "collection": collection,
        "page_id_to_title": page_id_to_title,
        "page_id_to_content": page_id_to_content,
    }


async def search_pages(query: str, wiki: dict[str, object]) -> list[dict[str, str]]:
    """Search for top 10 relevant articles using title embedding similarity."""
    collection = cast(chromadb.Collection, wiki["collection"])
    async with get_chroma_semaphore():
        results = await asyncio.to_thread(
            collection.query, query_texts=[query], n_results=10
        )
    if not results or not results["metadatas"]:
        raise ValueError(f"No results found for query: {query}")
    output: list[dict[str, str]] = []
    for i in range(len(results["ids"][0])):
        output.append(
            {
                "page_id": results["ids"][0][i],
                "title": results["metadatas"][0][i]["title"],
            }
        )
    return output


async def view_sections(page_id: str, wiki: dict[str, object]) -> list[dict[str, str]]:
    """View the sections of a page."""
    page_id_to_content = cast(dict[str, str], wiki["page_id_to_content"])
    content = page_id_to_content[page_id]
    sections: list[dict[str, str]] = []
    for line in content.split("\n"):
        if line.startswith("#"):
            section_name = line.lstrip("#").strip()
            sections.append(
                {
                    "section_id": f"{page_id}:{normalize_id(section_name)}",
                    "section_name": section_name,
                }
            )
    if not sections:
        sections.append(
            {
                "section_id": f"{page_id}:full",
                "section_name": "Full Page",
            }
        )
    return sections


async def read_section(section_id: str, wiki: dict[str, object]) -> str:
    """Read a section of a page."""
    if ":" not in section_id:
        raise ValueError("Invalid section_id format. Expected: page_id:section_name")
    page_id, section_name_id = section_id.split(":", 1)
    page_id_to_content = cast(dict[str, str], wiki["page_id_to_content"])
    content = page_id_to_content[page_id]
    if section_name_id == "full":
        return content
    lines = content.split("\n")
    section_start: int | None = None
    section_end: int | None = None
    for i, line in enumerate(lines):
        if not line.startswith("#"):
            continue
        current_section = normalize_id(line.lstrip("#").strip())
        if current_section == section_name_id and section_start is None:
            section_start = i
        elif section_start is not None and section_end is None:
            section_end = i
            break
    if section_start is None:
        raise ValueError(f"Section not found: {section_id}")
    return "\n".join(lines[section_start : section_end or len(lines)])


def source(config: WikiSearchTasksetConfig):
    def _iter():
        dataset = load_dataset(config.dataset_name, split=config.dataset_split)
        for index, raw_row in enumerate(dataset):
            if config.max_examples is not None and index >= config.max_examples:
                break
            row = cast(Mapping[str, object], raw_row)
            yield {
                **dict(row),
                "example_id": index,
                "max_turns": config.max_turns,
                "prompt": [{"role": "user", "content": str(row["question"])}],
            }

    return _iter


def judge_reward_factory(config: WikiSearchTasksetConfig):
    ensure_keys([config.judge_api_key_var])
    judge_client = AsyncOpenAI(
        api_key=os.environ[config.judge_api_key_var],
        base_url=config.judge_base_url,
    )

    @vf.reward(weight=1.0)
    async def judge_reward(task: vf.Task, state: vf.State) -> float:
        response = await judge_client.chat.completions.create(
            model=config.judge_model,
            messages=[
                {
                    "role": "user",
                    "content": JUDGE_PROMPT.format(
                        question=task["question"],
                        answer=task["answer"],
                        response=parser.parse_answer(state["completion"]) or "",
                    ),
                }
            ],
        )
        text = response.choices[0].message.content or ""
        return 1.0 if "yes" in text.lower() else 0.0

    return judge_reward


def load_toolset(config: vf.ToolsetConfig, taskset_config: WikiSearchTasksetConfig) -> vf.Toolset:
    return vf.Toolset(
        tools=[search_pages, view_sections, read_section],
        objects={"wiki": lambda: load_wiki(taskset_config)},
        bindings={
            "search_pages.wiki": "objects.wiki",
            "view_sections.wiki": "objects.wiki",
            "read_section.wiki": "objects.wiki",
        },
        config=config,
    )


def load_taskset(config: vf.TasksetConfig) -> vf.Taskset:
    taskset_config = WikiSearchTasksetConfig.from_config(config)
    return vf.Taskset(
        source=source(taskset_config),
        system_prompt=SYSTEM_PROMPT,
        toolsets=[load_toolset(vf.ToolsetConfig(), taskset_config)],
        rewards=[judge_reward_factory(taskset_config)],
        config=taskset_config,
    )


def load_harness(config: vf.HarnessConfig) -> vf.Harness:
    return vf.Harness(config=config)


def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(
        taskset=load_taskset(config=config.taskset),
        harness=load_harness(config=config.harness),
    )
