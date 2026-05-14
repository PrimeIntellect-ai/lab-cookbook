import asyncio
import os
from dataclasses import dataclass

import chromadb
import verifiers.v1 as vf
from chromadb.api.types import Embeddable, EmbeddingFunction
from chromadb.utils import embedding_functions
from datasets import load_dataset
from openai import AsyncOpenAI
from pydantic import model_validator
from typing_extensions import Self
from verifiers import ensure_keys

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
    embed_model: str = "text-embedding-3-small"
    embed_base_url: str = "https://api.openai.com/v1"
    embed_api_key_var: str = "OPENAI_API_KEY"
    corpus_dataset: str = "willcb/rare-wiki-pages"
    corpus_split: str = "train"
    chroma_db_dir: str = CHROMA_DB_DIR

    @model_validator(mode="after")
    def _ensure_required_keys(self) -> "Self":
        ensure_keys([self.embed_api_key_var])
        return self


# Module-scope semaphore: chromadb client is sync; cap concurrent thread-offloaded
# queries so a burst of rollouts can't exhaust the default executor.
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


@dataclass(frozen=True)
class WikiIndex:
    collection: chromadb.Collection
    page_id_to_title: dict[str, str]
    page_id_to_content: dict[str, str]


def load_wiki(config: WikiSearchTasksetConfig) -> WikiIndex:
    page_id_to_title: dict[str, str] = {}
    page_id_to_content: dict[str, str] = {}
    corpus = load_dataset(config.corpus_dataset, split=config.corpus_split)
    for raw_row in corpus:
        if not isinstance(raw_row, dict):
            raise TypeError("Corpus rows must be dicts.")
        page_id = str(raw_row["id"])
        page_id_to_title[page_id] = str(raw_row["title"])
        page_id_to_content[page_id] = str(raw_row["content"])

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        model_name=config.embed_model,
        api_base=config.embed_base_url,
        api_key=os.getenv(config.embed_api_key_var, "EMPTY"),
    )
    embedding_fn: EmbeddingFunction[Embeddable] = openai_ef
    client = chromadb.PersistentClient(path=config.chroma_db_dir)
    collection = client.get_or_create_collection(
        name="wiki_titles",
        embedding_function=embedding_fn,
    )
    init_chroma(collection, page_id_to_title)
    return WikiIndex(
        collection=collection,
        page_id_to_title=page_id_to_title,
        page_id_to_content=page_id_to_content,
    )


# Module-scope wiki index handle: the chroma collection + dictionaries are an
# expensive shared resource that must be built exactly once per process. A
# process-level handle is the only way to assert that invariant across the
# tool closures (see environments/AGENTS.md "Rare exception").
_wiki_singleton: WikiIndex | None = None


def _get_wiki(config: WikiSearchTasksetConfig) -> WikiIndex:
    global _wiki_singleton
    if _wiki_singleton is None:
        _wiki_singleton = load_wiki(config)
    return _wiki_singleton


async def _search_pages(query: str, wiki: WikiIndex) -> list[dict[str, str]]:
    async with get_chroma_semaphore():
        results = await asyncio.to_thread(wiki.collection.query, query_texts=[query], n_results=10)
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


async def _view_sections(page_id: str, wiki: WikiIndex) -> list[dict[str, str]]:
    content = wiki.page_id_to_content[page_id]
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


async def _read_section(section_id: str, wiki: WikiIndex) -> str:
    if ":" not in section_id:
        raise ValueError("Invalid section_id format. Expected: page_id:section_name")
    page_id, section_name_id = section_id.split(":", 1)
    content = wiki.page_id_to_content[page_id]
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


def _source(config: WikiSearchTasksetConfig):
    def _iter():
        dataset = load_dataset(config.dataset_name, split=config.dataset_split)
        for index, raw_row in enumerate(dataset):
            if config.max_examples is not None and index >= config.max_examples:
                break
            if not isinstance(raw_row, dict):
                raise TypeError("Dataset rows must be dicts.")
            yield {
                **raw_row,
                "example_id": index,
                "max_turns": config.max_turns,
                "prompt": [{"role": "user", "content": str(raw_row["question"])}],
            }

    return _iter


@vf.reward(weight=1.0)
async def judge_reward(task: vf.Task, state: vf.State) -> float:
    completion = state.get("completion") or []
    response_text = ""
    for message in reversed(completion):
        if isinstance(message, dict) and message.get("role") == "assistant":
            response_text = str(message.get("content") or "")
            break
    endpoint = state.get_endpoint_config(api="chat")
    judge = AsyncOpenAI(api_key=endpoint["api_key"], base_url=endpoint["api_base"])
    response = await judge.chat.completions.create(
        model=endpoint["model"],
        messages=[
            {
                "role": "user",
                "content": JUDGE_PROMPT.format(
                    question=task["question"],
                    answer=task["answer"],
                    response=response_text,
                ),
            }
        ],
    )
    text = response.choices[0].message.content or ""
    return 1.0 if "yes" in text.lower() else 0.0


def load_environment(config: vf.EnvConfig) -> vf.Env:
    cfg = WikiSearchTasksetConfig(config.taskset)

    async def search_pages(query: str) -> list[dict[str, str]]:
        """Search for top 10 relevant articles using title embedding similarity."""
        return await _search_pages(query, _get_wiki(cfg))

    async def view_sections(page_id: str) -> list[dict[str, str]]:
        """View the sections of a page."""
        return await _view_sections(page_id, _get_wiki(cfg))

    async def read_section(section_id: str) -> str:
        """Read a section of a page."""
        return await _read_section(section_id, _get_wiki(cfg))

    toolset = vf.Toolset(tools=[search_pages, view_sections, read_section])
    taskset = vf.Taskset(
        source=_source(cfg),
        system_prompt=SYSTEM_PROMPT,
        toolsets=[toolset],
        rewards=[judge_reward],
        config=cfg,
    )
    return vf.Env(
        taskset=taskset,
        harness=vf.Harness(config=config.harness),
    )
