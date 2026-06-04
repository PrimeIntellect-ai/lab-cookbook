import asyncio
import os
from dataclasses import dataclass

import chromadb
import verifiers as vf
from chromadb.api.types import Metadata
from chromadb.utils import embedding_functions
from datasets import load_dataset
from openai import AsyncOpenAI

CHROMA_DB_DIR = ".chroma_db/wiki-search"

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
    train_split: str = "train"
    eval_split: str = "train"
    max_examples: int | None = None
    judge_model: str = "openai/gpt-4.1-mini"
    judge_base_url: str = "https://api.pinference.ai/api/v1"
    judge_api_key_var: str = "PRIME_API_KEY"
    embed_model: str = "text-embedding-3-small"
    embed_base_url: str = "https://api.pinference.ai/api/v1"
    embed_api_key_var: str = "PRIME_API_KEY"
    corpus_dataset: str = "willcb/rare-wiki-pages"
    corpus_split: str = "train"
    chroma_db_dir: str = CHROMA_DB_DIR
    system_prompt: vf.SystemPrompt | vf.SystemPromptConfig | None = (
        "Use the provided Wikipedia search tools to help answer questions."
    )


@dataclass(frozen=True)
class WikiIndex:
    collection: chromadb.Collection
    page_id_to_title: dict[str, str]
    page_id_to_content: dict[str, str]


def normalize_id(text: str) -> str:
    return text.strip().lower().replace(" ", "_")


def init_chroma(collection: chromadb.Collection, page_id_to_title: dict[str, str]) -> None:
    all_ids = list(page_id_to_title)
    existing: set[str] = set()
    for index in range(0, len(all_ids), 500):
        batch = all_ids[index : index + 500]
        got = collection.get(ids=batch)
        existing.update(got.get("ids", []))
    missing = [page_id for page_id in all_ids if page_id not in existing]
    if not missing:
        return
    documents: list[str] = []
    metadatas: list[Metadata] = []
    for page_id in missing:
        title = str(page_id_to_title[page_id]).strip()
        if not title:
            raise ValueError(f"Empty title for page_id {page_id}")
        documents.append(title)
        metadatas.append({"title": title})
    for index in range(0, len(missing), 100):
        collection.upsert(
            ids=missing[index : index + 100],
            documents=documents[index : index + 100],
            metadatas=metadatas[index : index + 100],
        )


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

    client = chromadb.PersistentClient(path=config.chroma_db_dir)
    collection = client.get_or_create_collection(
        name="wiki_titles",
        configuration={
            "embedding_function": embedding_functions.OpenAIEmbeddingFunction(
                model_name=config.embed_model,
                api_base=config.embed_base_url,
                api_key=os.getenv(config.embed_api_key_var, "EMPTY"),
            ),
        },
    )
    init_chroma(collection, page_id_to_title)
    return WikiIndex(
        collection=collection,
        page_id_to_title=page_id_to_title,
        page_id_to_content=page_id_to_content,
    )


class WikiSearchTaskset(vf.Taskset[WikiSearchTasksetConfig]):
    wiki: WikiIndex
    chroma_semaphore: asyncio.Semaphore

    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        source_split = self.config.train_split if split == "train" else self.config.eval_split
        dataset = load_dataset(self.config.dataset_name, split=source_split)
        if self.config.max_examples is not None:
            dataset = dataset.select(range(min(self.config.max_examples, len(dataset))))
        return dataset

    def load_toolsets(self, config: WikiSearchTasksetConfig) -> vf.Toolsets:
        vf.ensure_keys([config.embed_api_key_var])
        self.wiki = load_wiki(config)
        self.chroma_semaphore = asyncio.Semaphore(100)
        return {
            "wiki": vf.Toolset(tools=[self.search_pages, self.view_sections, self.read_section])
        }

    async def search_pages(self, query: str) -> vf.JsonData:
        """Search for top 10 relevant articles using title embedding similarity."""
        async with self.chroma_semaphore:
            results = await asyncio.to_thread(
                self.wiki.collection.query, query_texts=[query], n_results=10
            )
        if not results or not results["metadatas"]:
            raise ValueError(f"No results found for query: {query}")
        pages: list[dict[str, str]] = []
        for index in range(len(results["ids"][0])):
            page_id = results["ids"][0][index]
            title = results["metadatas"][0][index]["title"]
            assert isinstance(page_id, str)
            assert isinstance(title, str)
            pages.append({"page_id": page_id, "title": title})
        return {"pages": pages}

    async def view_sections(self, page_id: str) -> vf.JsonData:
        """View the sections of a page."""
        content = self.wiki.page_id_to_content[page_id]
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
            sections.append({"section_id": f"{page_id}:full", "section_name": "Full Page"})
        return {"sections": sections}

    async def read_section(self, section_id: str) -> str:
        """Read a section of a page."""
        if ":" not in section_id:
            raise ValueError("Invalid section_id format. Expected: page_id:section_name")
        page_id, section_name_id = section_id.split(":", 1)
        content = self.wiki.page_id_to_content[page_id]
        if section_name_id == "full":
            return content
        lines = content.split("\n")
        section_start: int | None = None
        section_end: int | None = None
        for line_index, line in enumerate(lines):
            if not line.startswith("#"):
                continue
            current_section = normalize_id(line.lstrip("#").strip())
            if current_section == section_name_id and section_start is None:
                section_start = line_index
            elif section_start is not None and section_end is None:
                section_end = line_index
                break
        if section_start is None:
            raise ValueError(f"Section not found: {section_id}")
        return "\n".join(lines[section_start : section_end or len(lines)])

    @vf.reward(weight=1.0)
    async def judge_reward(self, task: vf.Task, state: vf.State) -> float:
        vf.ensure_keys([self.config.judge_api_key_var])
        completion = state.get("completion") or []
        messages = vf.get_messages(completion, role="assistant")
        response_text = str(messages[-1].content or "") if messages else ""
        judge = AsyncOpenAI(
            api_key=os.getenv(self.config.judge_api_key_var, ""),
            base_url=self.config.judge_base_url,
        )
        try:
            response = await judge.chat.completions.create(
                model=self.config.judge_model,
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
        finally:
            await judge.close()
        text = response.choices[0].message.content or ""
        return 1.0 if "yes" in text.lower() else 0.0


def load_taskset(config: WikiSearchTasksetConfig) -> WikiSearchTaskset:
    return WikiSearchTaskset(config=config)


def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(
        taskset=vf.load_taskset(config=config.taskset),
        harness=vf.load_harness(config=config.harness),
    )
