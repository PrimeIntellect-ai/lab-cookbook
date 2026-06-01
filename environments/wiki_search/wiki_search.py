import asyncio
import os
from dataclasses import dataclass

import chromadb
import verifiers as vf
from chromadb.api.types import Metadata
from chromadb.utils import embedding_functions
from datasets import load_dataset
from openai import AsyncOpenAI

CHROMA_DB_DIR = ".chroma_db"

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
    judge_model: str = "openai/gpt-4.1-mini"
    judge_base_url: str = "https://api.pinference.ai/api/v1"
    judge_api_key_var: str = "PRIME_API_KEY"
    embed_model: str = "text-embedding-3-small"
    embed_base_url: str = "https://api.openai.com/v1"
    embed_api_key_var: str = "OPENAI_API_KEY"
    corpus_dataset: str = "willcb/rare-wiki-pages"
    corpus_split: str = "train"
    chroma_db_dir: str = CHROMA_DB_DIR
    system_prompt: vf.PromptInput | vf.SystemPromptConfig | None = (
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
    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        _ = split
        dataset = load_dataset(self.config.dataset_name, split=self.config.dataset_split)
        for index, raw_row in enumerate(dataset):
            if self.config.max_examples is not None and index >= self.config.max_examples:
                break
            if not isinstance(raw_row, dict):
                raise TypeError("Dataset rows must be dicts.")
            yield {
                **raw_row,
                "example_id": index,
                "max_turns": self.config.max_turns,
                "prompt": [{"role": "user", "content": str(raw_row["question"])}],
            }

    def load_toolsets(self, config: WikiSearchTasksetConfig) -> vf.Toolsets:
        _ = config
        wiki = load_wiki(self.config)
        chroma_semaphore = asyncio.Semaphore(100)

        async def search_pages(query: str) -> vf.JsonData:
            """Search for top 10 relevant articles using title embedding similarity."""
            async with chroma_semaphore:
                results = await asyncio.to_thread(
                    wiki.collection.query, query_texts=[query], n_results=10
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

        async def view_sections(page_id: str) -> vf.JsonData:
            """View the sections of a page."""
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
                sections.append({"section_id": f"{page_id}:full", "section_name": "Full Page"})
            return {"sections": sections}

        async def read_section(section_id: str) -> str:
            """Read a section of a page."""
            if ":" not in section_id:
                raise ValueError("Invalid section_id format. Expected: page_id:section_name")
            page_id, section_name_id = section_id.split(":", 1)
            content = wiki.page_id_to_content[page_id]
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

        return {"wiki": vf.Toolset(tools=[search_pages, view_sections, read_section])}

    @vf.reward(weight=1.0)
    async def judge_reward(self, task: vf.Task, state: vf.State) -> float:
        completion = state.get("completion") or []
        response_text = ""
        for message in reversed(completion):
            if isinstance(message, dict) and message.get("role") == "assistant":
                response_text = str(message.get("content") or "")
                break
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
        harness=vf.Harness(config=config.harness),
    )
