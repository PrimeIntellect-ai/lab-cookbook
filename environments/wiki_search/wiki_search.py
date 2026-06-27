import asyncio
import os
from dataclasses import dataclass

import chromadb
import verifiers.v1 as vf
from chromadb.api.types import Metadata
from chromadb.utils import embedding_functions
from verifiers.v1.dialects import ChatDialect

CHROMA_DB_DIR = ".chroma_db/wiki-search"
SYSTEM = (
    "Use the wiki_search_pages, wiki_view_sections, and wiki_read_section tools to "
    "answer the question. Reply with a concise final answer when confident."
)
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

Respond either "yes" or "no" only. If a response contains incoherent text, respond with "no" even if the correct answer is also present.
"""


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


class WikiToolConfig(vf.ToolsetConfig):
    embed_model: str = "text-embedding-3-small"
    embed_base_url: str = "https://api.pinference.ai/api/v1"
    embed_api_key_var: str = "PRIME_API_KEY"
    corpus_dataset: str = "willcb/rare-wiki-pages"
    corpus_split: str = "train"
    chroma_db_dir: str = CHROMA_DB_DIR
    shared: bool = True


class TriviaTask(vf.Task):
    question: str
    answer: str


class JudgeConfig(vf.BaseClientConfig):
    model: str = "openai/gpt-4.1-mini"


class WikiSearchConfig(vf.TasksetConfig):
    dataset_name: str = "willcb/wiki-trivia-questions-v4"
    dataset_split: str = "train"
    max_examples: int | None = None
    judge: JudgeConfig = JudgeConfig()
    tools: WikiToolConfig = WikiToolConfig()


class WikiSearchToolset(vf.Toolset[WikiToolConfig]):
    TOOL_PREFIX = "wiki"

    async def setup(self) -> None:
        from datasets import load_dataset

        page_id_to_title: dict[str, str] = {}
        page_id_to_content: dict[str, str] = {}
        corpus = load_dataset(self.config.corpus_dataset, split=self.config.corpus_split)
        for raw_row in corpus:
            page_id = str(raw_row["id"])
            page_id_to_title[page_id] = str(raw_row["title"])
            page_id_to_content[page_id] = str(raw_row["content"])

        client = chromadb.PersistentClient(path=self.config.chroma_db_dir)
        collection = client.get_or_create_collection(
            name="wiki_titles",
            configuration={
                "embedding_function": embedding_functions.OpenAIEmbeddingFunction(
                    model_name=self.config.embed_model,
                    api_base=self.config.embed_base_url,
                    api_key=os.getenv(self.config.embed_api_key_var, "EMPTY"),
                ),
            },
        )
        init_chroma(collection, page_id_to_title)
        self.wiki = WikiIndex(collection, page_id_to_title, page_id_to_content)
        self.chroma_semaphore = asyncio.Semaphore(100)

    @vf.tool
    async def search_pages(self, query: str) -> dict[str, list[dict[str, str]]]:
        """Search for relevant articles using title embedding similarity."""
        async with self.chroma_semaphore:
            results = await asyncio.to_thread(
                self.wiki.collection.query, query_texts=[query], n_results=10
            )
        if not results or not results["metadatas"]:
            raise ValueError(f"No results found for query: {query}")
        pages: list[dict[str, str]] = []
        for index in range(len(results["ids"][0])):
            page_id = str(results["ids"][0][index])
            title = str(results["metadatas"][0][index]["title"])
            pages.append({"page_id": page_id, "title": title})
        return {"pages": pages}

    @vf.tool
    async def view_sections(self, page_id: str) -> dict[str, list[dict[str, str]]]:
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

    @vf.tool
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


class WikiSearchTaskset(vf.Taskset[TriviaTask, WikiSearchConfig]):
    def load_tasks(self) -> list[TriviaTask]:
        from datasets import load_dataset

        rows = load_dataset(self.config.dataset_name, split=self.config.dataset_split)
        if self.config.max_examples is not None:
            rows = rows.select(range(min(self.config.max_examples, len(rows))))
        return [
            TriviaTask(
                idx=i,
                prompt=f"{SYSTEM}\n\nQuestion: {row['question']}",
                question=str(row["question"]),
                answer=str(row["answer"]),
            )
            for i, row in enumerate(rows)
        ]

    def tools(self, task: TriviaTask) -> list[vf.Toolset]:
        _ = task
        return [WikiSearchToolset(self.config.tools)]

    @vf.reward(weight=1.0)
    async def judged(self, task: TriviaTask, trace: vf.Trace) -> float:
        response = trace.assistant_messages[-1].content if trace.assistant_messages else ""
        prompt = JUDGE_PROMPT.format(
            question=task.question,
            answer=task.answer,
            response=response or "",
        )
        client = vf.resolve_client(self.config.judge)
        try:
            verdict = await client.get_response(
                ChatDialect(),
                {"messages": [{"role": "user", "content": prompt}]},
                self.config.judge.model,
                vf.SamplingConfig(),
            )
        finally:
            await client.close()
        return float("yes" in (verdict.message.content or "").lower())


if __name__ == "__main__":
    WikiSearchToolset.run()


__all__ = ["WikiSearchTaskset"]
