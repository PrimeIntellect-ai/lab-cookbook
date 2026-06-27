import asyncio
import os
import re
from dataclasses import dataclass

import chromadb
import verifiers.v1 as vf
from chromadb.api.types import Metadata
from chromadb.utils import embedding_functions
from verifiers.v1.dialects import ChatDialect

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

CHROMA_DB_DIR = ".chroma_db"
SYSTEM = "Use the patent_search tools to answer questions about patents."
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
class PatentIndex:
    collection: chromadb.Collection
    patent_id_to_title: dict[str, str]
    patent_id_to_content: dict[str, str]
    patent_id_to_metadata: dict[str, dict[str, str]]
    patent_id_to_abstract: dict[str, str]


def normalize_id(text: str) -> str:
    return text.strip().lower().replace(" ", "_")


def extract_abstract(content: str) -> str:
    match = re.search(r"## Abstract\s*\n\n(.*?)(?=\n## |\Z)", content, re.DOTALL)
    return match.group(1).strip() if match else ""


def format_date(date_str: str) -> str:
    return (
        f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        if date_str and len(date_str) == 8
        else date_str
    )


def init_chroma(
    collection: chromadb.Collection,
    patent_id_to_title: dict[str, str],
    patent_id_to_abstract: dict[str, str],
) -> None:
    all_ids = list(patent_id_to_title)
    existing: set[str] = set()
    for index in range(0, len(all_ids), 500):
        got = collection.get(ids=all_ids[index : index + 500])
        existing.update(got.get("ids", []))
    missing = [patent_id for patent_id in all_ids if patent_id not in existing]
    documents: list[str] = []
    metadatas: list[Metadata] = []
    for patent_id in missing:
        title = patent_id_to_title[patent_id].strip()
        abstract = patent_id_to_abstract[patent_id].strip()
        text = f"{title}\n\n{abstract}" if abstract else title
        if not text:
            raise ValueError(f"Empty title and abstract for patent_id {patent_id}")
        documents.append(text)
        metadatas.append({"title": title, "abstract": abstract})
    for index in range(0, len(missing), 100):
        collection.upsert(
            ids=missing[index : index + 100],
            documents=documents[index : index + 100],
            metadatas=metadatas[index : index + 100],
        )


class PatentToolConfig(vf.ToolsetConfig):
    embed_model: str = "text-embedding-3-small"
    embed_base_url: str = "https://api.openai.com/v1"
    embed_api_key_var: str = "OPENAI_API_KEY"
    corpus_dataset: str = "jessicafeiyali/qualcomm-patents"
    corpus_file: str = "patents_formatted.json"
    chroma_db_dir: str = CHROMA_DB_DIR
    shared: bool = True


class PatentQATask(vf.Task):
    question: str
    answer: str


class JudgeConfig(vf.BaseClientConfig):
    model: str = "openai/gpt-4.1-mini"


class AdvancedPatentConfig(vf.TasksetConfig):
    corpus_dataset: str = "jessicafeiyali/qualcomm-patents"
    qa_file: str = "patent_qa_level2.jsonl"
    judge: JudgeConfig = JudgeConfig()
    tools: PatentToolConfig = PatentToolConfig()


class PatentToolset(vf.Toolset[PatentToolConfig]):
    TOOL_PREFIX = "patent"

    async def setup(self) -> None:
        from datasets import load_dataset

        titles: dict[str, str] = {}
        content: dict[str, str] = {}
        metadata: dict[str, dict[str, str]] = {}
        abstracts: dict[str, str] = {}
        rows = load_dataset(
            self.config.corpus_dataset, data_files=self.config.corpus_file, split="train"
        )
        for row in rows:
            patent_id = str(row["id"])
            text = str(row["content"])
            titles[patent_id] = str(row["title"])
            content[patent_id] = text
            metadata[patent_id] = {str(k): str(v) for k, v in (row.get("metadata") or {}).items()}
            abstracts[patent_id] = extract_abstract(text)
        client = chromadb.PersistentClient(path=self.config.chroma_db_dir)
        collection = client.get_or_create_collection(
            name="patent_titles_abstracts",
            configuration={
                "embedding_function": embedding_functions.OpenAIEmbeddingFunction(
                    model_name=self.config.embed_model,
                    api_base=self.config.embed_base_url,
                    api_key=os.getenv(self.config.embed_api_key_var, "EMPTY"),
                ),
            },
        )
        init_chroma(collection, titles, abstracts)
        self.patents = PatentIndex(collection, titles, content, metadata, abstracts)
        self.chroma_semaphore = asyncio.Semaphore(100)

    @vf.tool
    async def search_patents(self, query: str) -> dict[str, list[dict[str, str]]]:
        """Search for relevant patents using title and abstract embedding similarity."""
        async with self.chroma_semaphore:
            results = await asyncio.to_thread(
                self.patents.collection.query, query_texts=[query], n_results=10
            )
        hits: list[dict[str, str]] = []
        metadatas = results.get("metadatas") or [[]]
        for index in range(len(results["ids"][0])):
            patent_id = str(results["ids"][0][index])
            meta = metadatas[0][index] or {}
            hits.append(
                {
                    "patent_id": patent_id,
                    "title": str(meta["title"]),
                    "abstract": str(meta.get("abstract") or ""),
                }
            )
        return {"patents": hits}

    @vf.tool
    async def get_metadata(self, patent_id: str) -> dict[str, str]:
        """Get patent metadata."""
        metadata = self.patents.patent_id_to_metadata[patent_id]
        return {
            "title": self.patents.patent_id_to_title.get(patent_id, ""),
            "filing_date": format_date(metadata.get("filing_date", "")),
            "grant_date": format_date(metadata.get("grant_date", "")),
            "claim_count": metadata.get("claim_count", "0"),
        }

    @vf.tool
    async def get_abstract(self, patent_id: str) -> str:
        """Get the full abstract text for a patent."""
        return self.patents.patent_id_to_abstract[patent_id]

    @vf.tool
    async def view_sections(self, patent_id: str) -> dict[str, list[dict[str, str]]]:
        """View the sections of a patent."""
        sections: list[dict[str, str]] = []
        for line in self.patents.patent_id_to_content[patent_id].split("\n"):
            if line.startswith("#"):
                section_name = line.lstrip("#").strip()
                sections.append(
                    {
                        "section_id": f"{patent_id}:{normalize_id(section_name)}",
                        "section_name": section_name,
                    }
                )
        if not sections:
            sections.append({"section_id": f"{patent_id}:full", "section_name": "Full Patent"})
        return {"sections": sections}

    @vf.tool
    async def read_section(self, section_id: str) -> str:
        """Read a section of a patent."""
        patent_id, section_name_id = section_id.split(":", 1)
        content = self.patents.patent_id_to_content[patent_id]
        if section_name_id == "full":
            return content
        lines = content.split("\n")
        start: int | None = None
        end: int | None = None
        for index, line in enumerate(lines):
            if not line.startswith("#"):
                continue
            current = normalize_id(line.lstrip("#").strip())
            if current == section_name_id and start is None:
                start = index
            elif start is not None and end is None:
                end = index
                break
        if start is None:
            raise ValueError(f"Section not found: {section_id}")
        return "\n".join(lines[start : end or len(lines)])


class AdvancedPatentTaskset(vf.Taskset[PatentQATask, AdvancedPatentConfig]):
    def load_tasks(self) -> list[PatentQATask]:
        from datasets import load_dataset

        rows = load_dataset(
            self.config.corpus_dataset, data_files=self.config.qa_file, split="train"
        )
        return [
            PatentQATask(
                idx=i,
                prompt=f"{SYSTEM}\n\n{row['question']}",
                question=str(row["question"]),
                answer=str(row["answer"]),
            )
            for i, row in enumerate(rows)
        ]

    def tools(self, task: PatentQATask) -> list[vf.Toolset]:
        _ = task
        return [PatentToolset(self.config.tools)]

    @vf.reward(weight=1.0)
    async def judge_reward(self, task: PatentQATask, trace: vf.Trace) -> float:
        response_text = trace.assistant_messages[-1].content if trace.assistant_messages else ""
        prompt = JUDGE_PROMPT.format(
            question=task.question, answer=task.answer, response=response_text or ""
        )
        client = vf.resolve_client(self.config.judge)
        try:
            response = await client.get_response(
                ChatDialect(),
                {"messages": [{"role": "user", "content": prompt}]},
                self.config.judge.model,
                vf.SamplingConfig(),
            )
        finally:
            await client.close()
        return float("yes" in (response.message.content or "").lower())


if __name__ == "__main__":
    PatentToolset.run()


__all__ = ["AdvancedPatentTaskset"]
