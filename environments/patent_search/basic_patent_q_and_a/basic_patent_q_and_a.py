import asyncio
import os
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


def normalize_id(text: str) -> str:
    return text.strip().lower().replace(" ", "_")


def init_chroma(collection: chromadb.Collection, patent_id_to_title: dict[str, str]) -> None:
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
        if not title:
            raise ValueError(f"Empty title for patent_id {patent_id}")
        documents.append(title)
        metadatas.append({"title": title})
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


class BasicPatentConfig(vf.TasksetConfig):
    corpus_dataset: str = "jessicafeiyali/qualcomm-patents"
    qa_file: str = "patent_qa.jsonl"
    judge: JudgeConfig = JudgeConfig()
    tools: PatentToolConfig = PatentToolConfig()


class PatentToolset(vf.Toolset[PatentToolConfig]):
    TOOL_PREFIX = "patent"

    async def setup(self) -> None:
        from datasets import load_dataset

        patent_id_to_title: dict[str, str] = {}
        patent_id_to_content: dict[str, str] = {}
        patent_id_to_metadata: dict[str, dict[str, str]] = {}
        corpus = load_dataset(
            self.config.corpus_dataset,
            data_files=self.config.corpus_file,
            split="train",
        )
        for raw_row in corpus:
            patent_id = str(raw_row["id"])
            patent_id_to_title[patent_id] = str(raw_row["title"])
            patent_id_to_content[patent_id] = str(raw_row["content"])
            metadata = raw_row.get("metadata") or {}
            patent_id_to_metadata[patent_id] = {
                str(key): str(value) for key, value in metadata.items()
            }
        client = chromadb.PersistentClient(path=self.config.chroma_db_dir)
        collection = client.get_or_create_collection(
            name="patent_titles",
            configuration={
                "embedding_function": embedding_functions.OpenAIEmbeddingFunction(
                    model_name=self.config.embed_model,
                    api_base=self.config.embed_base_url,
                    api_key=os.getenv(self.config.embed_api_key_var, "EMPTY"),
                ),
            },
        )
        init_chroma(collection, patent_id_to_title)
        self.patents = PatentIndex(
            collection, patent_id_to_title, patent_id_to_content, patent_id_to_metadata
        )
        self.chroma_semaphore = asyncio.Semaphore(100)

    @vf.tool
    async def search_patents(self, query: str) -> dict[str, list[dict[str, str]]]:
        """Search for relevant patents using title embedding similarity."""
        async with self.chroma_semaphore:
            results = await asyncio.to_thread(
                self.patents.collection.query, query_texts=[query], n_results=10
            )
        hits: list[dict[str, str]] = []
        metadatas = results.get("metadatas") or [[]]
        for index in range(len(results["ids"][0])):
            metadata = metadatas[0][index] or {}
            hits.append(
                {
                    "patent_id": str(results["ids"][0][index]),
                    "title": str(metadata.get("title", "")),
                }
            )
        return {"patents": hits}

    @vf.tool
    async def get_metadata(self, patent_id: str) -> dict[str, str]:
        """Get patent metadata."""
        if patent_id not in self.patents.patent_id_to_metadata:
            raise ValueError(f"Patent not found: {patent_id}")
        result = dict(self.patents.patent_id_to_metadata[patent_id])
        result["title"] = self.patents.patent_id_to_title.get(patent_id, "")
        return result

    @vf.tool
    async def view_sections(self, patent_id: str) -> dict[str, list[dict[str, str]]]:
        """View the sections of a patent."""
        content = self.patents.patent_id_to_content[patent_id]
        sections: list[dict[str, str]] = []
        for line in content.split("\n"):
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
        if ":" not in section_id:
            raise ValueError("Invalid section_id format. Expected: patent_id:section_name")
        patent_id, section_name_id = section_id.split(":", 1)
        content = self.patents.patent_id_to_content[patent_id]
        if section_name_id == "full":
            return content
        lines = content.split("\n")
        section_start: int | None = None
        section_end: int | None = None
        for index, line in enumerate(lines):
            if not line.startswith("#"):
                continue
            current_section = normalize_id(line.lstrip("#").strip())
            if current_section == section_name_id and section_start is None:
                section_start = index
            elif section_start is not None and section_end is None:
                section_end = index
                break
        if section_start is None:
            raise ValueError(f"Section not found: {section_id}")
        return "\n".join(lines[section_start : section_end or len(lines)])


class BasicPatentTaskset(vf.Taskset[PatentQATask, BasicPatentConfig]):
    def load_tasks(self) -> list[PatentQATask]:
        from datasets import load_dataset

        rows = load_dataset(
            self.config.corpus_dataset,
            data_files=self.config.qa_file,
            split="train",
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
            question=task.question,
            answer=task.answer,
            response=response_text or "",
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


__all__ = ["BasicPatentTaskset"]
