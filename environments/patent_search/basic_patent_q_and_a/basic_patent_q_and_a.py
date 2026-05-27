import asyncio
import os
from dataclasses import dataclass
from pathlib import Path

import chromadb
import verifiers as vf
from chromadb.api.types import Metadata
from chromadb.utils import embedding_functions
from datasets import load_dataset
from dotenv import load_dotenv
from openai import AsyncOpenAI
from verifiers import ensure_keys

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)


CHROMA_DB_DIR = ".chroma_db"
SYSTEM_PROMPT = "Use the provided patent search tools to help answer questions about patents."
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


class BasicPatentTasksetConfig(vf.TasksetConfig):
    embed_model: str = "text-embedding-3-small"
    embed_base_url: str = "https://api.openai.com/v1"
    embed_api_key_var: str = "OPENAI_API_KEY"
    corpus_dataset: str = "jessicafeiyali/qualcomm-patents"
    corpus_file: str = "patents_formatted.json"
    qa_file: str = "patent_qa.jsonl"
    chroma_db_dir: str = CHROMA_DB_DIR


def normalize_id(text: str) -> str:
    return text.strip().lower().replace(" ", "_")


def init_chroma(collection: chromadb.Collection, patent_id_to_title: dict[str, str]) -> None:
    all_ids = list(patent_id_to_title)
    existing: set[str] = set()
    for index in range(0, len(all_ids), 500):
        batch = all_ids[index : index + 500]
        got = collection.get(ids=batch)
        existing.update(got.get("ids", []))
    missing = [patent_id for patent_id in all_ids if patent_id not in existing]
    if not missing:
        return
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


@dataclass(frozen=True)
class PatentIndex:
    collection: chromadb.Collection
    patent_id_to_title: dict[str, str]
    patent_id_to_content: dict[str, str]
    patent_id_to_metadata: dict[str, dict[str, str]]


def load_patents(config: BasicPatentTasksetConfig) -> PatentIndex:
    patent_id_to_title: dict[str, str] = {}
    patent_id_to_content: dict[str, str] = {}
    patent_id_to_metadata: dict[str, dict[str, str]] = {}
    corpus = load_dataset(
        config.corpus_dataset,
        data_files=config.corpus_file,
        split="train",
    )
    for raw_row in corpus:
        if not isinstance(raw_row, dict):
            raise TypeError("Corpus rows must be dicts.")
        patent_id = str(raw_row["id"])
        patent_id_to_title[patent_id] = str(raw_row["title"])
        patent_id_to_content[patent_id] = str(raw_row["content"])
        metadata = raw_row.get("metadata") or {}
        patent_id_to_metadata[patent_id] = {str(key): str(value) for key, value in metadata.items()}

    client = chromadb.PersistentClient(path=config.chroma_db_dir)
    collection = client.get_or_create_collection(
        name="patent_titles",
        configuration={
            "embedding_function": embedding_functions.OpenAIEmbeddingFunction(
                model_name=config.embed_model,
                api_base=config.embed_base_url,
                api_key=os.getenv(config.embed_api_key_var, "EMPTY"),
            ),
        },
    )
    init_chroma(collection, patent_id_to_title)
    return PatentIndex(
        collection=collection,
        patent_id_to_title=patent_id_to_title,
        patent_id_to_content=patent_id_to_content,
        patent_id_to_metadata=patent_id_to_metadata,
    )


class BasicPatentTaskset(vf.Taskset[BasicPatentTasksetConfig]):
    def __init__(self, config: BasicPatentTasksetConfig | None = None) -> None:
        super().__init__(config=config)
        if "toolsets" not in self.config.model_fields_set:
            self.add_toolset(self.load_toolset())

    def load_system_prompt(self) -> vf.SystemPrompt:
        return SYSTEM_PROMPT

    def load_tasks(self) -> vf.Tasks:
        dataset = load_dataset(
            self.config.corpus_dataset,
            data_files=self.config.qa_file,
            split="train",
        )
        for index, row in enumerate(dataset):
            if not isinstance(row, dict):
                raise TypeError("Dataset rows must be dicts.")
            question = str(row["question"])
            yield {
                **dict(row),
                "example_id": index,
                "prompt": [{"role": "user", "content": question}],
            }

    def load_toolset(self) -> vf.Toolset:
        ensure_keys([self.config.embed_api_key_var])
        patents = load_patents(self.config)
        chroma_semaphore = asyncio.Semaphore(100)

        async def search_patents(query: str) -> list[dict[str, str]]:
            """Search for relevant patents using title embedding similarity."""
            async with chroma_semaphore:
                results = await asyncio.to_thread(
                    patents.collection.query, query_texts=[query], n_results=10
                )
            if not results or not results["metadatas"]:
                raise ValueError(f"No results found for query: {query}")
            hits: list[dict[str, str]] = []
            for index in range(len(results["ids"][0])):
                patent_id = results["ids"][0][index]
                title = results["metadatas"][0][index]["title"]
                assert isinstance(patent_id, str)
                assert isinstance(title, str)
                hits.append({"patent_id": patent_id, "title": title})
            return hits

        async def get_metadata(patent_id: str) -> dict[str, str]:
            """Get patent metadata."""
            if patent_id not in patents.patent_id_to_metadata:
                raise ValueError(f"Patent not found: {patent_id}")
            result = dict(patents.patent_id_to_metadata[patent_id])
            result["title"] = patents.patent_id_to_title.get(patent_id, "")
            return result

        async def view_sections(patent_id: str) -> list[dict[str, str]]:
            """View the sections of a patent."""
            content = patents.patent_id_to_content[patent_id]
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
            return sections

        async def read_section(section_id: str) -> str:
            """Read a section of a patent."""
            if ":" not in section_id:
                raise ValueError("Invalid section_id format. Expected: patent_id:section_name")
            patent_id, section_name_id = section_id.split(":", 1)
            content = patents.patent_id_to_content[patent_id]
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

        return vf.Toolset(tools=[search_patents, get_metadata, view_sections, read_section])

    @vf.reward(weight=1.0)
    async def judge_reward(self, task: vf.Task, state: vf.State) -> float:
        endpoint = state.get_endpoint_config(api="chat")
        judge_client = AsyncOpenAI(api_key=endpoint["api_key"], base_url=endpoint["api_base"])
        completion = state["completion"]
        last_assistant = next(
            (msg for msg in reversed(completion) if msg.get("role") == "assistant"),
            None,
        )
        response_text = str(last_assistant["content"]) if last_assistant else ""
        response = await judge_client.chat.completions.create(
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


def load_taskset(config: BasicPatentTasksetConfig) -> vf.Taskset:
    return BasicPatentTaskset(config=config)


def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(taskset=vf.load_taskset(config=config.taskset))
