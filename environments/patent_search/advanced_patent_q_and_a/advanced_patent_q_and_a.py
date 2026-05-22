import asyncio
import os
import re
from dataclasses import dataclass

import chromadb
import verifiers.v1 as vf
from chromadb.api.types import Embeddable, EmbeddingFunction
from chromadb.utils import embedding_functions
from datasets import load_dataset
from dotenv import load_dotenv
from openai import AsyncOpenAI
from verifiers import ensure_keys

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

load_dotenv()


CHROMA_DB_DIR = ".chroma_db_patents"
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


class AdvancedPatentTasksetConfig(vf.TasksetConfig):
    embed_model: str = "text-embedding-3-small"
    embed_base_url: str = "https://api.openai.com/v1"
    embed_api_key_var: str = "OPENAI_API_KEY"
    corpus_dataset: str = "jessicafeiyali/qualcomm-patents"
    corpus_file: str = "patents_formatted.json"
    qa_file: str = "patent_qa_level2.jsonl"
    chroma_db_dir: str = CHROMA_DB_DIR


class AdvancedPatentEnvConfig(vf.EnvConfig):
    taskset: AdvancedPatentTasksetConfig
    harness: vf.HarnessConfig


class AdvancedPatentTaskset(vf.Taskset):
    config_type = AdvancedPatentTasksetConfig


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


def extract_abstract(content: str) -> str:
    abstract_match = re.search(r"## Abstract\s*\n\n(.*?)(?=\n## |\Z)", content, re.DOTALL)
    if abstract_match:
        return abstract_match.group(1).strip()
    return ""


def format_date(date_str: str) -> str:
    if date_str and len(date_str) == 8:
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    return date_str


def init_chroma(
    collection: chromadb.Collection,
    patent_id_to_title: dict[str, str],
    patent_id_to_abstract: dict[str, str],
) -> None:
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
    metadatas: list[dict[str, str]] = []
    for patent_id in missing:
        title = patent_id_to_title[patent_id].strip()
        abstract = patent_id_to_abstract[patent_id].strip()
        combined_text = f"{title}\n\n{abstract}" if abstract else title
        if not combined_text:
            raise ValueError(f"Empty title and abstract for patent_id {patent_id}")
        documents.append(combined_text)
        metadatas.append({"title": title, "abstract": abstract})
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
    patent_id_to_abstract: dict[str, str]


def load_patents(config: AdvancedPatentTasksetConfig) -> PatentIndex:
    patent_id_to_title: dict[str, str] = {}
    patent_id_to_content: dict[str, str] = {}
    patent_id_to_metadata: dict[str, dict[str, str]] = {}
    patent_id_to_abstract: dict[str, str] = {}
    corpus = load_dataset(
        config.corpus_dataset,
        data_files=config.corpus_file,
        split="train",
    )
    for raw_row in corpus:
        if not isinstance(raw_row, dict):
            raise TypeError("Corpus rows must be dicts.")
        patent_id = str(raw_row["id"])
        content = str(raw_row["content"])
        patent_id_to_title[patent_id] = str(raw_row["title"])
        patent_id_to_content[patent_id] = content
        metadata = raw_row.get("metadata") or {}
        patent_id_to_metadata[patent_id] = {str(key): str(value) for key, value in metadata.items()}
        patent_id_to_abstract[patent_id] = extract_abstract(content)

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        model_name=config.embed_model,
        api_base=config.embed_base_url,
        api_key=os.getenv(config.embed_api_key_var, "EMPTY"),
    )
    embedding_fn: EmbeddingFunction[Embeddable] = openai_ef
    client = chromadb.PersistentClient(path=config.chroma_db_dir)
    collection = client.get_or_create_collection(
        name="patent_titles_abstracts",
        embedding_function=embedding_fn,
    )
    init_chroma(collection, patent_id_to_title, patent_id_to_abstract)
    return PatentIndex(
        collection=collection,
        patent_id_to_title=patent_id_to_title,
        patent_id_to_content=patent_id_to_content,
        patent_id_to_metadata=patent_id_to_metadata,
        patent_id_to_abstract=patent_id_to_abstract,
    )


# Module-scope patent index handle: the chroma collection + dictionaries are an
# expensive shared resource that must be built exactly once per process. A
# process-level handle is the only way to assert that invariant across the
# tool closures (see environments/AGENTS.md "Rare exception").
_patents_singleton: PatentIndex | None = None


def _get_patents(config: AdvancedPatentTasksetConfig) -> PatentIndex:
    global _patents_singleton
    if _patents_singleton is None:
        _patents_singleton = load_patents(config)
    return _patents_singleton


def source(config: AdvancedPatentTasksetConfig):
    dataset = load_dataset(config.corpus_dataset, data_files=config.qa_file, split="train")
    for index, row in enumerate(dataset):
        question = str(row["question"])
        yield {
            **dict(row),
            "example_id": index,
            "prompt": [{"role": "user", "content": question}],
        }


@vf.reward(weight=1.0)
async def judge_reward(task: vf.Task, state: vf.State) -> float:
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


def load_taskset(config: AdvancedPatentTasksetConfig) -> AdvancedPatentTaskset:
    ensure_keys([config.embed_api_key_var])

    async def search_patents(query: str) -> list[dict[str, str]]:
        """Search for relevant patents using title and abstract embedding similarity.

        Args:
            query: The query to search for.
        """
        patents = _get_patents(config)
        async with get_chroma_semaphore():
            results = await asyncio.to_thread(
                patents.collection.query, query_texts=[query], n_results=10
            )
        if not results or not results["metadatas"]:
            raise ValueError(f"No results found for query: {query}")
        return [
            {
                "patent_id": results["ids"][0][index],
                "title": results["metadatas"][0][index]["title"],
                "abstract": results["metadatas"][0][index].get("abstract", ""),
            }
            for index in range(len(results["ids"][0]))
        ]

    async def get_metadata(patent_id: str) -> dict[str, str]:
        """Get patent metadata.

        Args:
            patent_id: The ID of the patent.
        """
        patents = _get_patents(config)
        if patent_id not in patents.patent_id_to_metadata:
            raise ValueError(f"Patent not found: {patent_id}")
        metadata = patents.patent_id_to_metadata[patent_id]
        return {
            "title": patents.patent_id_to_title.get(patent_id, ""),
            "filing_date": format_date(metadata.get("filing_date", "")),
            "grant_date": format_date(metadata.get("grant_date", "")),
            "claim_count": metadata.get("claim_count", "0"),
        }

    async def get_abstract(patent_id: str) -> str:
        """Get the full abstract text for a patent.

        Args:
            patent_id: The ID of the patent.
        """
        patents = _get_patents(config)
        if patent_id not in patents.patent_id_to_abstract:
            raise ValueError(f"Patent not found: {patent_id}")
        return patents.patent_id_to_abstract[patent_id]

    async def view_sections(patent_id: str) -> list[dict[str, str]]:
        """View the sections of a patent.

        Args:
            patent_id: The ID of the patent to view.
        """
        content = _get_patents(config).patent_id_to_content[patent_id]
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
        """Read a section of a patent.

        Args:
            section_id: The ID of the section to read.
        """
        if ":" not in section_id:
            raise ValueError("Invalid section_id format. Expected: patent_id:section_name")
        patent_id, section_name_id = section_id.split(":", 1)
        content = _get_patents(config).patent_id_to_content[patent_id]
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

    toolset = vf.Toolset(
        tools=[search_patents, get_metadata, get_abstract, view_sections, read_section],
    )
    return AdvancedPatentTaskset(
        source=lambda: source(config),
        system_prompt=SYSTEM_PROMPT,
        toolsets=[toolset],
        rewards=[judge_reward],
        config=config,
    )


def load_environment(config: AdvancedPatentEnvConfig) -> vf.Env:
    return vf.Env(
        taskset=load_taskset(config=config.taskset),
        harness=vf.Harness(config=config.harness),
    )
