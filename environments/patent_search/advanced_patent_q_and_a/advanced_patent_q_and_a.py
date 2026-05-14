from __future__ import annotations

import asyncio
import os
import re
from collections.abc import Mapping
from typing import Any, cast

import verifiers.v1 as vf
from datasets import load_dataset
from dotenv import load_dotenv
from openai import AsyncOpenAI
from verifiers import Parser, ensure_keys

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

_chroma_semaphore: asyncio.Semaphore | None = None


class AdvancedPatentTasksetConfig(vf.TasksetConfig):
    max_turns: int = 10
    judge_model: str = "gpt-4.1-mini"
    judge_base_url: str = "https://api.openai.com/v1"
    judge_api_key_var: str = "OPENAI_API_KEY"
    embed_model: str = "text-embedding-3-small"
    embed_base_url: str = "https://api.openai.com/v1"
    embed_api_key_var: str = "OPENAI_API_KEY"
    corpus_dataset: str = "jessicafeiyali/qualcomm-patents"
    corpus_file: str = "patents_formatted.json"
    qa_file: str = "patent_qa_level2.jsonl"
    chroma_db_dir: str = CHROMA_DB_DIR
    system_prompt: str | None = SYSTEM_PROMPT


def chroma_semaphore() -> asyncio.Semaphore:
    global _chroma_semaphore
    if _chroma_semaphore is None:
        _chroma_semaphore = asyncio.Semaphore(100)
    return _chroma_semaphore


parser = Parser()


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


class PatentCorpus:
    def __init__(self, config: AdvancedPatentTasksetConfig):
        self.config = config
        self.patent_id_to_title: dict[str, str] = {}
        self.patent_id_to_content: dict[str, str] = {}
        self.patent_id_to_metadata: dict[str, dict] = {}
        self.patent_id_to_abstract: dict[str, str] = {}
        self.collection: Any | None = None
        self.loaded = False

    def load(self) -> None:
        if self.loaded:
            return
        corpus = load_dataset(
            self.config.corpus_dataset,
            data_files=self.config.corpus_file,
            split="train",
        )
        for raw_row in corpus:
            row = cast(dict[str, Any], raw_row)
            patent_id = str(row["id"])
            content = str(row["content"])
            self.patent_id_to_title[patent_id] = str(row["title"])
            self.patent_id_to_content[patent_id] = content
            self.patent_id_to_metadata[patent_id] = cast(dict, row.get("metadata", {}))
            self.patent_id_to_abstract[patent_id] = extract_abstract(content)
        self.loaded = True

    def get_collection(self):
        self.load()
        if self.collection is None:
            import chromadb
            from chromadb.api.types import Embeddable, EmbeddingFunction
            from chromadb.utils import embedding_functions

            openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                model_name=self.config.embed_model,
                api_base=self.config.embed_base_url,
                api_key=os.environ[self.config.embed_api_key_var],
            )
            client = chromadb.PersistentClient(path=self.config.chroma_db_dir)
            self.collection = client.get_or_create_collection(
                name="patent_titles_abstracts",
                embedding_function=cast(EmbeddingFunction[Embeddable], openai_ef),
            )
            self.init_chroma()
        return self.collection

    def init_chroma(self) -> None:
        all_ids = list(self.patent_id_to_title.keys())
        existing: set[str] = set()
        for index in range(0, len(all_ids), 500):
            batch = all_ids[index : index + 500]
            got = self.collection.get(ids=batch)
            existing.update(got.get("ids", []))
        missing = [patent_id for patent_id in all_ids if patent_id not in existing]
        if not missing:
            return
        documents: list[str] = []
        metadatas: list[dict[str, str]] = []
        for patent_id in missing:
            title = self.patent_id_to_title[patent_id].strip()
            abstract = self.patent_id_to_abstract[patent_id].strip()
            combined_text = f"{title}\n\n{abstract}" if abstract else title
            if not combined_text:
                raise ValueError(f"Empty title and abstract for patent_id {patent_id}")
            documents.append(combined_text)
            metadatas.append({"title": title, "abstract": abstract})
        batch_size = 100
        for index in range(0, len(missing), batch_size):
            self.collection.upsert(
                ids=missing[index : index + batch_size],
                documents=documents[index : index + batch_size],
                metadatas=metadatas[index : index + batch_size],
            )

    async def search_patents(self, query: str) -> list[dict]:
        """Search for relevant patents using title and abstract embedding similarity.

        Args:
            query: The query to search for.
        """
        collection = self.get_collection()
        async with chroma_semaphore():
            results = await asyncio.to_thread(collection.query, query_texts=[query], n_results=10)
        if not results or not results["metadatas"]:
            raise ValueError(f"No results found for query: {query}")
        output = []
        for index in range(len(results["ids"][0])):
            output.append(
                {
                    "patent_id": results["ids"][0][index],
                    "title": results["metadatas"][0][index]["title"],
                    "abstract": results["metadatas"][0][index].get("abstract", ""),
                }
            )
        return output

    async def get_metadata(self, patent_id: str) -> dict:
        """Get patent metadata.

        Args:
            patent_id: The ID of the patent.
        """
        self.load()
        if patent_id not in self.patent_id_to_metadata:
            raise ValueError(f"Patent not found: {patent_id}")
        metadata = self.patent_id_to_metadata[patent_id]
        return {
            "title": self.patent_id_to_title.get(patent_id, ""),
            "filing_date": format_date(str(metadata.get("filing_date", ""))),
            "grant_date": format_date(str(metadata.get("grant_date", ""))),
            "claim_count": metadata.get("claim_count", 0),
        }

    async def get_abstract(self, patent_id: str) -> str:
        """Get the full abstract text for a patent.

        Args:
            patent_id: The ID of the patent.
        """
        self.load()
        if patent_id not in self.patent_id_to_abstract:
            raise ValueError(f"Patent not found: {patent_id}")
        return self.patent_id_to_abstract[patent_id]

    async def view_sections(self, patent_id: str) -> list[dict]:
        """View the sections of a patent.

        Args:
            patent_id: The ID of the patent to view.
        """
        self.load()
        content = self.patent_id_to_content[patent_id]
        sections = []
        for line_number, line in enumerate(content.split("\n")):
            if line.startswith("#"):
                section_name = line.lstrip("#").strip()
                sections.append(
                    {
                        "section_id": f"{patent_id}:{normalize_id(section_name)}",
                        "section_name": section_name,
                        "start_line": line_number,
                    }
                )
        if not sections:
            sections.append(
                {
                    "section_id": f"{patent_id}:full",
                    "section_name": "Full Patent",
                    "start_line": 0,
                }
            )
        return [
            {"section_id": section["section_id"], "section_name": section["section_name"]}
            for section in sections
        ]

    async def read_section(self, section_id: str) -> str:
        """Read a section of a patent.

        Args:
            section_id: The ID of the section to read.
        """
        self.load()
        if ":" not in section_id:
            raise ValueError("Invalid section_id format. Expected: patent_id:section_name")
        patent_id, section_name_id = section_id.split(":", 1)
        content = self.patent_id_to_content[patent_id]
        lines = content.split("\n")
        if section_name_id == "full":
            return content
        section_start = None
        section_end = None
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


def source(config: AdvancedPatentTasksetConfig):
    dataset = load_dataset(config.corpus_dataset, data_files=config.qa_file, split="train")
    for index, raw_row in enumerate(dataset):
        row = cast(Mapping[str, object], raw_row)
        question = str(row["question"])
        yield {
            **dict(row),
            "example_id": index,
            "prompt": [{"role": "user", "content": question}],
            "max_turns": config.max_turns,
        }


def judge_reward_factory(config: AdvancedPatentTasksetConfig):
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


def load_toolset(
    config: vf.ToolsetConfig | Mapping[str, object] | None = None,
    taskset_config: AdvancedPatentTasksetConfig | None = None,
) -> vf.Toolset:
    if taskset_config is None:
        taskset_config = AdvancedPatentTasksetConfig()
    ensure_keys([taskset_config.embed_api_key_var])
    corpus = PatentCorpus(taskset_config)
    return vf.Toolset(
        tools=[
            corpus.search_patents,
            corpus.get_metadata,
            corpus.get_abstract,
            corpus.view_sections,
            corpus.read_section,
        ],
        config=config,
    )


def load_taskset(
    config: vf.TasksetConfig | Mapping[str, object] | None = None,
    max_turns: int | None = None,
    judge_model: str | None = None,
    judge_base_url: str | None = None,
    judge_api_key_var: str | None = None,
    embed_model: str | None = None,
    embed_base_url: str | None = None,
    embed_api_key_var: str | None = None,
    corpus_dataset: str | None = None,
    corpus_file: str | None = None,
    qa_file: str | None = None,
    chroma_db_dir: str | None = None,
    system_prompt: str | None = None,
) -> vf.Taskset:
    taskset_config = AdvancedPatentTasksetConfig.from_config(
        config,
        max_turns=max_turns,
        judge_model=judge_model,
        judge_base_url=judge_base_url,
        judge_api_key_var=judge_api_key_var,
        embed_model=embed_model,
        embed_base_url=embed_base_url,
        embed_api_key_var=embed_api_key_var,
        corpus_dataset=corpus_dataset,
        corpus_file=corpus_file,
        qa_file=qa_file,
        chroma_db_dir=chroma_db_dir,
        system_prompt=system_prompt,
    )
    return vf.Taskset(
        source=lambda: source(taskset_config),
        system_prompt=taskset_config.system_prompt,
        toolsets=[load_toolset(taskset_config=taskset_config)],
        rewards=[judge_reward_factory(taskset_config)],
        config=taskset_config,
    )


def load_harness(
    config: vf.HarnessConfig | Mapping[str, object] | None = None,
) -> vf.Harness:
    return vf.Harness(config=config)


def load_environment(
    config: vf.EnvConfig | Mapping[str, object] | None = None,
    max_turns: int | None = None,
    judge_model: str | None = None,
    judge_base_url: str | None = None,
    judge_api_key_var: str | None = None,
    embed_model: str | None = None,
    embed_base_url: str | None = None,
    embed_api_key_var: str | None = None,
    corpus_dataset: str | None = None,
    corpus_file: str | None = None,
    qa_file: str | None = None,
    chroma_db_dir: str | None = None,
    system_prompt: str | None = None,
) -> vf.Env:
    config = vf.EnvConfig.from_config(
        config,
        taskset=AdvancedPatentTasksetConfig.from_config(
            max_turns=max_turns,
            judge_model=judge_model,
            judge_base_url=judge_base_url,
            judge_api_key_var=judge_api_key_var,
            embed_model=embed_model,
            embed_base_url=embed_base_url,
            embed_api_key_var=embed_api_key_var,
            corpus_dataset=corpus_dataset,
            corpus_file=corpus_file,
            qa_file=qa_file,
            chroma_db_dir=chroma_db_dir,
            system_prompt=system_prompt,
        ),
    )
    return vf.Env(
        taskset=load_taskset(config=config.taskset),
        harness=load_harness(config=config.harness),
    )
