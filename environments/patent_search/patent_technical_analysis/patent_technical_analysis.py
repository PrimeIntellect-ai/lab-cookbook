import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import cast

import chromadb
import verifiers.v1 as vf
from chromadb.api.types import Metadata
from chromadb.utils import embedding_functions
from pydantic import Field
from verifiers.v1.dialects import ChatDialect

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

CHROMA_DB_DIR = ".chroma_db"
HALLUCINATION_MULTIPLIER = 0.2
FACTUAL_ERROR_MULTIPLIER = 0.5
logger = logging.getLogger(__name__)

SYSTEM = """You are an expert patent analyst with access to a corpus of wireless communications patents.

Use the patent_search tools to retrieve patent content before answering. Base answers strictly on retrieved patent content; say when information is unavailable."""

JUDGE_PROMPT = """You are a strict patent analysis grader. Evaluate the response.

QUESTION:
{question}

REFERENCE ANSWER:
{answer}

SOURCE QUOTES FROM PATENT:
{source_quotes}

RESPONSE BEING EVALUATED:
{response}

Check these key points:
{key_points_checklist}

Return JSON only:
{{
    "key_points": [{{"point": "text", "covered": true or false}}],
    "hallucination": true or false,
    "factual_error": true or false
}}"""


@dataclass(frozen=True)
class PatentIndex:
    collection: chromadb.Collection
    patent_id_to_title: dict[str, str]
    patent_id_to_content: dict[str, str]
    patent_id_to_metadata: dict[str, dict[str, str]]
    patent_id_to_abstract: dict[str, str]
    patent_id_to_claims: dict[str, str]
    patent_id_to_description: dict[str, str]


JsonValue = str | int | float | bool | None | list[str]
JsonObject = dict[str, JsonValue]
RawGroundTruth = dict[str, JsonValue | JsonObject | list[str]]


def extract_section_by_header(content: str, header: str) -> str:
    marker = "## " + header
    start = content.find(marker)
    if start == -1:
        return ""
    body = content[start + len(marker) :].lstrip()
    end = body.find("\n## ")
    return (body if end == -1 else body[:end]).strip()


def normalize_id(text: str) -> str:
    return text.strip().lower().replace(" ", "_")


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


class TechnicalPatentTask(vf.Task):
    question: str
    answer: str
    ground_truth: JsonObject = Field(default_factory=dict)


class JudgeConfig(vf.BaseClientConfig):
    model: str = "openai/gpt-4.1-mini"


class PatentTechnicalConfig(vf.TasksetConfig):
    corpus_dataset: str = "jessicafeiyali/qualcomm-patents"
    qa_file: str = "patent_qa_level3.jsonl"
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
        claims: dict[str, str] = {}
        descriptions: dict[str, str] = {}
        rows = load_dataset(
            self.config.corpus_dataset, data_files=self.config.corpus_file, split="train"
        )
        for row in rows:
            patent_id = str(row["id"])
            text = str(row["content"])
            titles[patent_id] = str(row["title"])
            content[patent_id] = text
            metadata[patent_id] = {str(k): str(v) for k, v in (row.get("metadata") or {}).items()}
            abstracts[patent_id] = extract_section_by_header(text, "Abstract")
            claims[patent_id] = extract_section_by_header(text, "Claims")
            descriptions[patent_id] = extract_section_by_header(text, "Description")
        client = chromadb.PersistentClient(path=self.config.chroma_db_dir)
        collection = client.get_or_create_collection(
            name="patent_titles_abstracts_v3",
            configuration={
                "embedding_function": embedding_functions.OpenAIEmbeddingFunction(
                    model_name=self.config.embed_model,
                    api_base=self.config.embed_base_url,
                    api_key=os.getenv(self.config.embed_api_key_var, "EMPTY"),
                ),
            },
        )
        init_chroma(collection, titles, abstracts)
        self.patents = PatentIndex(
            collection, titles, content, metadata, abstracts, claims, descriptions
        )
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
    async def get_claims_text(self, patent_id: str) -> str:
        """Get the full claims section text for a patent."""
        return self.patents.patent_id_to_claims[patent_id]

    @vf.tool
    async def get_description(self, patent_id: str) -> str:
        """Get the description section for a patent."""
        description = self.patents.patent_id_to_description[patent_id]
        return description[:15000] + "\n\n[TRUNCATED]" if len(description) > 15000 else description

    @vf.tool
    async def get_full_content(self, patent_id: str) -> str:
        """Get the full content of a patent including all sections."""
        content = self.patents.patent_id_to_content[patent_id]
        return content[:20000] + "\n\n[TRUNCATED]" if len(content) > 20000 else content

    @vf.tool
    async def compare_patents(
        self, patent_id_1: str, patent_id_2: str
    ) -> dict[str, dict[str, str]]:
        """Get side-by-side comparison data for two patents."""
        meta1 = self.patents.patent_id_to_metadata.get(patent_id_1, {})
        meta2 = self.patents.patent_id_to_metadata.get(patent_id_2, {})
        return {
            "patent_1": {
                "id": patent_id_1,
                "title": self.patents.patent_id_to_title.get(patent_id_1, ""),
                "abstract": self.patents.patent_id_to_abstract.get(patent_id_1, ""),
                "filing_date": format_date(meta1.get("filing_date", "")),
                "claim_count": meta1.get("claim_count", "0"),
            },
            "patent_2": {
                "id": patent_id_2,
                "title": self.patents.patent_id_to_title.get(patent_id_2, ""),
                "abstract": self.patents.patent_id_to_abstract.get(patent_id_2, ""),
                "filing_date": format_date(meta2.get("filing_date", "")),
                "claim_count": meta2.get("claim_count", "0"),
            },
        }

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


def normalize_ground_truth(value: object) -> JsonObject:
    if isinstance(value, dict):
        data = cast(RawGroundTruth, value)
        raw = data["ground_truth"] if "ground_truth" in data else data
        if not isinstance(raw, dict):
            return {}
        output: JsonObject = {}
        for key, item in raw.items():
            if item is None or isinstance(item, str | int | float | bool):
                output[str(key)] = item
            elif isinstance(item, list):
                output[str(key)] = [str(entry) for entry in item if entry]
        return output
    if isinstance(value, str):
        return normalize_ground_truth(json.loads(value))
    return {}


def string_list(value: JsonValue | None) -> list[str]:
    return [str(item) for item in value if item] if isinstance(value, list) else []


class PatentTechnicalTaskset(vf.Taskset[TechnicalPatentTask, PatentTechnicalConfig]):
    def load_tasks(self) -> list[TechnicalPatentTask]:
        from datasets import load_dataset

        rows = load_dataset(
            self.config.corpus_dataset, data_files=self.config.qa_file, split="train"
        )
        return [
            TechnicalPatentTask(
                idx=i,
                prompt=f"{SYSTEM}\n\n{row['question']}",
                question=str(row["question"]),
                answer=str(row["answer"]),
                ground_truth=normalize_ground_truth(row.get("info", {})),
            )
            for i, row in enumerate(rows)
        ]

    def tools(self, task: TechnicalPatentTask) -> list[vf.Toolset]:
        _ = task
        return [PatentToolset(self.config.tools)]

    @vf.reward(weight=1.0)
    async def judge_reward(self, task: TechnicalPatentTask, trace: vf.Trace) -> float:
        key_points = string_list(task.ground_truth.get("key_points"))
        if not key_points:
            return 0.0
        source_quotes = string_list(task.ground_truth.get("source_quotes"))
        response_text = trace.assistant_messages[-1].content if trace.assistant_messages else ""
        prompt = JUDGE_PROMPT.format(
            question=task.question,
            answer=task.answer,
            source_quotes="\n".join(f"- {quote}" for quote in source_quotes) or "N/A",
            response=response_text or "",
            key_points_checklist="".join(
                f"{i}. {point}\n" for i, point in enumerate(key_points, 1)
            ),
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
        text = response.message.content or ""
        start = text.find("{")
        end = text.rfind("}") + 1
        if start < 0 or end <= 0:
            return 0.0
        try:
            evaluation = json.loads(text[start:end])
        except json.JSONDecodeError:
            logger.exception("judge returned invalid JSON")
            return 0.0
        covered = sum(1 for item in evaluation.get("key_points", []) if item.get("covered"))
        score = covered / len(key_points)
        if evaluation.get("hallucination", False):
            score *= HALLUCINATION_MULTIPLIER
        if evaluation.get("factual_error", False):
            score *= FACTUAL_ERROR_MULTIPLIER
        return max(0.0, min(1.0, score))


if __name__ == "__main__":
    PatentToolset.run()


__all__ = ["PatentTechnicalTaskset"]
