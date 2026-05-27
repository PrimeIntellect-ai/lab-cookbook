import asyncio
import json
import logging
import os
from dataclasses import dataclass

import chromadb
import verifiers as vf
from chromadb.api.types import Metadata
from chromadb.utils import embedding_functions
from datasets import load_dataset
from dotenv import load_dotenv
from openai import AsyncOpenAI
from verifiers import ensure_keys

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

load_dotenv()


CHROMA_DB_DIR = ".chroma_db"
HALLUCINATION_MULTIPLIER = 0.2
FACTUAL_ERROR_MULTIPLIER = 0.5

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert patent analyst with access to a corpus of patents in the wireless communications domain.

Your task is to answer questions about these patents accurately and comprehensively.

AVAILABLE TOOLS:
- search_patents(query): Find relevant patents by semantic search on titles and abstracts
- get_metadata(patent_id): Get filing date, grant date, and claim count
- get_abstract(patent_id): Get the full abstract text
- get_claims_text(patent_id): Get the full claims section
- get_description(patent_id): Get the description section
- get_full_content(patent_id): Get complete patent content
- compare_patents(patent_id_1, patent_id_2): Get side-by-side comparison data
- view_sections(patent_id): List all sections in a patent
- read_section(section_id): Read a specific section

RESPONSE GUIDELINES:
1. Always retrieve the relevant patent content before answering
2. Base your answers strictly on the patent content - do not hallucinate
3. For technical summaries, be concise (2-3 sentences)
4. For claim analysis, quote relevant claim language
5. For comparison questions, address both patents fairly
6. If information is not available, say so clearly"""

JUDGE_PROMPT = """You are a strict patent analysis grader. Evaluate the response using the checklist below. Err on the side of lower scores. Scores above 0.5 should be incredibly rare.

QUESTION:
{question}

REFERENCE ANSWER:
{answer}

SOURCE QUOTES FROM PATENT:
{source_quotes}

RESPONSE BEING EVALUATED:
{response}

STEP 1 - KEY POINT COVERAGE
For each key point, determine if the response COVERS it.
A key point is COVERED if and only if:
- The response addresses the same specific technical concept
- Using different terminology is acceptable if the technical meaning is equivalent
- The response must include the specific detail, not just reference the general topic area
- Partial coverage does NOT count and should be given a score of 0

EXAMPLES:
  Key point: "Uses beam-based spatial multiplexing with rank adaptation"
  Response says: "The patent uses spatial multiplexing techniques"
  -> covered: false (missing beam-based approach and rank adaptation specifics)

  Key point: "Uses beam-based spatial multiplexing with rank adaptation"
  Response says: "Employs beam-based MIMO with dynamic rank selection"
  -> covered: true (equivalent technical meaning)

KEY POINTS TO CHECK:
{key_points_checklist}

STEP 2 - HALLUCINATION CHECK
A hallucination is a specific technical claim in the response that is:
- Not supported by the reference answer, source quotes, OR reasonable domain knowledge
- Invented details such as fabricated patent numbers, claim numbers, dates, or technical specifications
Do NOT flag as hallucination:
- Well-known technical background (e.g., "5G NR is a cellular standard")
- Reasonable inferences from the patent content
- Accurate details the model may have retrieved from the patent even if not in the source quotes above

Generally, if anything not widely accepted shows up in the answer that is not supported by the reference answer, mark as hallucination.

STEP 3 - FACTUAL ERROR CHECK
A factual error is a statement that directly CONTRADICTS the patent content shown above.

First, write your step-by-step reasoning for each key point, hallucination check, and factual error check.
Then output your final verdict as a JSON block.

REASONING:
[Analyze each key point, then hallucinations, then factual errors]

VERDICT:
{{
    "key_points": [{{"point": "text", "covered": true or false}}],
    "hallucination": true or false,
    "factual_error": true or false
}}"""


class PatentTechnicalTasksetConfig(vf.TasksetConfig):
    embed_model: str = "text-embedding-3-small"
    embed_base_url: str = "https://api.openai.com/v1"
    embed_api_key_var: str = "OPENAI_API_KEY"
    corpus_dataset: str = "jessicafeiyali/qualcomm-patents"
    corpus_file: str = "patents_formatted.json"
    qa_file: str = "patent_qa_level3.jsonl"
    chroma_db_dir: str = CHROMA_DB_DIR


def extract_section_by_header(content: str, header: str) -> str:
    section_marker = "## " + header
    start_index = content.find(section_marker)
    if start_index == -1:
        return ""
    content_after_header = content[start_index + len(section_marker) :]
    newline_index = content_after_header.find("\n")
    if newline_index != -1:
        content_after_header = content_after_header[newline_index + 1 :]
    next_section_index = content_after_header.find("\n## ")
    section_content = (
        content_after_header
        if next_section_index == -1
        else content_after_header[:next_section_index]
    )
    return section_content.strip()


def extract_abstract(content: str) -> str:
    return extract_section_by_header(content, "Abstract")


def extract_claims_text(content: str) -> str:
    return extract_section_by_header(content, "Claims")


def extract_description(content: str) -> str:
    return extract_section_by_header(content, "Description")


def format_date(date_str: str) -> str:
    if date_str and len(date_str) == 8:
        return date_str[:4] + "-" + date_str[4:6] + "-" + date_str[6:]
    return date_str


def normalize_id(text: str) -> str:
    return text.strip().lower().replace(" ", "_")


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
    metadatas: list[Metadata] = []
    for patent_id in missing:
        title = patent_id_to_title[patent_id].strip()
        abstract = patent_id_to_abstract[patent_id].strip()
        combined_text = title + "\n\n" + abstract if abstract else title
        if not combined_text:
            raise ValueError("Empty title and abstract for patent_id " + patent_id)
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
    patent_id_to_claims: dict[str, str]
    patent_id_to_description: dict[str, str]


def load_patents(config: PatentTechnicalTasksetConfig) -> PatentIndex:
    patent_id_to_title: dict[str, str] = {}
    patent_id_to_content: dict[str, str] = {}
    patent_id_to_metadata: dict[str, dict[str, str]] = {}
    patent_id_to_abstract: dict[str, str] = {}
    patent_id_to_claims: dict[str, str] = {}
    patent_id_to_description: dict[str, str] = {}
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
        patent_id_to_claims[patent_id] = extract_claims_text(content)
        patent_id_to_description[patent_id] = extract_description(content)

    client = chromadb.PersistentClient(path=config.chroma_db_dir)
    collection = client.get_or_create_collection(
        name="patent_titles_abstracts_v3",
        configuration={
            "embedding_function": embedding_functions.OpenAIEmbeddingFunction(
                model_name=config.embed_model,
                api_base=config.embed_base_url,
                api_key=os.getenv(config.embed_api_key_var, "EMPTY"),
            ),
        },
    )
    init_chroma(collection, patent_id_to_title, patent_id_to_abstract)
    return PatentIndex(
        collection=collection,
        patent_id_to_title=patent_id_to_title,
        patent_id_to_content=patent_id_to_content,
        patent_id_to_metadata=patent_id_to_metadata,
        patent_id_to_abstract=patent_id_to_abstract,
        patent_id_to_claims=patent_id_to_claims,
        patent_id_to_description=patent_id_to_description,
    )


def normalize_info(value: object) -> vf.ConfigData:
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    if isinstance(value, str):
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise TypeError("Parsed info must be a dict.")
        return {str(key): item for key, item in parsed.items()}
    return {}


def ground_truth_from_info(info: vf.ConfigData) -> vf.ConfigData:
    ground_truth = info.get("ground_truth", {})
    if not isinstance(ground_truth, dict):
        raise TypeError("ground_truth must be a dict.")
    return {str(key): item for key, item in ground_truth.items()}


def string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


class PatentTechnicalTaskset(vf.Taskset[PatentTechnicalTasksetConfig]):
    def __init__(self, config: PatentTechnicalTasksetConfig | None = None) -> None:
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
                "info": normalize_info(row.get("info", {})),
            }

    def load_toolset(self) -> vf.Toolset:
        ensure_keys([self.config.embed_api_key_var])
        patents = load_patents(self.config)
        chroma_semaphore = asyncio.Semaphore(100)

        async def search_patents(query: str) -> list[dict[str, str]]:
            """Search for relevant patents using title and abstract embedding similarity."""
            async with chroma_semaphore:
                results = await asyncio.to_thread(
                    patents.collection.query, query_texts=[query], n_results=10
                )
            if not results or not results["metadatas"]:
                raise ValueError("No results found for query: " + query)
            hits: list[dict[str, str]] = []
            for index in range(len(results["ids"][0])):
                metadata = results["metadatas"][0][index]
                patent_id = results["ids"][0][index]
                title = metadata["title"]
                abstract_value = metadata.get("abstract", "")
                assert isinstance(patent_id, str)
                assert isinstance(title, str)
                assert isinstance(abstract_value, str) or abstract_value is None
                hits.append(
                    {
                        "patent_id": patent_id,
                        "title": title,
                        "abstract": abstract_value or "",
                    }
                )
            return hits

        async def get_metadata(patent_id: str) -> dict[str, str]:
            """Get patent metadata."""
            if patent_id not in patents.patent_id_to_metadata:
                raise ValueError("Patent not found: " + patent_id)
            metadata = patents.patent_id_to_metadata[patent_id]
            return {
                "title": patents.patent_id_to_title.get(patent_id, ""),
                "filing_date": format_date(metadata.get("filing_date", "")),
                "grant_date": format_date(metadata.get("grant_date", "")),
                "claim_count": metadata.get("claim_count", "0"),
            }

        async def get_abstract(patent_id: str) -> str:
            """Get the full abstract text for a patent."""
            if patent_id not in patents.patent_id_to_abstract:
                raise ValueError("Patent not found: " + patent_id)
            return patents.patent_id_to_abstract[patent_id]

        async def get_claims_text(patent_id: str) -> str:
            """Get the full claims section text for a patent."""
            if patent_id not in patents.patent_id_to_claims:
                raise ValueError("Patent not found: " + patent_id)
            return patents.patent_id_to_claims[patent_id]

        async def get_description(patent_id: str) -> str:
            """Get the description section for a patent."""
            if patent_id not in patents.patent_id_to_description:
                raise ValueError("Patent not found: " + patent_id)
            description = patents.patent_id_to_description[patent_id]
            if len(description) > 15000:
                return description[:15000] + "\n\n[TRUNCATED - description continues...]"
            return description

        async def get_full_content(patent_id: str) -> str:
            """Get the full content of a patent including all sections."""
            if patent_id not in patents.patent_id_to_content:
                raise ValueError("Patent not found: " + patent_id)
            content = patents.patent_id_to_content[patent_id]
            if len(content) > 20000:
                return content[:20000] + "\n\n[TRUNCATED - content continues...]"
            return content

        async def compare_patents(patent_id_1: str, patent_id_2: str) -> dict[str, dict[str, str]]:
            """Get side-by-side comparison data for two patents."""
            if patent_id_1 not in patents.patent_id_to_content:
                raise ValueError("Patent not found: " + patent_id_1)
            if patent_id_2 not in patents.patent_id_to_content:
                raise ValueError("Patent not found: " + patent_id_2)
            meta1 = patents.patent_id_to_metadata.get(patent_id_1, {})
            meta2 = patents.patent_id_to_metadata.get(patent_id_2, {})
            return {
                "patent_1": {
                    "id": patent_id_1,
                    "title": patents.patent_id_to_title.get(patent_id_1, ""),
                    "abstract": patents.patent_id_to_abstract.get(patent_id_1, ""),
                    "filing_date": format_date(meta1.get("filing_date", "")),
                    "claim_count": meta1.get("claim_count", "0"),
                },
                "patent_2": {
                    "id": patent_id_2,
                    "title": patents.patent_id_to_title.get(patent_id_2, ""),
                    "abstract": patents.patent_id_to_abstract.get(patent_id_2, ""),
                    "filing_date": format_date(meta2.get("filing_date", "")),
                    "claim_count": meta2.get("claim_count", "0"),
                },
            }

        async def view_sections(patent_id: str) -> list[dict[str, str]]:
            """View the sections of a patent."""
            content = patents.patent_id_to_content[patent_id]
            sections: list[dict[str, str]] = []
            for line in content.split("\n"):
                if line.startswith("#"):
                    section_name = line.lstrip("#").strip()
                    sections.append(
                        {
                            "section_id": patent_id + ":" + normalize_id(section_name),
                            "section_name": section_name,
                        }
                    )
            if not sections:
                sections.append({"section_id": patent_id + ":full", "section_name": "Full Patent"})
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
                raise ValueError("Section not found: " + section_id)
            return "\n".join(lines[section_start : section_end or len(lines)])

        return vf.Toolset(
            tools=[
                search_patents,
                get_metadata,
                get_abstract,
                get_claims_text,
                get_description,
                get_full_content,
                compare_patents,
                view_sections,
                read_section,
            ],
        )

    @vf.reward(weight=1.0)
    async def judge_reward(self, task: vf.Task, state: vf.State) -> float:
        info = normalize_info(task.get("info", {}))
        try:
            ground_truth = ground_truth_from_info(info)
        except TypeError:
            logger.error("[JUDGE] Invalid ground_truth in task info")
            return 0.0
        key_points = string_list(ground_truth.get("key_points", []))
        if not key_points:
            logger.error("[JUDGE] No key_points found in task info")
            return 0.0
        key_points_checklist = "".join(
            f"{index}. {key_point}\n" for index, key_point in enumerate(key_points, start=1)
        )
        source_quotes = string_list(ground_truth.get("source_quotes", []))
        source_quotes_text = (
            "\n".join("- " + str(source_quote) for source_quote in source_quotes)
            if source_quotes
            else "N/A"
        )
        completion = state["completion"]
        last_assistant = next(
            (msg for msg in reversed(completion) if msg.get("role") == "assistant"),
            None,
        )
        response_text = str(last_assistant["content"]) if last_assistant else ""
        formatted_prompt = JUDGE_PROMPT.format(
            question=task["question"],
            answer=task["answer"],
            source_quotes=source_quotes_text,
            response=response_text,
            key_points_checklist=key_points_checklist,
        )
        endpoint = state.get_endpoint_config(api="chat")
        judge_client = AsyncOpenAI(api_key=endpoint["api_key"], base_url=endpoint["api_base"])
        try:
            judge_response = await judge_client.chat.completions.create(
                model=endpoint["model"],
                messages=[{"role": "user", "content": formatted_prompt}],
                temperature=0.0,
                max_tokens=2048,
            )
            response_text_judge = (judge_response.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.error("[JUDGE] LLM call failed: %s", exc)
            return 0.0
        json_start = response_text_judge.find("{")
        json_end = response_text_judge.rfind("}") + 1
        if json_start == -1 or json_end == 0:
            logger.error("[JUDGE] No JSON block found in judge response")
            return 0.0
        try:
            evaluation = json.loads(response_text_judge[json_start:json_end])
        except json.JSONDecodeError as exc:
            logger.error("[JUDGE] JSON parse failed: %s", exc)
            return 0.0
        key_point_results = evaluation.get("key_points", [])
        covered_count = sum(1 for key_point in key_point_results if key_point.get("covered", False))
        score = covered_count / len(key_points)
        if evaluation.get("hallucination", False):
            score *= HALLUCINATION_MULTIPLIER
        if evaluation.get("factual_error", False):
            score *= FACTUAL_ERROR_MULTIPLIER
        return max(0.0, min(1.0, score))


def load_taskset(config: PatentTechnicalTasksetConfig) -> vf.Taskset:
    return PatentTechnicalTaskset(config=config)


def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(taskset=vf.load_taskset(config=config.taskset))
