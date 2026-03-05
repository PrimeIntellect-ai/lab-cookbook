import asyncio
import os
import re
from typing import cast

from dotenv import load_dotenv
load_dotenv()

import chromadb
from chromadb.api.types import Embeddable, EmbeddingFunction
from chromadb.utils import embedding_functions
from datasets import load_dataset
from openai import AsyncOpenAI

import verifiers as vf
from verifiers.rubrics.judge_rubric import JudgeRubric

CHROMA_DB_DIR = ".chroma_db_patents"
_chroma_semaphore: asyncio.Semaphore | None = None


def _get_chroma_semaphore() -> asyncio.Semaphore:
    global _chroma_semaphore
    if _chroma_semaphore is None:
        _chroma_semaphore = asyncio.Semaphore(100)
    return _chroma_semaphore


def extract_abstract(content: str) -> str:
    abstract_match = re.search(r'## Abstract\s*\n\n(.*?)(?=\n## |\Z)', content, re.DOTALL)
    if abstract_match:
        return abstract_match.group(1).strip()
    return ""


def format_date(date_str: str) -> str:
    if date_str and len(date_str) == 8:
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    return date_str


def load_environment(
    max_turns: int = 10,
    judge_model: str = "gpt-4.1-mini",
    judge_base_url: str = "https://api.openai.com/v1",
    judge_api_key_var: str = "OPENAI_API_KEY",
    embed_model: str = "text-embedding-3-small",
    embed_base_url: str = "https://api.openai.com/v1",
    embed_api_key_var: str = "OPENAI_API_KEY",
    corpus_dataset: str = "jessicafeiyali/qualcomm-patents",
    corpus_file: str = "patents_formatted.json",
    qa_file: str = "patent_qa_level2.jsonl",
    chroma_db_dir: str = CHROMA_DB_DIR,
) -> vf.Environment:
    corpus = load_dataset(corpus_dataset, data_files=corpus_file, split="train")
    patent_id_to_title: dict[str, str] = {}
    patent_id_to_content: dict[str, str] = {}
    patent_id_to_metadata: dict[str, dict] = {}
    patent_id_to_abstract: dict[str, str] = {}
    for row in corpus:
        row = cast(dict, row)
        pid = row["id"]
        title = row["title"]
        content = row["content"]
        patent_id_to_title[pid] = title
        patent_id_to_content[pid] = content
        patent_id_to_metadata[pid] = row.get("metadata", {})
        patent_id_to_abstract[pid] = extract_abstract(content)

    _chroma_state: dict = {"collection": None}

    def _get_collection():
        if _chroma_state["collection"] is None:
            openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                model_name=embed_model,
                api_base=embed_base_url,
                api_key=os.getenv(embed_api_key_var, "EMPTY"),
            )
            client = chromadb.PersistentClient(path=chroma_db_dir)
            _chroma_state["collection"] = client.get_or_create_collection(
                name="patent_titles_abstracts",
                embedding_function=cast(EmbeddingFunction[Embeddable], openai_ef),
            )
            _init_chroma(_chroma_state["collection"])
        return _chroma_state["collection"]

    def _init_chroma(collection) -> None:
        all_ids = list(patent_id_to_title.keys())
        existing: set[str] = set()
        for i in range(0, len(all_ids), 500):
            batch = all_ids[i : i + 500]
            got = collection.get(ids=batch)
            existing.update(got.get("ids", []))
        missing = [pid for pid in all_ids if pid not in existing]
        if missing:
            documents = []
            metadatas = []
            for pid in missing:
                title = str(patent_id_to_title[pid]).strip()
                abstract = str(patent_id_to_abstract[pid]).strip()
                combined_text = f"{title}\n\n{abstract}" if abstract else title
                if not combined_text:
                    raise ValueError(f"Empty title and abstract for patent_id {pid}")
                documents.append(combined_text)
                metadatas.append({"title": title, "abstract": abstract})
            bs = 100
            for i in range(0, len(missing), bs):
                collection.upsert(
                    ids=missing[i : i + bs],
                    documents=documents[i : i + bs],
                    metadatas=metadatas[i : i + bs],
                )

    def normalize_id(text: str) -> str:
        return text.strip().lower().replace(" ", "_")

    async def search_patents(query: str) -> list[dict]:
        """Search for relevant patents using title and abstract embedding similarity.

        args:
            query (str): The query to search for.

        returns:
            list[dict]: A list of dicts with patent_id, title, and abstract.
        """
        collection = _get_collection()
        async with _get_chroma_semaphore():
            results = await asyncio.to_thread(
                collection.query, query_texts=[query], n_results=10
            )
        if not results:
            raise ValueError(f"No results found for query: {query}")
        if not results["metadatas"]:
            raise ValueError(f"No results metadata found for query: {query}")
        output = []
        for i in range(len(results["ids"][0])):
            output.append(
                {
                    "patent_id": results["ids"][0][i],
                    "title": results["metadatas"][0][i]["title"],
                    "abstract": results["metadatas"][0][i].get("abstract", ""),
                }
            )
        return output

    async def get_metadata(patent_id: str) -> dict:
        """Get patent metadata.

        args:
            patent_id (str): The ID of the patent.

        returns:
            dict: Metadata with title, filing_date, grant_date, claim_count.
        """
        if patent_id not in patent_id_to_metadata:
            raise ValueError(f"Patent not found: {patent_id}")
        metadata = patent_id_to_metadata[patent_id]
        return {
            "title": patent_id_to_title.get(patent_id, ""),
            "filing_date": format_date(metadata.get("filing_date", "")),
            "grant_date": format_date(metadata.get("grant_date", "")),
            "claim_count": metadata.get("claim_count", 0),
        }

    async def get_abstract(patent_id: str) -> str:
        """Get the full abstract text for a patent.

        args:
            patent_id (str): The ID of the patent.

        returns:
            str: The full abstract text.
        """
        if patent_id not in patent_id_to_abstract:
            raise ValueError(f"Patent not found: {patent_id}")
        return patent_id_to_abstract[patent_id]

    async def view_sections(patent_id: str) -> list[dict]:
        """View the sections of a patent.

        args:
            patent_id (str): The ID of the patent to view.

        returns:
            list[dict]: A list of dicts with section_id and section_name.
        """
        content = patent_id_to_content[patent_id]
        sections = []
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("#"):
                section_name = line.lstrip("#").strip()
                section_id = f"{patent_id}:{normalize_id(section_name)}"
                sections.append(
                    {
                        "section_id": section_id,
                        "section_name": section_name,
                        "start_line": i,
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
            {"section_id": s["section_id"], "section_name": s["section_name"]}
            for s in sections
        ]

    async def read_section(section_id: str) -> str:
        """Read a section of a patent.

        args:
            section_id (str): The ID of the section to read.

        returns:
            str: The content of the section.
        """
        if ":" not in section_id:
            raise ValueError(
                "Invalid section_id format. Expected: patent_id:section_name"
            )
        patent_id, section_name_id = section_id.split(":", 1)

        content = patent_id_to_content[patent_id]
        lines = content.split("\n")

        if section_name_id == "full":
            return content

        section_start = None
        section_end = None

        for i, line in enumerate(lines):
            if line.startswith("#"):
                current_section = normalize_id(line.lstrip("#").strip())
                if current_section == section_name_id and section_start is None:
                    section_start = i
                elif section_start is not None and section_end is None:
                    section_end = i
                    break

        if section_start is not None:
            if section_end is None:
                section_end = len(lines)
            return "\n".join(lines[section_start:section_end])
        else:
            raise ValueError(f"Section not found: {section_id}")

    tools = [
        search_patents,
        get_metadata,
        get_abstract,
        view_sections,
        read_section,
    ]
    parser = vf.Parser()
    dataset = load_dataset(corpus_dataset, data_files=qa_file, split="train")

    JUDGE_PROMPT = """Given a ground truth answer \
    and a response, determine if the response is both correct and coherent.

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
    judge_client = AsyncOpenAI(
        base_url=judge_base_url, api_key=os.getenv(judge_api_key_var)
    )
    judge_rubric = JudgeRubric(
        judge_client=judge_client,
        judge_model=judge_model,
        parser=parser,
        judge_prompt=JUDGE_PROMPT,
    )

    async def judge_reward_func(judge, prompt, completion, answer, state) -> float:
        judge_response = await judge(prompt, completion, answer, state)
        if "yes" in judge_response.lower():
            return 1.0
        else:
            return 0.0

    system_prompt = "Use the provided patent search tools to help answer questions about patents."
    judge_rubric.add_reward_func(judge_reward_func, weight=1.0)
    vf_env = vf.ToolEnv(
        dataset=dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=judge_rubric,
        tools=tools,
        max_turns=max_turns,
    )
    return vf_env