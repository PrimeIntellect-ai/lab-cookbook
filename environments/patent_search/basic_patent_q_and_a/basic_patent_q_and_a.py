import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING

import verifiers.v1 as vf
from datasets import load_dataset
from dotenv import load_dotenv
from openai import AsyncOpenAI
from verifiers import ensure_keys

if TYPE_CHECKING:
    from chromadb.api.models.Collection import Collection

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)


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

# Module-scope semaphore: chromadb client is sync; cap concurrent thread-offloaded
# queries so a burst of rollouts can't exhaust the default executor.
_chroma_semaphore: asyncio.Semaphore | None = None


class BasicPatentTasksetConfig(vf.TasksetConfig):
    embed_model: str = "text-embedding-3-small"
    embed_base_url: str = "https://api.openai.com/v1"
    embed_api_key_var: str = "OPENAI_API_KEY"
    corpus_dataset: str = "jessicafeiyali/qualcomm-patents"
    corpus_file: str = "patents_formatted.json"
    qa_file: str = "patent_qa.jsonl"
    chroma_db_dir: str = CHROMA_DB_DIR


def chroma_semaphore() -> asyncio.Semaphore:
    global _chroma_semaphore
    if _chroma_semaphore is None:
        _chroma_semaphore = asyncio.Semaphore(100)
    return _chroma_semaphore


def normalize_id(text: str) -> str:
    return text.strip().lower().replace(" ", "_")


class PatentCorpus:
    def __init__(self, config: BasicPatentTasksetConfig):
        self.config = config
        self.patent_id_to_title: dict[str, str] = {}
        self.patent_id_to_content: dict[str, str] = {}
        self.patent_id_to_metadata: dict[str, dict[str, str]] = {}
        # chromadb.Collection has no public type stub; narrowed lazily.
        self.collection: "Collection | None" = None
        self.loaded = False

    def load(self) -> None:
        if self.loaded:
            return
        corpus = load_dataset(
            self.config.corpus_dataset,
            data_files=self.config.corpus_file,
            split="train",
        )
        for row in corpus:
            patent_id = str(row["id"])
            self.patent_id_to_title[patent_id] = str(row["title"])
            self.patent_id_to_content[patent_id] = str(row["content"])
            metadata = row.get("metadata") or {}
            self.patent_id_to_metadata[patent_id] = {
                str(key): str(value) for key, value in metadata.items()
            }
        self.loaded = True

    def get_collection(self) -> "Collection":
        self.load()
        if self.collection is None:
            import chromadb
            from chromadb.utils import embedding_functions

            openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                model_name=self.config.embed_model,
                api_base=self.config.embed_base_url,
                api_key=os.environ[self.config.embed_api_key_var],
            )
            client = chromadb.PersistentClient(path=self.config.chroma_db_dir)
            self.collection = client.get_or_create_collection(
                name="patent_titles",
                embedding_function=openai_ef,
            )
            self.init_chroma()
        return self.collection

    def init_chroma(self) -> None:
        assert self.collection is not None
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
            if not title:
                raise ValueError(f"Empty title for patent_id {patent_id}")
            documents.append(title)
            metadatas.append({"title": title})
        batch_size = 100
        for index in range(0, len(missing), batch_size):
            self.collection.upsert(
                ids=missing[index : index + batch_size],
                documents=documents[index : index + batch_size],
                metadatas=metadatas[index : index + batch_size],
            )

    async def search_patents(self, query: str) -> list[dict[str, str]]:
        """Search for relevant patents using title embedding similarity.

        Args:
            query: The query to search for.
        """
        collection = self.get_collection()
        async with chroma_semaphore():
            results = await asyncio.to_thread(collection.query, query_texts=[query], n_results=10)
        if not results or not results["metadatas"]:
            raise ValueError(f"No results found for query: {query}")
        output: list[dict[str, str]] = []
        for index in range(len(results["ids"][0])):
            output.append(
                {
                    "patent_id": results["ids"][0][index],
                    "title": results["metadatas"][0][index]["title"],
                }
            )
        return output

    async def get_metadata(self, patent_id: str) -> dict[str, str]:
        """Get patent metadata.

        Args:
            patent_id: The ID of the patent.
        """
        self.load()
        if patent_id not in self.patent_id_to_metadata:
            raise ValueError(f"Patent not found: {patent_id}")
        result = dict(self.patent_id_to_metadata[patent_id])
        result["title"] = self.patent_id_to_title.get(patent_id, "")
        return result

    async def view_sections(self, patent_id: str) -> list[dict[str, str]]:
        """View the sections of a patent.

        Args:
            patent_id: The ID of the patent to view.
        """
        self.load()
        content = self.patent_id_to_content[patent_id]
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
            sections.append(
                {
                    "section_id": f"{patent_id}:full",
                    "section_name": "Full Patent",
                }
            )
        return sections

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


def source(config: BasicPatentTasksetConfig):
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


def load_environment(config: vf.EnvConfig) -> vf.Env:
    cfg = BasicPatentTasksetConfig(config.taskset)
    ensure_keys([cfg.embed_api_key_var])
    corpus = PatentCorpus(cfg)
    toolset = vf.Toolset(
        tools=[
            corpus.search_patents,
            corpus.get_metadata,
            corpus.view_sections,
            corpus.read_section,
        ],
    )
    taskset = vf.Taskset(
        source=lambda: source(cfg),
        system_prompt=SYSTEM_PROMPT,
        toolsets=[toolset],
        rewards=[judge_reward],
        config=cfg,
    )
    return vf.Env(
        taskset=taskset,
        harness=vf.Harness(config=config.harness),
    )
