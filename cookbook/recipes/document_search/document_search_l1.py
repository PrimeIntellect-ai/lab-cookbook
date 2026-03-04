"""Document Search — Level 1 (metadata retrieval).

Agents answer metadata questions about documents using two tools:
  - search_docs(query) → [{id, title, score}]  — TF-IDF search over corpus
  - get_metadata(doc_id) → {title, field, ...} — fetch stored document fields

Dataset: synthetic corpus of 100 company profiles + 500 Q&A pairs.
Reward: binary exact match on field values.

Difficulty calibration: starting reward 0.15-0.35 with a 1B model.
L1 typically saturates to ~1.0 within 15-20 training steps.

The recipe mirrors the patent search blog post methodology:
  L1 → L2 → L3 is a curriculum (simple retrieval → reasoning → open analysis).
"""
import random
import verifiers as vf
from datasets import Dataset

from lab_cookbook.skills.lab.dataset_builder import DatasetBuilder
from lab_cookbook.skills.lab.semantic_search import TFIDFSearchIndex
from lab_cookbook.skills.verifiers.exact_match import exact_match_reward as _base_exact_match

SYSTEM_PROMPT = """You are a research assistant with access to a document corpus.
Use the search_docs and get_metadata tools to find information.

Workflow:
1. Call search_docs(query) to find relevant documents.
2. Call get_metadata(doc_id) on promising results to read their fields.
3. Once you have the answer, output it on its own line with no extra text.

Be concise. Your final answer must be exactly the requested value.
"""

# ---------------------------------------------------------------------------
# Corpus generation
# ---------------------------------------------------------------------------

_INDUSTRIES = [
    "Software", "Hardware", "Biotech", "Finance", "Retail",
    "Healthcare", "Energy", "Manufacturing", "Media", "Logistics",
]

_COMPANY_ADJECTIVES = [
    "Acme", "Apex", "Atlas", "Aurora", "Axiom", "Azure", "Blue",
    "Bright", "Cascade", "Cedar", "Core", "Crest", "Delta", "Echo",
    "Edge", "Ember", "Epic", "Falcon", "Flash", "Forge", "Frontier",
    "Galaxy", "Gem", "Global", "Gold", "Green", "Grid", "Harbor",
    "Horizon", "Icon", "Ignite", "Impact", "Iris", "Jade", "Key",
    "Kinetic", "Lance", "Lark", "Light", "Link", "Logic", "Lumen",
    "Matrix", "Maxim", "Merge", "Metro", "Mint", "Nano", "Nexus",
    "Noble", "Nord", "Nova", "Omni", "Open", "Orbit", "Pacific",
    "Peak", "Pinnacle", "Pixel", "Prime", "Prism", "Pulse", "Pyro",
    "Quartz", "Quest", "Rapid", "Relay", "Revel", "Ridge", "Rise",
    "Rocket", "Root", "Scale", "Shift", "Signal", "Silver", "Solar",
    "Solid", "Sonic", "Spark", "Sphere", "Sprint", "Star", "Steel",
    "Storm", "Stream", "Summit", "Swift", "Sync", "Synergy", "Terra",
    "Titan", "Torch", "Trek", "Trend", "Tri", "Turbo", "Ultra",
    "Union", "Unity", "Venture", "Verge", "Vertex", "Vibe", "Vista",
]

_SUFFIXES = ["Corp", "Inc", "Labs", "Tech", "Systems", "Group", "Solutions", "Works"]


def _build_corpus(n: int = 100, seed: int = 42) -> dict[str, dict]:
    """Build a synthetic corpus of company profiles.

    Returns:
        Dict mapping doc_id -> company profile dict.
    """
    rng = random.Random(seed)
    corpus: dict[str, dict] = {}
    used_names: set[str] = set()

    for i in range(n):
        while True:
            adj = rng.choice(_COMPANY_ADJECTIVES)
            sfx = rng.choice(_SUFFIXES)
            name = f"{adj} {sfx}"
            if name not in used_names:
                used_names.add(name)
                break

        doc_id = f"DOC{i + 1:04d}"
        corpus[doc_id] = {
            "id": doc_id,
            "name": name,
            "founded": rng.randint(1970, 2020),
            "industry": rng.choice(_INDUSTRIES),
            "employees": rng.choice([50, 100, 250, 500, 1000, 2500, 5000, 10000]),
            "revenue_millions": rng.choice([1, 5, 10, 25, 50, 100, 250, 500, 1000]),
            "headquarters": rng.choice([
                "New York", "San Francisco", "Austin", "Seattle",
                "Boston", "Chicago", "Denver", "Miami", "Portland",
            ]),
            "public": rng.choice([True, False]),
        }
    return corpus


def _build_qa_pairs(
    corpus: dict[str, dict],
    n: int = 500,
    seed: int = 42,
) -> Dataset:
    """Generate Q&A pairs over the corpus."""
    rng = random.Random(seed)
    builder = DatasetBuilder()
    docs = list(corpus.values())

    templates = [
        # (question_fn, answer_fn)
        (
            lambda d: f"What year was {d['name']} founded?",
            lambda d: str(d["founded"]),
        ),
        (
            lambda d: f"What industry is {d['name']} in?",
            lambda d: d["industry"],
        ),
        (
            lambda d: f"How many employees does {d['name']} have?",
            lambda d: str(d["employees"]),
        ),
        (
            lambda d: f"What is the revenue of {d['name']} in millions?",
            lambda d: str(d["revenue_millions"]),
        ),
        (
            lambda d: f"Where is {d['name']} headquartered?",
            lambda d: d["headquarters"],
        ),
        (
            lambda d: f"Is {d['name']} a public company?",
            lambda d: str(d["public"]),
        ),
    ]

    for _ in range(n):
        doc = rng.choice(docs)
        q_fn, a_fn = rng.choice(templates)
        question = q_fn(doc)
        answer = a_fn(doc)
        builder.add(question, answer, info={"doc_id": doc["id"]})

    return builder.build()


# ---------------------------------------------------------------------------
# Module-level corpus and search index (built once at import time)
# ---------------------------------------------------------------------------

_CORPUS: dict[str, dict] = {}
_SEARCH_INDEX = TFIDFSearchIndex()


def _ensure_corpus_loaded(seed: int = 42) -> None:
    global _CORPUS
    if _CORPUS:
        return
    _CORPUS = _build_corpus(seed=seed)
    docs_for_index = [
        {
            "id": doc["id"],
            "text": f"{doc['name']} {doc['industry']} {doc['headquarters']}",
        }
        for doc in _CORPUS.values()
    ]
    _SEARCH_INDEX.add_documents(docs_for_index)


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

def search_docs(query: str) -> str:
    """Search the document corpus by keyword or description.

    Args:
        query: Search query, e.g. "software company New York"

    Returns:
        JSON-style list of up to 5 results: [{id, title, score}]
    """
    _ensure_corpus_loaded()
    results = _SEARCH_INDEX.search(query, k=5)
    out = [
        {
            "id": r["id"],
            "title": _CORPUS[r["id"]]["name"],
            "score": r["score"],
        }
        for r in results
    ]
    return str(out)


def get_metadata(doc_id: str) -> str:
    """Get full metadata for a document by its ID.

    Args:
        doc_id: Document identifier, e.g. "DOC0001"

    Returns:
        Full metadata dict as string, or error message if not found.
    """
    _ensure_corpus_loaded()
    doc = _CORPUS.get(doc_id)
    if doc is None:
        return f"Document '{doc_id}' not found. Use search_docs to find valid IDs."
    return str(doc)


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

async def exact_match_reward(completion: str, answer: str, **kwargs) -> float:
    """Return 1.0 if last non-empty line matches the expected answer."""
    lines = [ln.strip() for ln in completion.split("\n") if ln.strip()]
    if not lines:
        return 0.0
    return await _base_exact_match(completion=lines[-1], answer=answer)


# ---------------------------------------------------------------------------
# load_environment
# ---------------------------------------------------------------------------

def load_environment(
    num_examples: int = -1,
    seed: int = 42,
) -> vf.Environment:
    """Load Document Search L1 environment.

    Args:
        num_examples: Number of Q&A pairs to use (-1 = all 500).
        seed: Random seed.

    Returns:
        ToolEnv with search_docs + get_metadata tools.
    """
    _ensure_corpus_loaded(seed=seed)
    dataset = _build_qa_pairs(_CORPUS, seed=seed)
    if num_examples != -1:
        dataset = dataset.select(range(min(num_examples, len(dataset))))

    rubric = vf.Rubric(funcs=[exact_match_reward], weights=[1.0])

    return vf.ToolEnv(
        dataset=dataset,
        system_prompt=SYSTEM_PROMPT,
        tools=[search_docs, get_metadata],
        rubric=rubric,
    )


if __name__ == "__main__":
    _ensure_corpus_loaded()
    print(f"Corpus size: {len(_CORPUS)} documents")
    env = load_environment(num_examples=10)
    print(f"Dataset size: {len(env.dataset)}")
    for row in env.dataset.select(range(3)):
        print(f"  Q: {row['question']}")
        print(f"  A: {row['answer']}")
