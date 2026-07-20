# Fantasy Football Co-Pilot

A Retrieval-Augmented Generation (RAG) assistant that answers fantasy football
**draft** and **start/sit** questions by grounding a large language model in real
NFL statistics from the completed 2025 season.

**Author:** Ishan Makkar (solo project, approved by the course instructor)
**Course:** IE 7374 — Generative AI
**Milestone 3:** Data Pipeline

---

## What this project does

Ask a natural-language question like:

> *"In Week 5 of the 2025 season, should I have started Justin Jefferson or CeeDee Lamb?"*

The system retrieves the relevant player-week statistics from a vector store and asks
an LLM to answer **using only that retrieved context**, citing the stats it relied on.
This grounds the answer in verifiable data instead of the model's (possibly stale or
hallucinated) memory.

## Research question (RQ1)

> Does grounding an LLM's fantasy football recommendations in a retrieval-augmented
> pipeline over structured 2025-season statistics improve factual faithfulness and
> outcome accuracy compared to prompting the same LLM directly, without retrieval?

To answer this, the repo implements **two systems** — a RAG pipeline and a no-retrieval
baseline — and an evaluation harness that compares them.

## Architecture

```
nfl_data_py (2025 stats)
        │  src/data_loader.py
        ▼
  weekly stat rows
        │  src/preprocess.py   (one passage per player per week)
        ▼
 natural-language passages
        │  src/embed_store.py  (MiniLM embeddings)
        ▼
   ChromaDB vector store
        │  retrieve top-k
        ▼
   ┌────────────────────┬─────────────────────┐
   │  src/rag_pipeline  │   src/baseline      │
   │  (retrieval + LLM) │   (LLM, no context) │
   └─────────┬──────────┴──────────┬──────────┘
             ▼                      ▼
        eval/evaluate.py  (RQ1 comparison)
```

## Repository structure

```
fantasy-football-copilot/
├── README.md                  # this file
├── requirements.txt           # dependencies
├── .gitignore
├── src/
│   ├── config.py              # central configuration
│   ├── data_loader.py         # nfl_data_py ingestion (cached)
│   ├── preprocess.py          # stat rows -> text passages
│   ├── embed_store.py         # embeddings + ChromaDB
│   ├── build_index.py         # one-command end-to-end pipeline
│   ├── rag_pipeline.py        # RAG system (RQ1 treatment)
│   ├── baseline.py            # no-retrieval baseline (RQ1 control)
│   └── model_runner.py        # single-command demo entry point
├── eval/
│   ├── benchmark_queries.json # fixed 20-query benchmark
│   └── evaluate.py            # RQ1 evaluation harness
├── docs/
│   └── methods.md             # research, literature review, benchmarking
└── data/
    └── README.md              # data source and passage design
```

## Setup

Requires Python 3.9+.

```bash
# 1. Clone and enter the repo
git clone <your-repo-url>
cd fantasy-football-copilot

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Only needed for the generator LLM) set your OpenAI key
export OPENAI_API_KEY="sk-..."   # Windows: set OPENAI_API_KEY=sk-...
```

The embedding step uses a **local** model by default (no API key needed), so the
entire data pipeline and vector store can be built for free and offline. An OpenAI
key is only required to run the generator LLM in `rag_pipeline` / `baseline` /
`model_runner` / full `evaluate`.

## Usage

### 1. Build the data pipeline (ingest → passages → embed → store)

```bash
python -m src.build_index
```

This downloads the 2025 weekly stats (cached to `data/raw/`), builds passages
(`data/processed/passages.jsonl`), and populates the ChromaDB vector store
(`data/chroma/`).

### 2. Inspect ground truth without any API key

```bash
python -m eval.evaluate --ground-truth
```

Prints, for each start/sit benchmark question, which player actually scored more PPR
points that week — computed directly from the data.

### 3. Run the end-to-end demo (needs OPENAI_API_KEY)

```bash
python -m src.model_runner
```

Runs both the RAG pipeline and the baseline on the first few benchmark queries and
saves sample generations to `outputs/`.

### 4. Run the full RQ1 evaluation (needs OPENAI_API_KEY)

```bash
python -m eval.evaluate
```

Produces `outputs/evaluation_results.csv` with per-query RAG vs. baseline answers,
automatic outcome-accuracy scoring on start/sit questions, latency, and blank columns
for human faithfulness/helpfulness annotation.

## How outcomes are scored (and why it is honest)

Start/sit questions are scored **objectively**: the ground-truth "correct" answer is
whichever player actually scored more PPR fantasy points that week, computed directly
from `fantasy_points_ppr` in the 2025 data — never hand-labeled. A system is "correct"
if its recommendation names that player. Draft questions have no single correct
outcome and are evaluated qualitatively (faithfulness + helpfulness).

## Method choices (short version)

| Component | Choice | Why |
|-----------|--------|-----|
| Data | `nfl_data_py` (single source) | Clean, maintained, no scraping |
| Embeddings | `all-MiniLM-L6-v2` (local) | Free, fast, reproducible; OpenAI optional |
| Vector store | ChromaDB | Lightweight, persistent, local |
| Generator | `gpt-4o-mini` (pretrained, frozen) | Strong grounded-QA at low cost |

Full rationale, literature review, and benchmarking are in
[`docs/methods.md`](docs/methods.md).

## Roadmap

- **Milestone 3 (this):** data pipeline, RAG + baseline implementation, evaluation
  harness, repository.
- **Milestone 4:** package as a single-command runnable pipeline with saved sample
  outputs.
- **Milestone 5:** full RQ1 evaluation, Streamlit UI with a transparency sidebar,
  technical report, and presentation. Stretch goals: news corpus (RQ2), chunking
  study (RQ3).

## Limitations

- The generator requires an OpenAI API key; a local Llama-3 fallback is planned.
- The current corpus is stats-only; a news corpus is a documented stretch goal.
- Faithfulness and helpfulness are human-annotated at this stage.
