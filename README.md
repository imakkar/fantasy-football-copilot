# Fantasy Football Co-Pilot

A Retrieval-Augmented Generation (RAG) assistant that answers fantasy football
**draft** and **start/sit** questions by grounding a large language model in real
NFL statistics from the completed 2024 season.

**Author:** Ishan Makkar (solo project, approved by the course instructor)
**Course:** IE 7374 — Generative AI
**Milestone 3:** Data Pipeline

---

## What this project does

Ask a natural-language question like:

> *"In Week 5 of the 2024 season, should I have started Justin Jefferson or CeeDee Lamb?"*

The system retrieves the relevant player-week statistics from a vector store and asks
an LLM to answer **using only that retrieved context**, citing the stats it relied on.
This grounds the answer in verifiable data instead of the model's (possibly stale or
hallucinated) memory.

## Research question (RQ1)

> Does grounding an LLM's fantasy football recommendations in a retrieval-augmented
> pipeline over structured 2024-season statistics improve factual faithfulness and
> outcome accuracy compared to prompting the same LLM directly, without retrieval?

To answer this, the repo implements **two systems** — a RAG pipeline and a no-retrieval
baseline — and an evaluation harness that compares them.

## Architecture

```
nfl_data_py (2024 stats)
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
├── Dockerfile                 # reproducible container
├── .gitignore
├── src/
│   ├── config.py              # central configuration (loads configs/model_config.yaml)
│   ├── data_loader.py         # nfl_data_py ingestion (cached)
│   ├── preprocess.py          # stat rows -> text passages
│   ├── embed_store.py         # embeddings + ChromaDB
│   ├── entities.py            # entity extraction for retrieval filtering
│   ├── generator.py           # provider-agnostic LLM (Gemini / OpenAI)
│   ├── build_index.py         # one-command data pipeline
│   ├── rag_pipeline.py        # RAG system (RQ1 treatment)
│   ├── baseline.py            # no-retrieval baseline (RQ1 control)
│   └── model_runner.py        # single-command entry point
├── utils/
│   └── helpers.py             # shared helpers (formatting, timing, logging)
├── configs/
│   └── model_config.yaml      # model + pipeline hyperparameters
├── eval/
│   ├── benchmark_queries.json # fixed 20-query benchmark
│   └── evaluate.py            # RQ1 evaluation harness
├── outputs/
│   ├── samples.txt            # generated RAG vs. baseline answers
│   ├── sample_generations.json
│   └── README.md              # description + analysis of outputs
├── docs/
│   ├── methods.md             # research, literature review, benchmarking
│   └── preliminary_results.md # initial findings
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

# 4. (Only needed for the generator LLM) set your API key.
#    Default backend is Google Gemini (free tier, no billing):
export GEMINI_API_KEY="AIza..."     # free key from https://aistudio.google.com
#    Or use OpenAI instead:
#    export LLM_BACKEND=openai && export OPENAI_API_KEY="sk-..."
```

The embedding step uses a **local** model by default (no API key needed), so the
entire data pipeline and vector store can be built for free and offline. A generator
key (Gemini by default, free) is only required to run `rag_pipeline` / `baseline` /
`model_runner` / full `evaluate`.

**Note on Python version:** `nfl_data_py` pins `pandas<2.0`, which requires **Python
3.9–3.11**. The provided Dockerfile uses Python 3.11 for a clean, reproducible setup.

## Usage

### 1. Build the data pipeline (ingest → passages → embed → store)

```bash
python src/build_index.py
```

This downloads the 2024 weekly stats (cached to `data/raw/`), builds passages
(`data/processed/passages.jsonl`), and populates the ChromaDB vector store
(`data/chroma/`). No API key needed.

### 2. Inspect ground truth without any API key

```bash
python -m eval.evaluate --ground-truth
```

Prints, for each start/sit benchmark question, which player actually scored more PPR
points that week — computed directly from the data.

### 3. Run the end-to-end pipeline (needs a generator key)

```bash
python src/model_runner.py            # 5 sample queries
python src/model_runner.py --num 8    # 8 sample queries
```

Runs both the RAG pipeline and the baseline on benchmark queries and saves generations
to `outputs/samples.txt` and `outputs/sample_generations.json`.

### 4. Run the full RQ1 evaluation (needs a generator key)

```bash
python -m eval.evaluate
```

Produces `outputs/evaluation_results.csv` with per-query RAG vs. baseline answers,
automatic outcome-accuracy scoring on start/sit questions, latency, and blank columns
for human faithfulness/helpfulness annotation.

### Docker (optional, fully reproducible)

```bash
docker build -t ff-copilot .
docker run --rm -v "$PWD/data:/app/data" ff-copilot python src/build_index.py
docker run --rm -e GEMINI_API_KEY=$GEMINI_API_KEY \
  -v "$PWD/data:/app/data" -v "$PWD/outputs:/app/outputs" \
  ff-copilot python src/model_runner.py --num 8
```

## Preliminary results

Running `python src/model_runner.py --num 8` (Gemini `gemini-2.5-flash` generator,
local MiniLM embeddings) on eight start/sit questions produced RAG vs. baseline answers
saved in [`outputs/`](outputs/). Three patterns emerged (full analysis in
[`outputs/README.md`](outputs/README.md)):

1. **Grounding supplies knowledge the base model lacks.** The baseline model's training
   cutoff predates the 2024 season, so it often replies that "the 2024 season has not
   happened yet." The RAG system answers correctly because the real statistics are
   retrieved and supplied at inference time.
2. **Grounding produces correct, cited answers.** e.g. Bijan Robinson vs. Saquon Barkley
   (Wk8): RAG names Bijan and cites 23.6 vs. 12.1 PPR; the baseline gives an unsourced
   pick.
3. **Grounding reduces hallucination.** When a compared player did not play that week,
   the RAG system says so instead of fabricating a stat line.

These early observations support RQ1 (retrieval grounding improves faithfulness and
outcome accuracy). The full 20-query evaluation with human faithfulness ratings is the
final-milestone deliverable.

## How outcomes are scored (and why it is honest)

Start/sit questions are scored **objectively**: the ground-truth "correct" answer is
whichever player actually scored more PPR fantasy points that week, computed directly
from `fantasy_points_ppr` in the 2024 data — never hand-labeled. A system is "correct"
if its recommendation names that player. Draft questions have no single correct
outcome and are evaluated qualitatively (faithfulness + helpfulness).

## Method choices (short version)

| Component | Choice | Why |
|-----------|--------|-----|
| Data | `nfl_data_py` (single source) | Clean, maintained, no scraping |
| Embeddings | `all-MiniLM-L6-v2` (local) | Free, fast, reproducible; OpenAI optional |
| Vector store | ChromaDB | Lightweight, persistent, local |
| Generator | `gemini-2.5-flash` (pretrained, frozen; OpenAI optional) | Strong grounded-QA, free tier |

Full rationale, literature review, and benchmarking are in
[`docs/methods.md`](docs/methods.md).

## Roadmap

- **Milestone 3:** data pipeline, RAG + baseline implementation, evaluation harness,
  repository.
- **Milestone 4 (current):** single-command runnable pipeline (`python
  src/model_runner.py`), modular `utils/` + `configs/`, saved sample outputs, Dockerfile.
- **Milestone 5:** full 20-query RQ1 evaluation with human faithfulness ratings,
  Streamlit UI with a transparency sidebar, technical report, and presentation. Stretch
  goals: news corpus (RQ2), chunking study (RQ3).

## Known issues / limitations

- The generator requires an API key (Gemini free tier by default, OpenAI optional). A
  local Llama-3 fallback via Ollama is planned.
- `nfl_data_py` pins `pandas<2.0`, so the project requires Python 3.9–3.11 (the
  Dockerfile pins 3.11 to make this reproducible).
- When one player in a start/sit comparison did not play that week (bye/injury),
  retrieval returns only the other player and the RAG system declines to compare rather
  than guessing. A clearer "player did not play" message is a planned improvement.
- The current corpus is stats-only; a news corpus is a documented stretch goal.
- Faithfulness and helpfulness are assessed qualitatively at this stage; the full
  20-query human evaluation is the final-milestone deliverable.
