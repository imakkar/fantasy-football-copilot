# Research and Selection of Methods

This document records the method research behind the Fantasy Football Co-Pilot and
justifies the choices made in the data pipeline. It maps directly to Milestone 3
Required Component 1.

## 1. Objectives

The system answers natural-language fantasy football questions of two kinds:

- **Start/sit** ("In Week 5 of 2025, should I start Jefferson or Lamb?")
- **Draft** ("Is Malik Nabers a strong first-round pick for 2026?")

The core NLP task is **retrieval-augmented question answering (QA) over a
domain-specific corpus** of NFL statistics. The primary research question (RQ1) is
whether grounding a large language model in retrieved statistics improves factual
faithfulness and outcome accuracy relative to prompting the same model with no
retrieval.

## 2. Literature Review

**Retrieval-Augmented Generation (RAG).** Lewis et al. (2020) introduced RAG,
combining a parametric generator with a non-parametric retriever so answers are
grounded in an external corpus. Follow-up work (Karpukhin et al., 2020 on dense
passage retrieval; Gao et al., 2023 survey) shows RAG substantially reduces
hallucination on knowledge-intensive tasks and is now the standard architecture for
domain-specific assistants. This directly motivates our RQ1.

**Sentence embeddings.** Reimers and Gurevych (2019) introduced Sentence-BERT, the
basis for the `all-MiniLM-L6-v2` model used here. It offers a strong
quality-to-cost ratio: 384-dimensional embeddings, fast CPU inference, and no API
dependency, which matters for reproducibility.

**Instruction-tuned LLMs.** Decoder-only instruction-tuned models (OpenAI's GPT
family) follow task instructions and cite provided context reliably, making them a
good fit for a grounded-QA generator. We treat the LLM as a frozen, pretrained
component and do not fine-tune, consistent with the course guidance to prefer
pretrained models.

**Sports QA with LLMs.** Prior work on automated sports recaps and stat-grounded
narratives shows LLMs produce fluent, accurate sports language when supplied with
structured statistics, supporting the feasibility of our approach.

## 3. Benchmarking of Options

| Component | Options considered | Choice | Rationale |
|-----------|-------------------|--------|-----------|
| Embeddings | OpenAI `text-embedding-3-small`; `all-MiniLM-L6-v2` (local) | **Local MiniLM (default), OpenAI optional** | Local model is free, fast on CPU, and fully reproducible with no API key. OpenAI available via one env var for a quality comparison. |
| Vector store | ChromaDB; FAISS; Pinecone | **ChromaDB** | Lightweight, persistent, local, zero external services. FAISS lacks built-in metadata/persistence ergonomics; Pinecone needs an account. |
| Generator LLM | `gpt-4o-mini`; Llama-3-8B (Ollama) | **gpt-4o-mini** | Strong instruction-following and citation behavior at low cost; suits a solo project's budget. Llama-3-8B remains a documented fallback. |
| Orchestration | LangChain; LlamaIndex; hand-rolled | **Hand-rolled thin layer** | The pipeline is small enough that a direct implementation is clearer and easier to document/grade than a heavy framework dependency. |

**Efficiency / scalability.** One season of QB/RB/WR/TE weekly stats is a few
thousand passages (~10-15 MB), which embeds in minutes on CPU and queries in
milliseconds. The design scales to multiple seasons simply by adding years to the
loader.

## 4. Preliminary Experiments

Small-scale validation performed before full implementation:

1. **Data availability.** Confirmed `nfl_data_py.import_weekly_data([2025])` returns
   complete weekly stats for the 2025 season including `fantasy_points_ppr`, which we
   use as objective ground truth. This let us drop all web scraping (a major
   scope/risk reduction versus the Milestone 2 plan).
2. **Passage design.** Compared one-passage-per-player-week against
   one-passage-per-player-season. The per-week granularity retrieves far more
   precisely for start/sit questions (which are week-specific), so it was adopted.
3. **Ground-truth automation.** Verified that start/sit correctness can be computed
   directly from `fantasy_points_ppr` rather than hand-labeled, making the outcome
   metric objective and reproducible.

## 5. Selected Approach (summary)

A hand-rolled RAG pipeline: `nfl_data_py` -> natural-language passages -> MiniLM
embeddings -> ChromaDB -> top-k retrieval -> gpt-4o-mini generation, evaluated
against a no-retrieval baseline on a fixed 20-query benchmark.

## References

- Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS.
- Karpukhin et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering.* EMNLP.
- Gao et al. (2023). *Retrieval-Augmented Generation for Large Language Models: A Survey.* arXiv:2312.10997.
- Reimers and Gurevych (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP.
