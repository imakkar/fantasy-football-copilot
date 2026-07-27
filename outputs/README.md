# Outputs

This directory holds representative outputs produced by the end-to-end pipeline.

## Files

- **`samples.txt`** — human-readable RAG vs. baseline answers for 8 benchmark queries.
- **`sample_generations.json`** — the same results in machine-readable form (query,
  both answers, latencies, number of passages retrieved).
- **`evaluation_results.csv`** — per-query results from the evaluation harness
  (`eval/evaluate.py`), including data-derived ground truth for start/sit questions.

## How to regenerate

```bash
python src/build_index.py        # once, to build the vector store (no API key)
export GEMINI_API_KEY=...         # free key from https://aistudio.google.com
python src/model_runner.py --num 8
```

## What was generated

For each of 8 start/sit questions, the system produced two answers: one from the **RAG
pipeline** (grounded in retrieved 2024 statistics) and one from the **no-retrieval
baseline** (the same Gemini model with no context). Generator: `gemini-2.5-flash`;
embeddings: local `all-MiniLM-L6-v2`.

## Observations (preliminary analysis)

Three patterns appear consistently and support the project's RQ1 hypothesis:

1. **Grounding supplies knowledge the base model lacks.** The baseline model's training
   cutoff predates the 2024 season, so on several questions (e.g. Kelce vs. Kittle Wk7,
   Hurts vs. Mahomes Wk14) it responds that "the 2024 season has not happened yet" and
   cannot answer. The RAG system answers correctly because the real box-score data is
   retrieved and injected at inference time.

2. **Grounding produces correct, cited answers.** On questions where both systems
   answer (e.g. Bijan vs. Saquon Wk8), the RAG answer names the right player *and cites
   the exact PPR totals* (Bijan 23.6 vs. Saquon 12.1), while the baseline gives an
   unsourced pick based on general reputation.

3. **Grounding reduces hallucination.** When a compared player has no game that week
   (bye/injury, e.g. Josh Allen Wk12, A.J. Brown Wk3, Tyreek Hill Wk6), the RAG system
   states that the context lacks that player rather than fabricating a stat line. The
   baseline instead produces a confident guess.

## Known limitations

- When one player in a comparison did not play that week, retrieval returns only the
  other player, so the RAG system correctly declines to compare rather than answering.
  A future improvement could detect this and surface a clearer "player X did not play"
  message.
- Faithfulness and helpfulness are assessed qualitatively here; the full 20-query
  evaluation with human ratings is planned for the final milestone.
