# Final Results (Full 20-Query Evaluation)

This document reports the complete RQ1 evaluation over the fixed 20-query benchmark,
comparing the RAG pipeline against the no-retrieval baseline. It is the final,
full-scale version of the earlier `preliminary_results.md`.

Generator: Google `gemini-2.5-flash`. Embeddings: local `all-MiniLM-L6-v2`. Corpus:
completed 2024 NFL season (weekly QB/RB/WR/TE stats). Raw per-query results are in
`outputs/evaluation_results.csv`.

## Headline metrics

| Metric | RAG | Baseline |
|--------|-----|----------|
| Outcome accuracy (start/sit, 11 scored) | **100% (11/11)** | **55% (6/11)** |
| Faithfulness (human, 1-5, n=20) | **5.00** | **2.30** |
| Helpfulness (human, 1-5, n=20) | **3.95** | **1.75** |
| Avg latency (s) | ~3.2 | ~2.7 |

## How each metric was measured

- **Outcome accuracy** is objective and computed from the data: for each start/sit
  question, the correct answer is whichever player actually scored more PPR fantasy
  points that week (from `fantasy_points_ppr`). A system is correct if its answer names
  that player. 11 of 15 start/sit questions produced ground truth; the other 4 involved
  a player who did not play that week (bye/inactive) and are excluded from accuracy.
- **Faithfulness** (human-rated 1-5): are the answer's factual claims traceable to the
  retrieved data? RAG answers that cite specific stats score high; baseline answers that
  assert unsupported claims (or falsely state the season "hasn't happened") score low.
- **Helpfulness** (human-rated 1-5): does the answer actually help the user decide?
- **Latency**: end-to-end response time per query.

## Interpretation

1. **Grounding turns a coin-flip into certainty.** Start/sit questions are binary, so
   random guessing scores ~50%. The baseline scored 55% - essentially chance - because
   the frozen LLM has no access to the actual outcomes. The RAG system scored 100% by
   retrieving the real box scores. The 45-point gap is the core evidence for RQ1.

2. **The baseline's knowledge is stale.** Gemini's training cutoff predates the 2024
   season, so on many questions the baseline responds that "the 2024 season has not
   happened yet" and cannot answer. RAG supplies the missing facts at inference time.
   This is a concrete, real-world illustration of why retrieval augmentation matters.

3. **RAG is faithful by construction, and honest about gaps.** RAG faithfulness averaged
   5.0 because every claim traces to a retrieved passage. When a compared player did not
   play that week, RAG stated that the context lacked that player rather than fabricating
   a stat line - lowering its helpfulness slightly on those items but keeping
   faithfulness perfect. The baseline, by contrast, produced confident but unsupported
   (and often wrong) picks.

4. **Latency is comparable.** RAG adds a retrieval step but remains within ~0.5s of the
   baseline on average, well under the 5s target - the accuracy gains do not come at a
   meaningful speed cost.

## Limitations

- Faithfulness and helpfulness are single-rater human judgments; a multi-rater study
  would strengthen them.
- Outcome accuracy covers the 11 start/sit questions with ground truth; draft questions
  are inherently qualitative and were assessed on faithfulness/helpfulness only.
- The corpus is stats-only; a news corpus (RQ2) and chunking study (RQ3) remain future
  work.
