# Preliminary Results (Milestone 3)

This document records the initial results from running the data pipeline end to end
on real 2024-season data. It satisfies the "preliminary experiments" and "initial
results and analysis" requirements of Milestone 3.

## Pipeline execution

Running `python -m src.build_index` on the completed 2024 NFL season produced:

- **5,597** weekly stat rows loaded from `nfl_data_py` (QB/RB/WR/TE after filtering).
- **5,480** natural-language passages constructed (one per player per week).
- **5,480** passages embedded with `all-MiniLM-L6-v2` and stored in a persistent
  ChromaDB collection (`nfl_2024_passages`).

The entire data pipeline runs on CPU with no API key, in a few minutes.

## Ground-truth computation (start/sit)

`python -m eval.evaluate --ground-truth` computes the true higher scorer for each
start/sit question directly from `fantasy_points_ppr`. Selected verified results:

| Question | Player A | Player B | Winner (more PPR) |
|----------|----------|----------|-------------------|
| Wk5: Jefferson vs Lamb | Jefferson 15.2 | Lamb 11.4 | **Jefferson** |
| Wk8: Bijan vs Saquon | Bijan 23.6 | Saquon 12.1 | **Bijan Robinson** |
| Wk4: Henry vs Jacobs | Henry 35.9 | Jacobs 11.8 | **Derrick Henry** |
| Wk2: Kyren vs Kamara | Kyren 15.2 | Kamara 44.0 | **Alvin Kamara** |
| Wk9: Nacua vs G.Wilson | Nacua 2.1 | Wilson 30.0 | **Garrett Wilson** |
| Wk13: Bowers vs McBride | Bowers 30.2 | McBride 21.6 | **Brock Bowers** |
| Wk15: Daniels vs Burrow | Daniels 23.6 | Burrow 16.8 | **Jayden Daniels** |

11 of 15 start/sit questions produced objective ground truth. The other 4 involved a
player who did not record a game that week (bye or injury); these are automatically
flagged as qualitative rather than hand-labeled, keeping the metric honest.

## Retrieval sanity check

For the query *"In Week 5 of the 2024 season, should I have started Justin Jefferson
or CeeDee Lamb?"*, the retriever returns relevant CeeDee Lamb and Justin Jefferson
passages (cosine distances ~0.43-0.45).

**Observed limitation (initial):** pure semantic retrieval matched strongly on player
name and sentence structure but did not reliably prioritize the *specific week* asked
about, and could over-retrieve one player. For the Week 5 query above, the top hits
were CeeDee Lamb passages from weeks 2, 10, and 13, and Justin Jefferson did not
appear in the top 5 at all.

**Fix implemented: metadata filtering.** Because each passage carries structured
`player`, `week`, and `season` metadata, we added a lightweight entity extractor
(`src/entities.py`) that pulls player names and the week number from the question
text, then builds a ChromaDB `where` filter so retrieval is restricted to matching
passages before ranking. The RAG pipeline applies this automatically and falls back to
plain semantic search if the filter matches nothing (e.g. a bye week).

**Result after the fix** — same Week 5 query now retrieves exactly the two relevant
passages:

```
1. Week 5 2024: CeeDee Lamb (WR, DAL) vs PIT -- 11.4 PPR
2. Week 5 2024: Justin Jefferson (WR, MIN) vs NYJ -- 15.2 PPR
```

Both players, correct week, nothing extraneous. For draft questions (no week), the
filter restricts to the named player across the full season, which is exactly the
context needed for draft reasoning. This directly grounds the LLM in the correct facts
and is expected to improve RQ1 outcome accuracy.

## End-to-end generation: early RQ1 signal

Running `python -m src.model_runner` (Gemini `gemini-2.5-flash` generator, local
MiniLM embeddings) on the first five benchmark questions already shows the RAG vs.
baseline contrast that RQ1 predicts:

| # | Question | RAG | Baseline | Ground truth |
|---|----------|-----|----------|--------------|
| 2 | Bijan vs Saquon, Wk8 | **Bijan** (cites 23.6 PPR) — correct | **Saquon** (cites "Eagles offense") — wrong | Bijan 23.6 > Saquon 12.1 |
| 3 | Allen vs Lamar, Wk12 | States the context lacks Josh Allen; does not guess | Confidently picks Allen | Allen has no Wk12 game row |
| 4 | St. Brown vs A.J. Brown, Wk3 | States the context lacks A.J. Brown; does not guess | Confidently picks St. Brown | A.J. Brown row missing |
| 1, 5 | Jefferson/Lamb; CMC/Gibbs | Correct, with cited PPR points | Correct, but unsourced | — |

Two patterns stand out:

1. **Grounding fixes confident errors.** On Q2 the baseline reasoned from narrative
   ("high-powered Eagles offense") and picked the wrong player; the RAG system read the
   actual box score and picked correctly.
2. **Grounding reduces hallucination.** On Q3 and Q4 the RAG system explicitly reported
   that the retrieved context did not contain the second player (who did not play that
   week) instead of fabricating an answer, whereas the baseline always produced a
   confident pick.

These are qualitative observations on five questions; the full 20-query evaluation
(with human faithfulness/helpfulness ratings) is Milestone 5 work. But the early signal
supports the RQ1 hypothesis.

## Takeaways

- The full data pipeline (ingest -> passages -> embed -> store -> retrieve) works end
  to end on real data.
- Objective, reproducible ground truth for start/sit accuracy is available directly
  from the data.
- Retrieval precision on week-specific questions was initially weak but is resolved by
  metadata filtering driven by entity extraction from the question text.
