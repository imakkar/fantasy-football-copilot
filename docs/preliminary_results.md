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

**Observed limitation:** pure semantic retrieval matches strongly on player name and
sentence structure but does not reliably prioritize the *specific week* asked about,
and can over-retrieve one player. For example, Lamb passages from weeks 2, 10, and 13
rank above the week-5 passage.

**Planned improvement (Milestone 4/5):** add metadata filtering. Because each passage
carries structured `player`, `week`, and `season` metadata, the retriever can filter
to the exact players and week named in the query before ranking. This should sharply
improve retrieval precision and is a natural next step motivated directly by this
finding.

## Takeaways

- The full data pipeline (ingest -> passages -> embed -> store -> retrieve) works end
  to end on real data.
- Objective, reproducible ground truth for start/sit accuracy is available directly
  from the data.
- The main open problem is retrieval precision on week-specific questions, which
  metadata filtering should address in the next milestone.
