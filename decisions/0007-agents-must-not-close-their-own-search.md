---
title: Stop the team from closing its own search
status: accepted
created: 2026-11-04
decided: 2026-11-04
---

## Problem

The ledger of the Playground run showed three failures that no single agent could see.

1. **A lesson closed a direction nobody had measured.** The Analyst wrote "Model Tuner: no tuning runs". The
   Tuner followed it, so it trained one model per round and never tried tiny trees, although its prompt
   suggested them. Public results for this competition had measured them as a large gain (decision 0006).
2. **The semantic check rejected good plans over wording.** The Feature Engineer proposed the best plan measured
   so far. The check rejected it twice over small inferences in the rationale, such as saying that amounts feed
   the digit keys.
3. **The fallback then made things worse.** After two rejections the heuristic took over and applied its baseline
   schedule, which has no digit keys and only a low target encoding cap. A whole round of models trained on weaker
   features than the team already knew how to build.

## Decision

- The Tuner fills its batch, and every batch keeps one model for a family or tree size not yet tried.
- The Analyst may not close a direction that past experiments have not measured, or ask for fewer models.
- The semantic check rejects contradictions and invented measurements. A loose summary, a reasonable inference
  about a switch, or a disagreement with an outdated lesson is no longer a reason to reject.
- From round 3 on, a heuristic taking over a feature decision reuses the plan of the best experiment. The first
  rounds keep the baseline schedule, because a run with no agents depends on it.
- `--no-gain-rounds` makes the stop rule explicit, because a one-switch comparison often brings no gain.

## Alternatives discarded

- **Keeping the agent's last answer after a semantic rejection:** the deterministic layer had passed it, but this
  would make the third layer advisory and the report would no longer say who decided.
- **Removing the semantic layer:** it caught real errors in earlier runs, such as a false claim about recency
  weighting.
- **Telling the agents the answer for this competition:** prompts stay general. They name directions any tabular
  competition can use, and CV decides.

## How we would know it worked

The Tuner proposes full batches with untried tree sizes, the fallback count drops, and the best CV rises. The
ledger of `playground-series-s6e11` records all three.

Measured on that ledger:

| | Rounds 7 to 10, before | Rounds 12 to 18, right after | Rounds 26 to 35 |
|---|---|---|---|
| Models per round | 1 | 3 to 4 | 4 |
| Fallbacks | 1 | 4 | 1 |
| Best single model | 0.94921 | 0.94997 | 0.95065 |

Batches filled at once. Fallbacks took longer to fall, and they did not stop: the semantic check still rejects 23 answers
over the whole ledger, some of them over rounding, and those rejections now cost a retry more often than a heuristic round.
The rise in the last column also owes much to decision 0008.
