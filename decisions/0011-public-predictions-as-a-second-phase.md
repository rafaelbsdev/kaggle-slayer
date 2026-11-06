---
title: Measure what other people's predictions add, as a second phase kept apart from the team's score
status: accepted
created: 2026-11-04
decided: 2026-11-04
---

## Problem

Decision 0009 kept this team to models it trains itself. By the night of 2026-11-04 that line had a known ceiling. Reading the
top public notebooks showed that the scores above about 0.9520 come from stacking other people's published predictions, and
that one author's own models stopped at 0.95146 and reached 0.95211 with them. The owner wanted two answers for later: how far
this team goes alone, and how much the rest of the board is worth.

## Decision

The owner chose two phases, in this order.

1. **The team's own models, as far as they go.** Before anything else, its standing is recorded in the ledger as
   `phase1_snapshot`. At that moment: public 0.9517, rank 148 of 721, top 20.5%. Top 20% needed 0.95172. Own models kept running
   after this, and later snapshots are in the same ledger.
2. **With public predictions.** `tools/public_pool.py` stacks the team's models with published ones. It runs by hand, the
   agents never see these files, and its scores go to `runs/playground-series-s6e11-public-pool/ledger.jsonl`.

Kaggle shows only a team's best score, so from here the public board shows phase 2. Phase 1's number lives in the snapshot and in
the team's own ledger, and the two are never to be reported as one.

## Rules and sources

Section 6 of the competition's rules allows external data and models that are public and equally accessible to all participants,
unless the host forbids them. The owner approved each download by name, source and size. All three use the same five folds as
this team, and the tool refuses a split that differs where the files let it check.

| Source | What | License |
|---|---|---|
| Masaya Kawamata, `masayakawamata/s6e11-gpt2-target-first-oof` | GPT-2 target-first classifier, by the method of Chris Deotte | CC BY 4.0 |
| goodpjw2008, `goodpjw2008/s6e11-oof-bank` | 70 models | CC0 |
| alex chilton, `alexchilton/s6e11-oof-predictions` | 38 models | CC0 |

## What we measured

Weights fitted on four folds and scored on the fifth.

| Mix | CV | Public |
|---|---|---|
| Own models, hill climb | 0.95177 | |
| Own models and GPT-2, hill climb | 0.95203 | |
| Own models, logistic stack | 0.95166 | |
| 108 public models, logistic stack, none of ours | 0.95201 | |
| Own and public, logistic stack | 0.95212 | 0.95202 |
| The same without GPT-2 | 0.95159 | |
| Own, GPT-2 and goodpjw2008's bank, logistic stack | 0.95202 | |
| Own, GPT-2 and alex chilton's models, logistic stack | 0.95206 | |
| Rank mean of the stack over everything and the hill climb over own and GPT-2 | 0.95218 | |

The team's models still add 0.00011 to 108 public ones, and the best single model here (0.95159) is above every public one
(0.95151). GPT-2 is worth the most: taking it out of the stack costs 0.00054, and adding it to the hill climb over our own models
gives 0.00026. We could not have trained it: its author reports about 120 GPU hours. Decision 0012 tried a smaller one of our
own, adds three more sources, and fixes a solver tolerance that held these stacks back.

## What we did not do

- Rank-average other people's submission files. That is what fills the very top of this board, and those files contain each
  other. Every predictor here has out-of-fold scores on our folds, so its worth can be measured.
- Let the agents use these files. The brief asks for a team that competes on its own, and this phase is a person's decision.

## How we would know

The public board is noisy and these stacks are close to each other, and final ranks use rows it does not show. The competition
closes on 2026-11-30, after delivery, so whether this mix holds up is unknown at delivery.
