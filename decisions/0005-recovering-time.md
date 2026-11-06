---
title: Validate on a time order measured from the data, not assumed from the row order
status: accepted
created: 2026-11-03
decided: 2026-11-03
---

## Problem

CloudWalk's competition states a strict temporal split, so the first version used time folds following
the row order. It scored CV 0.811 while the best public score is 0.741: a difference far beyond any noise.
The folds were not temporal at all, because nothing in the row order or the ids carries time.

## What we measured

Everything below is on the real training data, with one LightGBM model. Only derived statistics were
inspected, in line with decision 0002.

| Check | Result | Reading |
|---|---|---|
| First vs last fifth of train, by row order | AUC 0.4996 | No time in the row order |
| First vs last fifth of train, by id | AUC 0.4981 | No time in the ids |
| Same, on a random split (control) | AUC 0.4987 | What "no signal" looks like |
| Test ids also present in train | 36,324 of 36,324 | Ids are numbered per file, not a shared key |
| Train vs test, all features | AUC 1.0 | Test is plainly a different period |

| Holdout (20% of train unless noted) | AUC | |
|---|---|---|
| Random | 0.815 | What the first version measured |
| Rows that look most like test, by adversarial model | 0.802 | Rejected, see below |
| Same, most similar 5% | 0.798 | |
| Same, dropping the 5 to 300 most drifting features | 0.800 to 0.801 | Dropping features did not close the gap |
| **Latest by `feature_492`** | **0.737** | Matches the leaderboard |
| **Latest by `feature_903`** | **0.747** | |
| **Latest by `feature_586`** | **0.743** | |
| Latest by `feature_96` | 0.701 | 18% nulls, 68% in test: too many to order rows |

`feature_492`, `feature_903` and `feature_586` are nearly one variable (Spearman 0.99), and 76% to 89% of
test lies above train's 99th percentile. They behave like a clock.

We do not know what these features are, because they are anonymised. The evidence for treating them as time
is that they reproduce the leaderboard level, and a real submission is what confirms or refutes it.

## Decision

- `slayer.data.find_drift_clock` looks through the most drifting features for one whose test values lie
  beyond train's range, with few nulls. No feature name is hardcoded, so this runs on any competition.
- `order_by: drift_clock` sorts training rows by that column.
- The verifier rejects time folds on any order not measured to follow time (`order_is_time`). This is the
  check that would have stopped the original mistake.
- Time folds are scored on their last block only, the one nearest the test period. Pooling blocks ranks rows
  across periods, so a drifting target rate looks like skill: an independent review measured AUC 0.575 on
  pure noise that way, against 0.50 to 0.52 per block.

## Alternatives discarded

- **Order by test likeness from an adversarial model:** implemented, then measured at 0.80, far from the
  leaderboard. It finds rows that resemble test, which is not the same as rows from later in time.
- **Dropping the drifting features to close the gap:** measured with almost no effect on that holdout.
- **Hardcoding `feature_492` for this competition:** works once and breaks the promise of no
  competition-specific code.

## How we would know it worked

The gap between CV and public score on real submissions. It should be near zero, not the 0.07 of the first
version. If it is not, this decision is wrong and the ledger will show it.
