---
title: Compete in CloudWalk's own competition and in a live fraud Playground
status: accepted
created: 2026-11-03
decided: 2026-11-03
---

## Problem

The Nimbus brief asks for a team of agents that takes part in Kaggle competitions and aims for
the top 20%. It names no competition. The owner raised the target to top 5%, top 1% if possible.

## Decision

Run one system, with no competition-specific code, on two live tabular competitions:

| | `large-scale-loan-repayment-modeling-challenge` | `playground-series-s6e11` |
|---|---|---|
| Host | CloudWalk | Kaggle |
| Domain | Credit | Card fraud |
| Teams on 2026-11-03 | 11 | about 590 |
| Metric | AUC | AUC |
| Submissions per day | 5 | 10 |
| Public leaderboard | about 30% of test | about 20% of test |
| Closes | about 6 months away | 2026-11-30 |

Snapshot of public scores on 2026-11-03: CloudWalk first place 0.74111. Playground rank 6 (top 1%)
0.95231, rank 29 (top 5%) 0.95206.

## Limits we state up front

- With 12 teams, first place on the CloudWalk board is 8.3%. Top 5% does not exist there, so the
  goal is first place.
- On the Playground, ranks 1 to 49 sit within 0.0004 AUC and the cutoffs will rise. Top 1% is a
  stretch goal, not a promise.
- At delivery (2026-11-06) both boards are public only. Final standings come after each closes.

## Alternatives discarded

- **Only CloudWalk's competition:** a percentile over 12 teams says little, and one competition
  does not show the system generalises.
- **Getting Started competitions (Titanic and similar):** the top of those boards is widely known
  to be polluted by leaked labels, so a rank there would prove nothing.
- **Closed competitions:** late submissions get a score but no live rank to monitor.

## How we would know it worked

The ledger of each run shows real submissions through the API and the rank read back from Kaggle.
