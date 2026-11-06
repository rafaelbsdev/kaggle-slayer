---
title: Widen the feature vocabulary with ideas measured by the Kaggle community
status: accepted
created: 2026-11-04
decided: 2026-11-04
---

## Problem

After four rounds on `playground-series-s6e11`, CV and public score agree to within 0.0003, so validation is
sound. But the best public score was 0.94763, rank 332 of 606, while the top 5% needs about 0.9521. The agents
could not close a gap that size, because the closed vocabulary (decision 0003) did not hold the missing ideas.

## Source

Public discussions on the competition, read by a person, not by the system:

- Yash Teotia, "What each idea is worth: a 0.9419 to 0.9508 score ladder on the same 5 folds", which credits
  goodpjw2008 (digit keys, tiny trees), SIDHAARTH SHREE (per-value encoding), Areeb Ali and Tudor Beloiu
  (seed-bagged encodings). Discussion 755149.
- Mohammed Majeed, "Stuck at 0.9505 CV / 0.9511 LB". Discussion 755019.

The numbers in this file come from those posts until our own ledger replaces them.

| Idea | Their measured gain | What we did |
|---|---|---|
| Digit keys | +0.00147 | Added `digit_keys`, later rebuilt as `number_keys` (decision 0008) |
| Per-value target encoding, nested out of fold | +0.00113 | Already supported, and the prompt now says so |
| Tiny trees | +0.00108 | Already allowed, and the prompt now suggests them |
| Seed-bagged encodings | +0.00019 | Added `target_encode_bags`, later a separate inner split per model seed (decision 0008) |
| GLM in the blend | Part of +0.00033 | Added the `logistic` family, later rebuilt on log-odds (decision 0008) |

We reimplemented the ideas in our own code. Building them from the short descriptions alone got three of them wrong,
and decision 0008 records how that was found. Decision 0009 adds the next round of public ideas.

## Decision

Each idea enters as a generic option that any competition can use, and the agents choose whether to try it.
CV decides whether it stays. The leak canary covers the new encodings too.

## Alternatives discarded

- **Letting the agents write feature code:** still ruled out by decisions 0002 and 0003.
- **Adding the original dataset as extra rows:** the public posts report no clear gain, and our data pipeline
  would need a second source. Open to revisiting if our own results point there.
- **Ranking by public score:** the same post measures about 0.003 of noise per submission on this board, more
  than most gains above. Finals should be chosen by CV.

## Limit worth stating

A person found these ideas and added them to the vocabulary. The agents then choose among them, but they did
not discover them. An agent that reads public discussions on its own would close that gap, and is not built.

## How we would know it worked

The Playground CV and public score rise by about the amounts above, measured in the ledger.

Our own measurements on `playground-series-s6e11`, before the rebuild in decision 0008:

| Idea | Compared | Result |
|---|---|---|
| Digit keys | LightGBM 0.94675 to 0.94855, XGBoost 0.94709 to 0.94881, CatBoost 0.94712 to 0.94875 | Clear gain |
| Exact-value encoding, cap 700000 | About +0.0005 on all three families | Clear gain |
| Tiny trees | LightGBM on plain columns, 31 leaves 0.94629, depth 2 0.94757 | Clear gain |
| Seed-bagged encodings as we first built them | LightGBM, 1 bag 0.94912, 3 bags 0.94878 | Loss, against the public +0.00019 |
| Row statistics | -0.00007 | No gain |
| Frequency encoding, cap 700000 | +0.00009 on deep trees, within noise on tiny ones | No clear gain |
| Dropping 5 drifting features | 0.91997 | Heavy loss. This data has no drift, and the profile had said so |
