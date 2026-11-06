---
title: A solver that stopped early, and what the last day tried
status: accepted
created: 2026-11-05
decided: 2026-11-05
---

## Problem

On 2026-11-05, the last full day before delivery, phase 1 stood at public 0.95171, rank 159 of 766, and phase 2 at 0.95202,
rank 115. Both had a day of compute left and no submissions until 21:00 local, when the daily quota returns. This records what
that day tried, what helped, and what did not.

## A solver that stopped early

Both the team's logistic start (decision 0009) and the phase 2 stack fit scikit-learn's `LogisticRegression`. Its default
tolerance, `tol=1e-4`, stops L-BFGS after about 20 to 45 steps on this data. With frauds at 1.7% of rows and many nearly
identical columns, that is far from the optimum: on synthetic data of the same shape, the default lands 0.079 away from the
exact coefficients. A stack that stops early can lose CV when members are added, which is what we kept seeing.

- **Phase 2.** `tools/public_pool.py` now fits with `tol=1e-7`. The same 123 members went from 0.952167 to 0.952196 at 1e-6.
  Tolerances below 1e-6 changed nothing further.
- **The team.** `FeaturePlan` has two new options, `logistic_tolerance` (default 1e-4, as before) and `logistic_start_penalty`
  (default 1.0, as before). On the top logistic_boost plan, 1e-6 alone raised CV by 4.5e-5, 1.9e-5 and 5.2e-5 on seeds 320,
  321 and 324; a start penalty of 10 added more on two seeds of three. Defaults stay where they were, so old plans reproduce the
  ledger exactly.
- **What reached the blend.** A six-seed bag of the new plan scored the same as the old bag (0.951503 against 0.951504), and
  swapping it into a fixed mix lost 0.8e-5. The team's own blend still rose from 0.951905 to 0.951927 during a run that spent
  26 of 30 slots on the new options. That gain is inside board noise.

Adding the two options broke the next run: the API refused `FeaturePlan` as "Schema is too complex" at 15 optional fields, and
the ledger has a `brain_unavailable` stop from it. `slayer/brains.py` now sends every field as required, with each default still
written in its description.

## A language model of our own

Phase 2 showed the public GPT-2 member is worth the most (decision 0011). We tried training one ourselves: a small transformer
that writes a row, label first, and is asked which label makes the row more likely. The method is Chris Deotte's, as described
in Masaya Kawamata's public notebook; the model and its training are ours. Pretraining read every train and test row with the
label hidden, then each fold was fine-tuned on its own labels. It ran on the RTX 4060 in a separate environment with torch,
which the owner approved.

Fold 0, where our own mix scores 0.9508:

| Member | AUC alone | Added to our mix |
|---|---|---|
| Ours, 256 wide, 6 layers, best epochs averaged | 0.94732 | +1.3e-5 at 5% |
| Ours, 320 wide, 8 layers, 10.5M parameters, 20 epochs | 0.94346 | none |
| Public GPT-2, same fold | 0.94888 | +23.1e-5 at 20% |

Ours is close alone and adds almost nothing. We stopped it there. Its author reports about 120 GPU hours for the public one, and
we had one day. Its scripts were scratch work and are not in this repository. Running two training jobs on one 8 GB GPU at once spilled into system memory and cost 89 minutes with nothing
finished; after that, one job at a time.

## Other things that did not help

- **The original dataset** (`uditjain13/credit-card-fraud-detection-2026`, CC0, 20,000 rows, 339 frauds). Playground data is
  generated from a dataset like this, and some winners use it. A model trained on it alone scored 0.93184 on our rows, with
  correlation 0.839 to ours; in the mix it lost 2.5e-5, as a stack member 1.7e-5, as value rates 10.4e-5, and as "seen in the
  original" flags 3.8e-5. No row matched exactly. Its label is stored as text such as `"0.0"`, and a cast error printed three
  label values of that file (no features) into this session.
- **Stacking our own plans.** A nested stack over the top 15, 30, 60 and all 225 plans scored 0.95169, 0.951771, 0.951864 and
  0.951642. Half of the 60-plan stack mixed into our fixed mix moved the fold means by -1.5, +4.7, +7.1, +4.8 and -3.3 (e-5).
  Too uneven to trust.
- **Non-negative stack weights in phase 2.** 0.95192 against 0.95231 for the plain stack, and the number of members used
  ranged from 18 to 108 across folds.

## Phase 2: three more sources

The owner approved each download by name, source and size.

| Source | What | License | In the final stack |
|---|---|---|---|
| goodpjw2008, `goodpjw2008/s6e11-fraud-series-artifacts` | predictions of a notebook series | CC0 | 24 |
| queyrusi, `queyrusi/fraud-shared-digit-cnn-weights` | a CNN and a generative model over digits | Apache 2.0 by its `NOTICE.txt`; Kaggle shows "unknown" | 2 |
| malyshevdanil, `malyshevdanil/s6e11-pooled-stack-v9` | 32 members pooled from lazyyeou, nilanjansaha123, goodpjw2008 and malyshevdanil's own work | CC0 | 32 |
| sadamtorres, `sadamtorres/ps-s6e11-tabpfn-gbdt-oof-test-pairs` | TabPFN and tree models | CC0 | 0 |
| malyshevdanil, `malyshevdanil/s6e11-cross-fitted-artifacts` | six models, also inside the pooled set | CC0 | 0 |

The series ships no fold ids. For it and for sadamtorres' members, a fold check compared how far the fold means spread under our
folds against 20 random splits; a member is kept when ours spread at least three times wider. Three members failed it, and three
more were trained on ten folds, so none of those went in.

| Stack | Members | CV |
|---|---|---|
| Decision 0011, default tolerance | 123 | 0.952121 |
| The same sources later in the day, default tolerance | 123 | 0.952167 |
| The same, tolerance fixed | 123 | 0.952196 |
| Plus a smoothed bag of our own, the series and the two queyrusi models | 150 | 0.952287 |
| Plus sadamtorres' 7 | 157 | 0.952293 |
| Plus the pooled 32 | 182 | 0.952322 |
| Everything | 195 | 0.952316 |

Our own members change as the team adds plans, so rows measured hours apart differ by that too. The pooled set gave the largest
step, and the 182-member stack is what goes forward. That set also ships one author's submission file as an "anchor"; it has no
out-of-fold predictions, so, as in decision 0011, it stays out.

Seen and not downloaded: two GPT-2 checkpoints from `dosomethingbetter` (CC0, about 500 MB each; tuning them on five folds
would take about 13 GPU hours here and needs `transformers`), sets that hold only submission files and no out-of-fold
predictions, and two with the license "other".

## How we would know

The quota returned at 21:00 local, and all ten submissions of the next day went in that night, read between 00:01 and 00:36 UTC
on 2026-11-06. The first three:

| File | Phase | CV | Public |
|---|---|---|---|
| `blend-r92`, the team's best blend | 1 | 0.951927 | 0.95165 |
| The same, mixed 70/30 with the 60-plan stack above | 1 | 1.4e-5 over `blend-r92` on the mean of folds | 0.95171 |
| The 182-member stack | 2 | 0.952314 | 0.95199 |

That left phase 1 at 185 of 841, 0.00017 under the top 20%, and the owner asked to spend the other seven on phase 1 to try for it.
We mixed `blend-r92` with the best bag of each own family and with the stack, and measured the mean of fold AUCs (`blend-r92`
alone: 0.951965). CatBoost, logistic and glm_boost lowered it at every weight from 10% to 30%, and every mix kept a test rank
correlation with `blend-r92` of at least 0.9994. Six with the best mean went in, spread over the stack's weight, and the stack
alone, the mix least like `blend-r92`:

| Mixed into `blend-r92` | Mean of folds, gain | Public |
|---|---|---|
| 50% stack, 10% LightGBM | +2.2e-5 | 0.95175 |
| 50% stack | +1.6e-5 | 0.95173 |
| 30% stack, 10% XGBoost | +1.5e-5 | 0.95169 |
| 60% stack | +1.5e-5 | 0.95174 |
| 10% LightGBM | +1.2e-5 | 0.95168 |
| 70% stack | +1.1e-5 | 0.95175 |
| The stack alone | -0.8e-5 | 0.95177 |

The stack mixes went in although their fold gains were uneven: fold 0 lost and fold 2 gained at every weight. Like the operator's
rank means before them, these mixes were written by scratch scripts that are not in this repository. Each submission's reason in
the ledger gives its recipe, and the stack is the one `tools/public_pool.py` fits, over bags of up to six seeds of our 60 strongest
plans. Every reading is
within 0.0002 of its phase's best before this day (0.95171 and 0.95202), so by decision 0010's rule none of this day's gains can be
read on the public board. The best phase 1 reading, 0.95177, came from the mix with the lowest CV of the seven, and it is the best of
28 phase 1 submissions, so selection favours it. Phase 1 stands at 176 of 841, top 20.9%, 0.00011 under the 0.95188 the top 20%
needed at that time. Phase 2 stands at 139 of 841, top 16.5%. The gains here are CV gains on all 699,960 labelled rows; the private
board, when the competition closes on 2026-11-30, is the reading that can confirm them.
