---
title: Key the pairs that do not add up, count what stands behind each key, and pool tokens across fields
status: accepted
created: 2026-11-04
decided: 2026-11-04
---

## Problem

After decision 0009 the team's scores on `playground-series-s6e11` stopped rising. Best single model about 0.95119, best
blend 0.95136, and new seeds and depths moved neither by more than the noise. Top 20% needs about 0.9516 on the public
board, so another round of the same search was unlikely to pay. Three ideas from public work had not been measured here.

## Source

megayak's public notebook "S6E11 | Fast Per-Value + Digit Encodings | Shallow-Tree Stack", read by us and reimplemented in
our own code. It reports +0.00007 for joint keys on its booster, +0.00006 for pooled tokens on its blend and +0.00026 for
key counts on its booster. All models in this project are still trained by this team (decision 0009).

## What we measured

On the real folds, with `logistic_boost` at depth 4, learning rate 0.03, colsample 0.5, min_child 5, with `number_keys`,
`number_tokens`, `row_stats` and inner folds 3. The same seed gives the same base score as the ledger (0.95109 on seed 320).
Scores below skip the refit, which cannot change CV.

| Variant | Seed 320 | Seed 321 | Seed 322 | Mean | Logistic start |
|---|---|---|---|---|---|
| Base | 0.95109 | 0.95100 | 0.95119 | 0.95109 | 0.95057 |
| `key_counts` | 0.95109 | | | | 0.95057 |
| `shared_token_rates`, logistic only | 0.95120 | 0.95116 | 0.95120 | 0.95119 | 0.95066 |
| 8 joint keys, logistic only | 0.95114 | | | | 0.95075 |
| 8 joint keys, our scan | 0.95123 | 0.95119 | 0.95124 | 0.95122 | 0.95071 |
| 8 joint keys, the notebook's list | 0.95123 | 0.95113 | 0.95118 | 0.95118 | 0.95074 |
| Joint keys and counts, our scan | 0.95130 | 0.95123 | 0.95119 | 0.95124 | 0.95071 |
| Joint keys and counts, the notebook's list | 0.95131 | 0.95129 | 0.95126 | 0.95129 | 0.95074 |
| All three, the notebook's list | 0.95144 | 0.95143 | 0.95134 | 0.95140 | 0.95081 |
| All three, our scan | 0.95141 | 0.95132 | 0.95130 | 0.95134 | 0.95080 |
| All three, our scan, trees also read the rates | 0.95138 | 0.95135 | 0.95131 | 0.95135 | 0.95080 |
| The same with `joint_by_value` | 0.95144 | 0.95156 | 0.95141 | 0.95147 | 0.95080 |

The real pipeline with all three, refit included, scored 0.95139 on seed 320 with no missing prediction on any test row. The
team then chose this plan by itself and scored 0.95140 on seeds 323 and 324, above every blend it had made until then.

Blend gain, weights fitted on four folds and scored on the fifth, adding each variant to the 24 strongest plans:

| Added | Blend | Change |
|---|---|---|
| Nothing | 0.951442 | |
| `key_counts` | 0.951442 | 0, weight 0 |
| Joint keys, logistic only | 0.951442 | 0, weight 0 |
| Joint keys, our scan, three seeds | 0.951448 | +0.00001 |
| Joint keys, the notebook's list, three seeds | 0.951459 | +0.00002 |
| Joint keys and counts, three seeds | 0.951462 | +0.00002 |
| `shared_token_rates`, three seeds | 0.951498 | +0.00006 |
| All three, three seeds | 0.951511 | +0.00007 |

So each idea alone is small and they gain far more together: 0.0001 or so each, 0.0003 at once. Counts alone do nothing and
pay only once joint keys and pooled rates give them something to qualify.

CV can overstate this, so we also held out each fifth of the rows in turn and ran the real pipeline on the rest, test side included.
Seed 320, base plan against every option on:

| Rows no model saw | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | All five |
|---|---|---|---|---|---|---|
| Base | 0.94993 | 0.95149 | 0.95593 | 0.95172 | 0.94649 | 0.95111 |
| All options | 0.95046 | 0.95186 | 0.95625 | 0.95204 | 0.94704 | 0.95153 |
| Gain | +0.00053 | +0.00037 | +0.00032 | +0.00032 | +0.00055 | +0.00042 |

The gain holds on every holdout and is larger than the CV inside those runs showed (+0.00026). All five together also match this plan's
CV on the same seed, 0.95144, so that CV is not inflated.

## Decision

Three options in `FeaturePlan`, each off by default, so the agents choose them and every run says which it used:

- `joint_keys: int` target-encodes that many column pairs whose label effect does not add up, strongest first. Trees and the
  logistic model both read them. The notebook measured them as noise for plain trees and gave them to the booster, and here
  trees reading them beat a logistic that reads them alone by 0.00009 on seed 320.
- `joint_by_value: bool` keeps deciles in a joint key only for columns with more than 1000 distinct values, so a column with
  few keys each value. The scan still ranks pairs on deciles, so the same pairs are picked either way.
- `key_counts: bool` shows trees how many rows, in train and test, share each key's value. A linear model leaves them out.
- `shared_token_rates: bool` adds decimal token rates pooled over fields written alike: all decimals, first decimal, last
  whole token for fields with few values, and within a field the last whole token with its decimals. The notebook gives
  them to its linear model alone. Here trees reading them too scored the same, 0.95135 against 0.95134, so every model
  reads them and the code needs no mask to hide columns from trees.

Pairs come from a scan, not from a list of column names, so the option works on any competition. For each pair, on training
rows, it compares the label rate of every cell with what the two columns say alone added up in log-odds, and keeps the
pair when that likelihood ratio exceeds its degrees of freedom by 4 standard deviations. Continuous columns are cut in
deciles for this. On this data it finds 10 pairs in 16 seconds, and its strongest 8 are the 8 the notebook keeps. Scanning
is refused past 40 features, because pairs grow with the square of the columns.

Fields split into two groups by whether they have more than 1000 distinct values. The notebook names its groups by hand,
and this rule gives the same two groups here. We first assumed the smaller group sat between 0 and 1. Checked, it does not.

## The blend was missing other families

The first public reading of all this, blend-r57, was five seeds of one plan with test rows correlated at 0.999: CV 0.95168,
public 0.95151. So CV had risen twice as much as the board. By hand we then sent rank means of models the team already had,
and marked them as the operator's in the ledger:

| Rank mean | CV | Public |
|---|---|---|
| Five seeds of one plan | 0.95168 | 0.95151 |
| Three parts of the top plan's bag, one of LightGBM | 0.95178 | 0.95168 |
| Two parts of it, one each of two older boosters, LightGBM and XGBoost | 0.95171 | 0.95169 |
| The same with more of the top plan | 0.95175 | 0.95167 |
| The same plus CatBoost and logistic | 0.95162 | 0.95158 |

A quarter of LightGBM was worth 0.00017 on the board and almost nothing on CV. The hill climb could not find this for two
reasons: from one pick a second model only joins at half weight, which hurts, and no single LightGBM plan was strong enough
to reach the search. So the climb now starts its top pick at three parts, and each family gets one bag of its strongest models
across plans, which always reaches the search. On the same models it then chose three parts boosters to one part trees by
itself, CV 0.95175.

## How much of that reading was noise

We read that table as "the board rewards families that CV does not", and it was too strong. Later we measured the board's noise on
labelled rows: 400 draws of out-of-fold rows the size of the public board (60,000 rows, about 1,000 positive). One model's AUC moves
by a standard deviation of 0.0033 from draw to draw, which matters little because every team shares those rows. What matters is the
gap between two models on the same rows:

| Pair | True gap on all train rows | Standard deviation of the gap on a public-sized draw | First ahead |
|---|---|---|---|
| Family mix against five seeds of one plan | +0.00004 | 0.00017 | 62% of draws |
| Family mix against the same with more of the top plan | -0.00002 | 0.00004 | 29% |
| Own models with GPT-2 against own models alone | +0.00025 | 0.00018 | 92% |
| Five seeds of one plan against a bag of shallow LightGBM | +0.00043 | 0.00042 | 83% |

So a board gain of 0.00017 between our mixes is about one standard deviation: this table could not tell them apart. The last row is
wider, and it matters: one whole family can drift about 0.0004 against another on the rows the public board happens to hold, and every
model of that family drifts with it. That fits a family looking weak on the board in every reading, without any defect in it. Before
reading it this way we checked for one: the holdouts above show the test side working. The change to the hill climb costs nothing on CV
and stays, but its evidence is weaker than the paragraph above suggests. Two rules follow: final submissions are chosen by CV, not by
public score, and a difference under about 0.0002 on the public board is not read as a result. goodpjw2008's notebook series says the
same about gains of this size.

## Heavier smoothing as another view of the same plan

Trees and more seeds had stopped adding anything, so we asked what else could differ while staying strong. `target_encode_smoothing` sets
how hard each encoding leans on the base rate, and it was fixed at 20. On the top plan, seed 320, as a 20% rank mix with our best mix of the time
(CV 0.951795):

| Smoothing | Alone | Rank correlation with the top bag | Gain to the mix |
|---|---|---|---|
| 5 | 0.95144 | | -0.00001 |
| 100 | 0.95147 | | 0 |
| 300 | 0.95146 | 0.994 | +0.00003 |
| 1000 | 0.95120 | 0.990 | +0.00004 |
| 3000 | 0.95098 | 0.987 | +0.00005 |
| 10000 | 0.95052 | | +0.00003 |
| 30000 | 0.95020 | | +0.00001 |

So a plan can score less alone and still add more, and the option went into `FeaturePlan`, default 20, with a note to the Feature Engineer to judge it
by what it adds. Asked by operator note to train it on new seeds, the team tried 3000 and 1000, and its hill climb chose by itself 60% top bag, 20% shallow
LightGBM and 20% smoothing 1000 at depth 4: CV 0.951902, its best. Depth 5 and 6 at high smoothing, and CatBoost at 1000, did not reach the blend.

The same weights as fixed rank means, against the older mix at 0.951800 with the models of the same day:

| Rank mean | CV | Gain | Gain per fold, in 0.00001 |
|---|---|---|---|
| Top bag 75%, LightGBM 25% | 0.951818 | +0.00002 | +1.9, -2.4, -0.2, -6.6, +12.5 |
| Top bag 60%, LightGBM 20%, smoothing 1000 20% | 0.951887 | +0.00009 | +0.4, +13.7, +7.7, +3.9, +14.9 |
| The same with smoothing 3000 | 0.951864 | +0.00006 | -3.3, +13.8, +2.6, +0.7, +15.6 |
| The same with half the smoothing 1000 bag, first half | 0.951890 | +0.00009 | -0.9, +14.3, +8.9, +3.5, +17.5 |
| The same, second half | 0.951880 | +0.00008 | +1.4, +12.7, +6.4, +4.0, +12.0 |

Against the same proportions without it, smoothing adds about 0.00007, and either half of its seeds gives nearly the same gain. Four folds carry it and
the first barely moves. It is too small to read on the public board, by the rule above.

## What we did not do

- Keep the notebook's rule of a ratio of 1.2 between the likelihood ratio and its degrees of freedom. A test found it lets
  noise through on pairs with few degrees of freedom. Its list had already been filtered by held-out deviance; ours has not.
- Scan inside each fold. The scan reads every training label, held-out rows included. It only picks which pairs get a key
  and their encodings stay out of fold. The 8 pairs it picks are the 8 the notebook found by held-out deviance, so the
  selection bias on this data looks small, but that comparison is the only evidence we have.
- Train on 10 folds instead of 5. CV rises by about 0.00025 (0.95160 against 0.95135 on three seeds), but each model simply
  sees more rows. Holding out a fifth of the rows and running the real pipeline on the rest, test predictions from 5 and 10
  folds score the same on rows no model saw: 0.95046 and 0.95044 on one holdout, 0.95204 and 0.95207 on another.
- Add boosters whose trees read raw columns only (0.95116 with XGBoost, 0.95117 with LightGBM, rank correlation 0.997 with the top
  plan), or a LightGBM booster on every column (0.95147, correlation 0.998). The plain shallow trees differ more from the top plan
  (bags at 0.95128 and 0.95117, correlation 0.992), and each of these lowered a 3 to 1 rank mean from 0.95172 to between 0.95164 and
  0.95169.
- Ask the scan for all 10 pairs by default. Measured with all options on: 0.95139, 0.95141, 0.95132, no better than 8.
- Blend with a logistic stack on rank logits (0.95137 against 0.95155 for hill climbing on the same 24 plans), or add a
  scikit-learn neural net on the logistic's view (0.94336, weight 0).
- Key interactions between written forms. Of 2278 pairs of keys, 142 pass 4 standard deviations, and across fields the only
  real one is a pair the scan already keys. Decimals of one field against decimals of another look strong, 6 to 10 standard
  deviations, but shuffled labels give the same excess: about 1300 on 9801 degrees of freedom. That is the ratio's bias on
  sparse cells, with about one positive row per cell. Our scan avoids it here by cutting decimal columns in deciles.
- Train the language model public stacks rely on, at its size. One author reports about 120 GPU hours for a target-first GPT-2
  classifier, out of reach on one RTX 4060 before delivery. A smaller one of our own came close alone and added almost nothing
  (decision 0012). Blending other people's prediction files is what fills the top of this board, and the team's own score
  stays without them (decisions 0009 and 0011).
- Try cheap stand-ins for that language model. Each was measured in a rank mean with our top bag or our best mix (CV 0.95172 and
  0.95178), and each lowered it at every weight we tried, 5% to 30%. All of them read GPT-2 tokens or keys, and none needs a new
  dependency.

  | Stand-in | CV | Rank correlation with the top plan |
  |---|---|---|
  | Token n-grams counted per label, by field | 0.9392 | 0.92 |
  | The same over pairs of fields' first tokens | 0.9397 | 0.92 |
  | Sparse linear model over token n-grams | 0.94577 | 0.94 |
  | Sparse linear model over all 80 keys | 0.9464 | 0.95 |
  | CatBoost reading 30 keys as categories, alone | 0.94685 | 0.95 |
  | The same from the logistic start | 0.9512 | 0.996 |
  | Neural net over tokens and the logistic's view, eight seeds, numpy only, first fold only | 0.94925 (top mix there 0.95075) | 0.99 |

  So whatever the language model adds, these simpler readings of the same tokens do not catch it.
- Train on confident predicted rows of the other side. Holding out a fifth of the rows and adding the rows the model was surest of,
  labelled by its own guesses: 0.95046 fell to 0.95025 and 0.95015 on one holdout, and 0.95186 stayed the same on another.
- Add more seeds of shallow trees. A night of them moved the mix by less than 0.00001.

## How we would know

The public board decides, since refit and test side cannot show in CV. Read on 2026-11-06, it could not decide. Phase 1
candidates with CV from 0.95176 to 0.95199 all read between 0.95161 and 0.95177 on the public board (decision 0012). That spread
is under the 0.0002 the board can resolve, so it neither confirms nor refutes gains of this size. The private board, when the
competition closes on 2026-11-30, can.
