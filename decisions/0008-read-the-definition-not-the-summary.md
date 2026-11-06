---
title: Rebuild number keys, encoding seeds and the logistic model from their real definitions
status: accepted
created: 2026-11-04
decided: 2026-11-04
---

## Problem

After decision 0007 the Playground team stopped gaining. Its best models on `playground-series-s6e11` were
LightGBM 0.94997, XGBoost 0.94947 and logistic 0.94738, with a blend at 0.95022 and a public score of 0.95073.

The public ladder in decision 0006 uses `StratifiedKFold(5, shuffle=True, random_state=42)`, the same folds as our
validation plan, so its scores can be compared with ours directly. It reports 0.95031 for one model, 0.95051 after
seed bagging, 0.95027 for its GLM and 0.95084 for its stack. A gap of 0.0003 to 0.0029 on identical folds is not
noise, so something in our build was wrong.

We had built those ideas from a one-line summary in a discussion post. Reading the notebook behind the post
(Yash Teotia, "S6E11 | Beginner → Expert | Every Step Measured", Apache 2.0) showed three differences.

| Part | What we built | What it really is |
|---|---|---|
| Tenths key | The first decimal digit alone | The number cut to one decimal, 13713 for 1371.37 |
| Cents key | The last digit alone | The two decimals together, 37 for 1371.37 |
| Leading keys | The first digit alone | Digit count with the first digit, and digit count with the first two |
| Columns keyed | Any column with 50 or more values | Columns with decimals only |
| Seed bagging | The same inner split for every model, averaging encodings inside one model | Each model seed gets its own inner split, and blending the models averages the noise |
| Logistic model | Raw matrix: probabilities, raw keys and category codes | Encodings as log-odds, numbers on a log scale, categories one-hot |

## Decision

- `number_keys` replaces `digit_keys`. A new name keeps old ledger rows honest, because an old row with
  `digit_keys: true` did not measure the same thing, and the new name gives new experiment ids.
- A model's seed now picks its inner encoding splits, so seeds of one plan differ in encoding noise.
- The `logistic` family reads what a linear model can add up (`_logistic_view`).
- `target_encode_bags` stays as an option. CV decides, and on this data it lost in every comparison we ran.

We reimplemented these from the definitions in our own code. The ideas belong to the people credited in decision 0006.

## Measured on our folds, one seed

| Model | Before | After |
|---|---|---|
| LightGBM, depth 2 | 0.94997 | 0.95037 |
| XGBoost, depth 2 | 0.94947 | 0.95043 |
| Logistic | 0.94738 | 0.95018 |

The logistic gain has two parts: 0.94952 with log-odds and log scale, 0.95018 once categories are one-hot.

## One finding beyond the ladder

The notebook reports that 10 inner encoding folds lose to 5, because their noise regularizes. We measured the other
direction, one seed per cell, and it peaks at 3. So `target_encode_inner_folds` joined the vocabulary, default 5, and
the agents choose.

| Inner folds | XGBoost, depth 2 | LightGBM, depth 2 | Logistic |
|---|---|---|---|
| 2 | 0.95024 | 0.95044 | |
| 3 | 0.95064 | 0.95064 | 0.95020 |
| 4 | 0.95052 | | |
| 5 | 0.95043 | 0.95037 | 0.95018 |

At 3 folds XGBoost beat 5 folds on all five outer folds, and LightGBM on four of them.

## Alternatives measured and discarded

Each is a change to the XGBoost model above unless it says otherwise, against 0.95043.

- **Removing raw keys after encoding, as the notebook does:** 0.95032, within seed noise, so raw keys stay.
- **Adding the single digits back as two more keys:** 0.95027, so they stay out.
- **Smoothing 50 instead of 20:** 0.95046 here and 0.95024 against 0.95018 on logistic. Within noise, so one value
  for all families stays.

## Telling the team what changed

The first rounds on the rebuilt keys went badly for a reason that had nothing to do with the keys. Old lessons and
old experiments still compared everything against `digit_keys`, and one public score below the previous one made the
Analyst call the new keys suspect and ask to go back to an option that no longer existed. Three rounds went to
switching `number_keys` off, at about 0.948.

- **`--note`:** a person can tell the agents something from outside the run. A note replaces earlier lessons, stays
  in view, and is marked as the operator's in the ledger and the report. Prompts stay general, as decision 0007 asks,
  and what is particular to one competition goes in a note that anyone reading the ledger can see.
- **Retired options leave the agents' view.** Their experiments stay in the ledger and can still join a blend, but
  agents no longer compare against them.
- **Prompts:** the Analyst may not reverse a switch over one public result. The Tuner judges a switch over at least two
  seeds and tries lower learning rates on a winning plan. The Feature Engineer does not switch off a clear gain to
  measure it again.

## Limit worth stating

The agents could not have found this. They see scores and a closed vocabulary, not how a key is built. The Analyst
kept asking for new digit-key combinations, which was the right direction, but only a person comparing with an
outside result on the same folds could see that our keys were wrong. A check against public results on shared folds
would catch this class of mistake, and it is not built.

## How we would know it worked

The Playground blend passes 0.9507 on CV with agents choosing the plan, and the public score rises with it. The
ledger of `playground-series-s6e11` records both.

Half of that happened. With agents choosing, the blend reached 0.95098 on CV, and best single models rose to 0.95065. The
public score rose less, from 0.95073 to 0.95107, and then stopped following CV. Decision 0009 takes it from there.

After the operator's note, the Tuner did adopt the measured plan in its first round: learning rate 0.05, depth 2, three
inner folds and logistic in the batch. It then moved on its own to frequency encoding at the same cap and row statistics,
and over the next two rounds blend CV rose from 0.95082 to 0.95095.
