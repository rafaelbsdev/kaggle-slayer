---
title: Predict test rows with every row's help, blend plans, and grow trees from a linear model
status: accepted
created: 2026-11-04
decided: 2026-11-04
---

## Problem

On `playground-series-s6e11` CV kept rising and the public board did not follow. From blend-r6 to blend-r30 CV rose from
0.94969 to 0.95095 while public moved from 0.95066 to 0.95096, rank 203 of 656. Public notebooks that train their own
models scored higher with a CV no better than ours: one reports CV 0.95084 and public 0.95154.

CV could not explain this, so we read what they do differently.

## Source

Public notebooks on the competition, read by us and reimplemented in our own code, with credit to:

- alex chilton, "Credit Card Fraud | LB 0.9521 | Shallow-Tree Stack". Its version history shows our exact pattern, CV up
  and board flat, until it changed how test rows are predicted.
- SIDHAARTH SHREE, "S6E11 | Highest Public LB 0.95154 | Self-Trained".
- Muhammad Ridwan Nasir Firdaus, "S6E11 | Digit-Trie TE | Joint GLM & Nested Stack".
- goodpjw2008 (their notebook series, Part 6) and megayak ("S6E11 | Per-Value + Digit TE | Shallow-Tree Stack"), for the GLM-boost
  recipe. Areeb Ali, for how the top public score is built.

## What we measured

All on our folds, which are the same as theirs.

| Idea | Reported in public | Measured here | Outcome |
|---|---|---|---|
| Refit on all rows, one seed | +0.00007 to +0.00041 on their boards | Holdout, LightGBM: fold models 0.95076, refit 0.95056, both 0.95081. Logistic: 0.95044, 0.95047, 0.95048 | Refit alone loses, both together win |
| Refit on all rows, more seeds | | LightGBM, three seeds: fold models 0.95077, refit 0.95079, both 0.95084. Two seeds: 0.95079, 0.95075, 0.95085 | Refit catches up with seeds, both stay ahead |
| Average a plan's seeds before weighing plans, search on every row | | Blend 0.95098 to 0.95105 on the same models | Adopted |
| Joint GLM with one coefficient per value | 0.9501 | 0.95012, so the definition is right, but weight 0 in the blend | Not added on its own |
| Encodings smoothed by 100 as a second view | 0.95072 as a bag | 0.95065 and 0.95043, no blend gain | Not added |
| Trees grown from the joint GLM's logit | 0.95042 to 0.95104 alone | 0.95066, weight 0.2, blend 0.95105 to 0.95114 | First built this way, then replaced |
| Trees grown from our dense logistic model's logit | | **0.95105**, far above every other model, weight 0.5, blend 0.95116 to 0.95126 | Added as `logistic_boost` |
| The same with LightGBM trees | Part of public GLM-boost banks | 0.95100 alone, rank correlation 0.9986, no blend gain | Not added |
| Trees from the linear model that see only raw columns and counts | 0.95100 alone, blend +0.00005 | 0.95034 alone, weight 0 | Not added |
| GPT-2 token keys | CV 0.95082 to 0.95101, and elsewhere no gain | LightGBM 0.95062 to 0.95074. `glm_boost` 0.95064 to 0.95054 alone but weight 0.2. Blend 0.95110 to 0.95118 | Added as `number_tokens` |
| How many rows share each key's value | Booster 0.95111 to 0.95137 | LightGBM 0.95062 to 0.95053, `glm_boost` 0.95064 to 0.95061 | Not added: our frequency encoding may already carry it |

The first two rows are the useful surprise. We expected the refit to win clearly and measured a loss at one seed, because
five fold models are already a bag and one refit is not. Seeds stop helping the fold models almost at once and keep helping
the refit, which catches up by three. Each comparison ran on all five outer folds, and at these sizes they differ by less
than the folds do, so we read them as small. We kept both sides because the pair led at every number of seeds.

Those holdouts train on 80% of the rows, not 100%, and they say nothing about the public board, where other notebooks report
larger gains. Only a public reading can settle that, and it is what the next submissions are for.

## Decision

- **Test side:** on random folds, test rows are ranked by the fold models and a refit on every row together. On time folds
  only the refit counts, as before, because only it saw the newest rows.
- **Blend:** seeds of a plan are averaged first, so the search weighs ideas and not lucky seeds. It runs on up to a million
  rows instead of 200k, over the 24 strongest plans.
- **`logistic_boost`:** the `logistic` model from decision 0008 is fitted first, and XGBoost trees grow from its logit, so
  they fit only what it misses. We first built this from the public recipe, on a sparse joint GLM of about 62,000 columns,
  and got 0.95066. The notebook behind the recipe reached 0.95104 from a GLM scoring 0.95047, while ours scored 0.95012, so
  we tried our stronger dense model as the start. That reached 0.95105 and let us delete the sparse design and splines. The
  old family's runs stay in the ledger and can join a blend, but agents no longer see them.
- **The gate's bar:** a submission records which blend made it, and only blends made the current way set the bar. On CloudWalk
  the new blend scored 0.75365 against 0.75419 for the old one, yet the old one could pick a lucky seed, and that luck carries
  into its cross-fit rows. The numbers are not comparable, and the test side changed as well.
- **Critic:** the prompt called submissions limited, and it held three blends in a row on CloudWalk with slots to spare, so
  the Analyst's questions never got an answer. It now knows that unused slots are lost and that a public reading is the only
  outside check on CV. The next blend it submitted scored 0.74226, up from 0.74076, and took rank 1 of 12.
- **Analyst:** it kept asking to submit single models, which the deterministic gate refuses. It now knows the gate.

## Alternatives discarded

- **Pooling public out-of-fold predictions, or rank-averaging public submission files.** This is what the top of the public
  board does: the best public notebook, 0.95231, is a fusion of other notebooks' files with weights tuned on the board. It
  would likely raise our rank more than anything above, and the top 1% of this board probably needs it. It would also stop
  being this team's work, and a fusion's weights cannot be validated. The project's owner weighed that trade and chose to
  submit only models this team trains, so our Playground rank may stop short of the top 5% and that is accepted.
- **Neural models.** They need `torch`, which is heavy, and public results run from weight 0 to +0.00012 in a stack. The
  owner approved `tokenizers` for the token keys, which also brought `huggingface-hub` and four smaller packages, but
  `torch` was not worth asking for.
- **CatBoost with native categories.** Public stacks report never picking CatBoost, so it was not worth a run.

## Limit worth stating

As in decisions 0006 and 0008, a person found these ideas and the agents only choose among them. Nothing in the loop can
notice that CV and the board have stopped moving together and go read why. An Analyst that flags that pattern would be the
next step, and it is not built.

## How we would know it worked

The public score rises past the 0.9510 plateau with CV near 0.9511, and the ledger of `playground-series-s6e11` records it.
The top 20% stood at 0.95159 when this was written.

It did. The team's next reading, with decision 0010 on top of this one, was CV 0.95168 and public 0.95151. The board moved too:
by the time the team's own models reached 0.95171, the top 20% needed 0.95173.
