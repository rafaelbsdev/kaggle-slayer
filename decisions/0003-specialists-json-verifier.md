---
title: Specialist agents emit validated JSON, the orchestrator does all the mechanics
status: accepted
created: 2026-11-03
decided: 2026-11-03
---

## Problem

An LLM that downloads, trains and submits by itself can invent a CV score, leak the target or burn
the daily submission quota, and nobody would see it happen.

## Decision

- Six specialists, each with one job and one Pydantic contract: Scout, Validation Strategist,
  Feature Engineer, Model Tuner, Submission Critic, Leaderboard Analyst.
- The orchestrator runs every mechanical step: download, Parquet, training, blending, submitting,
  reading the leaderboard. Agents cannot touch files or the network.
- A verifier sits between every pair of steps, cheapest layer first: schema, deterministic checks,
  semantic check with a cheap model. It never fixes anything. A rejected output goes back to the
  same agent with the reason, and after the retries the run uses a safe heuristic and logs it.
- Scores are observed, never declared. CV comes from our training code and public score from Kaggle.
- Features and parameters come from a closed vocabulary, so a bad idea costs one experiment.

## Alternatives discarded

- **LangGraph, CrewAI or similar:** the flow is a short deterministic loop, and those frameworks
  add dependencies without adding checks.
- **Agents that write and execute code:** more flexible, but breaks decision 0002 and needs a
  sandbox we cannot build and audit by Friday.
- **Claude Managed Agents:** the sandbox would be hosted, and competition data cannot leave the
  machine.

## Known limit

Only tabular binary classification scored by AUC is supported. The Scout rejects anything else
with a clear reason, instead of pretending. Both target competitions fit.

## How we would know it worked

The offline end-to-end test runs the whole loop against a fake Kaggle, and mutation runs confirm
the tests fail when each verifier check is removed.
