---
title: Agents never see competition rows
status: accepted
created: 2026-11-03
decided: 2026-11-03
---

## Problem

CloudWalk's rules (section 2.4.b) forbid transmitting Competition Data to anyone who has not
accepted the rules. An LLM provider is such a party, and the data is CC BY-NC.

## Decision

- Agents receive only derived statistics: row counts, target rate, null rates, cardinality,
  drift and importance scores, and our own experiment metrics.
- No min, max or quantiles either: each of those can be a literal value from a row.
- `slayer.privacy.assert_no_rows` runs on every payload before it leaves the machine, and a test
  plants sentinel values in the data to prove they never reach a payload.
- `data/`, `artifacts/`, `submissions/` and all CSV, Parquet and NumPy files are git-ignored.

## Alternatives discarded

- **Send a sample of rows to help the LLM reason:** breaks the rule, however small the sample.
- **Let agents write and run their own pandas code:** that code would read rows and could print
  them back into the conversation. See decision 0003.

## How we would know it worked

`tests/test_privacy.py` fails if any sentinel from the synthetic data reaches a payload.
