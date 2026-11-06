---
title: Delivering code while the competitions are open
status: accepted
created: 2026-11-03
decided: 2026-11-05
---

## Problem

Kaggle's rules on both competitions forbid sharing competition code privately during the
competition, and require public sharing to happen on the competition's forum or notebooks. The
brief asks for delivery via GitHub on 2026-11-06, while both competitions are open.

## Options

1. Private repository shared only with CloudWalk. CloudWalk is the host of one competition, but
   the Playground rule has no exception for this.
2. Public repository, with a link posted on both competition forums. Explicitly allowed, but it
   publishes the solution to CloudWalk's recruiting competition.

## Decision

Option 2. The owner chose a public repository while CloudWalk's answer was still pending, because it is the option the rules
explicitly allow.

## Consequences

- The owner publishes on delivery, on the morning of 2026-11-06, and posts the link on the discussion forum of both competitions.
  The rules require that for public sharing, so a public repository alone is not enough.
- The rules treat shared code as licensed under an open source license with no limit on commercial use, so the repository
  carries the MIT license.
- Rivals can read and reuse this work while both competitions are open, including CloudWalk's. That cannot be undone.
- Nothing from the competitions' data may be in the repository. `data/`, `artifacts/`, `submissions/` and every `.csv` and `.npy` are
  ignored. Before publishing we searched the files to be committed for numbers matching the data, and removed one real value
  that had reached a docstring and some tests through a public notebook.
