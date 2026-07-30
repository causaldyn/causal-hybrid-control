# Contributing

This is a single-author research library. Issues and pull requests are welcome; please open an issue
before a large change, so we do not both build the same thing differently.

## Setup

```bash
uv sync --group dev
```

Python 3.12–3.14. `uv.lock` is committed and authoritative — run everything through `uv run`, never a
bare `python` / `pytest` / `ruff`, or you are testing a different dependency set from CI.

## The gates

CI runs exactly these, and a PR is expected to pass all of them locally first:

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest
```

The formal proofs are gated too. `proofs/*.v` must compile under **Rocq 9.2** (pinned: the proofs use
the post-rename `From Stdlib Require Import`, which does not exist before Rocq 9.0):

```bash
for f in proofs/*.v; do rocq compile -q "$f"; done
```

`tests/conftest.py` enables `jax_enable_x64`, so the suite runs in **float64** while a standalone
`uv run python` script runs float32. Numbers calibrated in one regime can fail in the other — if you
quote a measurement anywhere, produce it under the suite's settings.

## What a change should look like

- **New dependencies need discussion first.** The locked set is small on purpose, and optional
  backends (`tigramite`, `lightgbm`, …) stay optional and lazily imported.
- **Tests are two-sided.** A test that only shows the good case passing does not establish that the
  bad case fails; assert both, and prefer a test that fails when the implementation is reverted.
- **No number in a docstring, README or CHANGELOG that the test suite does not produce.** If a claim
  is measured, there should be a test or a script that regenerates it.
- **Scope claims precisely.** Where a result holds only for a scalar action, a control-affine plant,
  a linear smoother or a local neighbourhood, the docstring says so. Over-general claims are the one
  kind of bug this codebase treats as severe.
- Everything written — code, comments, docstrings, commit messages, issues — is in English.
- Commit messages use a conventional-commit prefix (`feat:` / `fix:` / `refactor:` / `test:` /
  `docs:` / `chore:`) and explain the *why* in the body.

## Out of scope

Reimplementations of estimators that `econml` / `dowhy` / `linearmodels` already provide well — this
library integrates them behind `chc.estimators` adapters rather than competing with them. The
contribution here is the decision layer: control, pessimism, certification, and the benchmark that
prices them.
