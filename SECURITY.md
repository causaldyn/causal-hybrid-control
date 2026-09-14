# Security policy

## Supported versions

| Version | Supported |
|---|---|
| 0.5.x | yes |
| < 0.5 | no |

This is a pre-1.0, single-author research library. Only the latest minor gets fixes; there are no
backports. If you are pinned to an older minor, the upgrade path is the [changelog](CHANGELOG.md),
which records every scope correction and every behaviour change, not only the additions.

## Reporting a vulnerability

Use GitHub's [private vulnerability reporting](https://github.com/causaldyn/causal-hybrid-control/security/advisories/new).
It is enabled on this repository, so the report stays private until a fix ships. Do not open a public
issue for a vulnerability.

Expect a first response within **7 days**. That is a single maintainer's honest number, not a
service level — if it matters to you commercially, say so in the report and say what you need.

Useful in a report: the version, the Python version, the minimal input that triggers it, and what an
attacker gains. A proof of concept is welcome; a CVSS score is not required.

## What is in scope

The threat model of a numerical library is narrower than a service's, and pretending otherwise
wastes your time:

- **In scope.** Code execution or file access reachable from data a caller passes in — a panel, an
  edge list, a lockfile-installed dependency chain, a model artefact. Unsafe deserialisation. A
  dependency with a known advisory that this package pins into place.
- **In scope, and treated as a bug of the same severity.** Any path where the library writes a
  secret, a token or a credential into a log record, an exception message or a `Provenance`. The
  logging added in 0.5.0 emits structured records from `chc.decision`; a field in one of those that
  leaks caller data beyond the column *names* it was given is a defect, report it as one.
- **Out of scope.** Wrong numbers. A miscalibrated certificate, an interval that does not cover, an
  estimator biased under an assumption it documents — these are correctness bugs, and they matter
  more to this project than most security bugs do, but they go in the public issue tracker where
  they can be argued about in the open.
- **Out of scope.** Denial of service by giving the library a large problem. It will use the memory
  the problem needs.
- **Out of scope.** Vulnerabilities in optional backends that are not installed by default
  (`tigramite`, `lightgbm`, …). Report those upstream; tell us if the pin is what exposes you.

## Supply chain

- Releases are published to PyPI by **Trusted Publishing** (OIDC), from `.github/workflows/release.yml`
  on an annotated `v*` tag. No API token exists to leak.
- Every artifact carries a **PEP 740 attestation** signed through Sigstore and logged in Rekor.
  Verify before you trust:

  ```bash
  gh attestation verify --repo causaldyn/causal-hybrid-control causal_hybrid_control-*.whl
  # or, straight from the index:
  curl -s https://pypi.org/integrity/causal-hybrid-control/0.5.1/causal_hybrid_control-0.5.1-py3-none-any.whl/provenance
  ```

  The publisher in a valid attestation is `causaldyn/causal-hybrid-control` via `release.yml`. A
  wheel for this project that carries no attestation, or one naming a different workflow, did not
  come from here.
- `uv.lock` is committed and authoritative, and Dependabot watches it.
- The sdist ships the proofs and derivations deliberately (`proofs/`, `validation/`), because the
  docstrings cite them. It ships no credentials, and `.github/workflows` is the only executable
  content in it that a build could run.
