# Verification loop for this project. `just check` runs exactly the four commands ci.yml's
# `lint` and `test` jobs run, in the same order, so a green check here is a green CI; `just all`
# adds the two gates CI keeps in separate jobs because they need Rocq and Maxima rather than
# Python. CI invokes the commands directly rather than through `just`, so that a runner needs
# no extra tooling -- if a recipe below and ci.yml ever disagree, ci.yml is the one that ships.

default:
    @just --list

# The Python ladder, cheapest first. Stops at the first failure.
check: fmt lint types test

# Everything, including the formal and symbolic gates. Minutes, not seconds.
all: check proofs assumptions derivations

# ── Python (uv + ruff + ty) ───────────────────────────────────────────────────

fmt:
    uv run ruff format --check .

lint:
    uv run ruff check .

types:
    uv run ty check

# addopts already carries -q; a second one suppresses the summary line entirely.
test:
    uv run pytest

fix:
    uv run ruff check --fix .
    uv run ruff format .

# ── Rocq ──────────────────────────────────────────────────────────────────────

# Compile every proof in a scratch directory: `rocq compile` writes .vo/.glob next to its
# input, and the repo does not carry build artefacts.
proofs:
    #!/usr/bin/env bash
    set -euo pipefail
    work=$(mktemp -d); trap 'rm -rf "$work"' EXIT
    cp proofs/*.v "$work"/
    cd "$work"
    for f in *.v; do timeout 900 rocq compile -q "$f"; done
    echo "compiled $(ls -1 *.vo | wc -l) proofs"

# A compiled proof is not a finished proof: Print Assumptions is what reveals an axiom or an
# admitted step holding a result up. Every lemma here is stated over R, and Rocq's classical
# reals are themselves axiomatic, so "Closed under the global context" is unreachable and would
# be the wrong gate. What is checked instead is that nothing OUTSIDE Stdlib's own four axioms
# gets in -- a project Axiom, an admitted step, or a Hypothesis leaking out of its Section.
# Measured 2026-09-03 over 386 lemmas in 55 files: functional_extensionality_dep and
# sig_forall_dec on all of them, sig_not_dec on 36, classic on 9, and nothing else.
assumptions:
    #!/usr/bin/env bash
    set -euo pipefail
    work=$(mktemp -d); trap 'rm -rf "$work"' EXIT
    allowed='FunctionalExtensionality.functional_extensionality_dep|ClassicalDedekindReals.sig_forall_dec|ClassicalDedekindReals.sig_not_dec|Classical_Prop.classic'
    for f in proofs/*.v; do
      names=$(grep -oP '^\s*(Lemma|Theorem|Corollary|Proposition)\s+\K[A-Za-z_][A-Za-z0-9_'"'"']*' "$f" || true)
      [ -z "$names" ] && continue
      probe="$work/$(basename "$f")"
      cp "$f" "$probe"
      for n in $names; do echo "Print Assumptions $n." >> "$probe"; done
      timeout 900 rocq top -batch -load-vernac-source "$probe" >> "$work/all.out" 2>&1 || true
    done
    # Axiom names are the lines that start a fresh declaration inside a Print Assumptions block.
    stray=$(grep -oE '^[A-Za-z][A-Za-z0-9_.]*[[:space:]]*:' "$work/all.out" | sed 's/[[:space:]]*:$//' \
            | sort -u | grep -vE "^($allowed|Axioms)$" || true)
    lemmas=$(grep -c '^Axioms:' "$work/all.out")
    if [ -n "$stray" ]; then
      echo "assumptions outside Stdlib's classical reals:"; echo "$stray" | sed 's/^/       /'; exit 1
    fi
    grep -q 'Admitted' "$work/all.out" && { echo "an Admitted proof reached the batch"; exit 1; } || true
    echo "$lemmas lemmas rest on Stdlib's classical-reals axioms and nothing else"

# ── Maxima ────────────────────────────────────────────────────────────────────

# `maxima -b` exits 0 on a parse error, so the runner greps the output instead.
derivations:
    ./validation/run_all.sh

# ── The numbers other documents quote ─────────────────────────────────────────

# Counts drift, and several documents have quoted stale ones. This is the authority.
counts:
    #!/usr/bin/env bash
    set -euo pipefail
    log=../../discoveries/theorems.md
    printf '%-22s %s\n' modules "$(ls src/chc/*.py | wc -l)"
    printf '%-22s %s\n' "public names" "$(grep -cE '^    "' src/chc/__init__.py)"
    printf '%-22s %s\n' "rocq proofs" "$(ls proofs/*.v | wc -l)"
    printf '%-22s %s\n' "maxima derivations" "$(ls validation/*.mac | wc -l)"
    printf '%-22s %s\n' "test files" "$(ls tests/*.py | wc -l)"
    printf '%-22s %s\n' "collected tests" "$(uv run pytest --collect-only -q 2>/dev/null | awk '{s+=$2} END {print s}')"
    [ -f "$log" ] && printf '%-22s %s\n' "tabled results" "$(grep -c '^## Result' "$log")" || true

# ── Benchmarks ────────────────────────────────────────────────────────────────

bench:
    uv run python scripts/run_benchmark.py

flagship:
    uv run --group viz python scripts/flagship_demo.py
