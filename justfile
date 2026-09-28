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
all: check types-matrix proofs assumptions derivations crosschecks

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

# `just types` checks one interpreter, and that is not enough here. uv.lock resolves jax 0.10.2
# below Python 3.12 and 0.11.0 at or above it, and 0.11 declares `Config.jax_enable_x64` where
# 0.10 injects it -- so a green local `ty` on 3.14 was a red CI job on 3.11, with the failure
# living in a dependency's own class definition rather than in this code. This runs the same
# matrix ci.yml does, each in its own environment so `.venv` is not swapped underneath you.
# Minutes, not seconds: run it before pushing anything that touches a dependency's API.
types-matrix:
    #!/usr/bin/env bash
    set -euo pipefail
    for v in 3.11 3.12 3.13 3.14; do
      echo "== ty on $v =="
      UV_PROJECT_ENVIRONMENT=".venv-ty-$v" uv run --python "$v" --group dev ty check
    done

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
    ./proofs/assumptions.sh

# ── Rocq + MathComp ───────────────────────────────────────────────────────────

# The matrix lifts in proofs/mathcomp/ need MathComp 2.6, which Fedora does not package, so they
# compile in the opam switch `chc-mathcomp` (CI: mathcomp/mathcomp:2.6.0-rocq-prover-9.2). The
# system rocq never sees them: `just proofs` globs proofs/*.v, which does not descend.
proofs-mathcomp:
    #!/usr/bin/env bash
    set -euo pipefail
    export PATH="$HOME/.local/bin:$PATH"
    work=$(mktemp -d); trap 'rm -rf "$work"' EXIT
    cp proofs/mathcomp/*.v "$work"/
    cd "$work"
    for f in *.v; do timeout 900 opam exec --switch=chc-mathcomp -- rocq compile -q "$f"; done
    echo "compiled $(ls -1 *.vo | wc -l) MathComp proofs"

# These are stated over abstract MathComp fields, not over Stdlib's axiomatic reals, so here
# "Closed under the global context" IS reachable, and it is the gate: every lemma must print it,
# and one axiom of any kind -- a project Axiom, an admitted step, a Section hypothesis leaking out
# of its Section -- fails the recipe.
assumptions-mathcomp:
    #!/usr/bin/env bash
    set -euo pipefail
    export PATH="$HOME/.local/bin:$PATH"
    work=$(mktemp -d); trap 'rm -rf "$work"' EXIT
    expected=0
    for f in proofs/mathcomp/*.v; do
      names=$(grep -oP '^\s*(Lemma|Theorem|Corollary|Proposition)\s+\K[A-Za-z_][A-Za-z0-9_'"'"']*' "$f" || true)
      [ -z "$names" ] && continue
      probe="$work/$(basename "$f")"
      cp "$f" "$probe"
      for n in $names; do echo "Print Assumptions $n." >> "$probe"; expected=$((expected + 1)); done
      if ! (cd "$work" && timeout 900 opam exec --switch=chc-mathcomp -- rocq compile -q "$(basename "$f")") \
          >> "$work/all.out" 2>&1; then
        cat "$work/all.out"; exit 1
      fi
    done
    closed=$(grep -c '^Closed under the global context$' "$work/all.out" || true)
    if [ "$closed" -ne "$expected" ] || grep -qE '^Axioms:|Admitted' "$work/all.out"; then
      cat "$work/all.out"; echo "$closed of $expected lemmas closed under the global context"; exit 1
    fi
    echo "$closed lemmas closed under the global context"

# ── Maxima ────────────────────────────────────────────────────────────────────

# `maxima -b` exits 0 on a parse error, so the runner greps the output instead.
derivations:
    ./validation/run_all.sh

# The SMT files' verdicts against the ones each states, and the PARI/GP and Octave files for
# completion. Seconds. Needs z3, gp and octave-cli, and runs cvc5 too where it is installed.
crosschecks:
    ./validation/run_crosschecks.sh

# ── The numbers other documents quote ─────────────────────────────────────────

# Counts drift, and several documents have quoted stale ones. This is the authority.
counts:
    #!/usr/bin/env bash
    set -euo pipefail
    log=../../discoveries/theorems.md
    printf '%-22s %s\n' modules "$(ls src/chc/*.py | wc -l)"
    printf '%-22s %s\n' "public names" "$(grep -cE '^    "' src/chc/__init__.py)"
    printf '%-22s %s\n' "rocq proofs" "$(ls proofs/*.v | wc -l)"
    printf '%-22s %s\n' "rocq lemmas" "$(grep -ohE '^[[:space:]]*(Lemma|Theorem|Corollary|Proposition)[[:space:]]+[A-Za-z_][A-Za-z0-9_]*' proofs/*.v | wc -l)"
    printf '%-22s %s\n' "maxima derivations" "$(ls validation/*.mac | wc -l)"
    printf '%-22s %s\n' "test files" "$(ls tests/*.py | wc -l)"
    printf '%-22s %s\n' "collected tests" "$(uv run pytest --collect-only -q 2>/dev/null | awk '{s+=$2} END {print s}')"
    [ -f "$log" ] && printf '%-22s %s\n' "tabled results" "$(grep -c '^## Result' "$log")" || true

# ── Benchmarks ────────────────────────────────────────────────────────────────

bench:
    uv run python scripts/run_benchmark.py

flagship:
    uv run --group viz python scripts/flagship_demo.py

# ── Docs ──────────────────────────────────────────────────────────────────────

# Execute the notebooks and the scripts the pages quote, then `mkdocs build --strict` into site/.
# Needs `uv sync --group docs --group notebooks`.
docs:
    ./docs/build.sh
