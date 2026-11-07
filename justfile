# Verification loop for this project. `just check` runs the four commands ci.yml's `lint` and
# `test` jobs run, in the same order, so a green check here is a green CI -- the tests on the CPU
# as there, spread over worker processes where CI runs them in one. `just all` adds the two gates
# CI keeps in separate jobs because they need Rocq and Maxima rather than Python. CI invokes the
# commands directly rather than through `just`, so that a runner needs no extra tooling -- if a
# recipe below and ci.yml ever disagree, ci.yml is the one that ships.

# JAX's build for this machine's accelerator, as the pyproject extra that installs it: CUDA 13
# where the NVIDIA driver is 580 or newer and the GPU of compute capability 7.5 or newer, CUDA 12
# from driver 525, none otherwise. `sync` and `test` install it into `.venv`; CI never does.
# Override with `just accelerator=cuda12 test`, or `just accelerator= test` for none.
accelerator := `timeout 10 nvidia-smi --query-gpu=driver_version,compute_cap --format=csv,noheader 2>/dev/null | awk -F', ' 'NR == 1 { if ($1 + 0 >= 580 && $2 + 0 >= 7.5) print "cuda13"; else if ($1 + 0 >= 525) print "cuda12" }'`
extra := if accelerator == "" { "" } else { "--extra " + accelerator }

default:
    @just --list

# The Python ladder, cheapest first. Stops at the first failure.
check: fmt lint types test-cpu

# `.venv` exactly as the lock has it, with this machine's accelerator build. A plain `uv sync`
# removes the build again; `uv run` leaves it in place.
sync:
    uv sync {{extra}}

# Everything, including the formal and symbolic gates. Minutes, not seconds.
all: check types-matrix proofs assumptions derivations crosschecks

# ── Python (uv + ruff + ty) ───────────────────────────────────────────────────

fmt:
    uv run ruff format --check .

lint:
    uv run ruff check .

types:
    uv run ty check

# addopts already carries -q; a second one suppresses the summary line entirely. The tests run
# over `workers` pytest-xdist processes, each file in one of them: a worker keeps the memory of
# every program it compiled, so `--dist loadfile` compiles a file's programs and runs its module
# fixtures once rather than in every worker, and the suite's memory barely depends on how many
# workers share it (two ended at 6.0 and 5.6 GB). `just test-cpu 0` runs one process on the CPU,
# as CI does; a test that passes there and fails across workers reads another test's state.
# `test` runs on the device the accelerator build gives jax -- the GPU, where there is one, in the
# float64 conftest.py sets -- and `test-cpu` on the CPU CI runs on, whatever is installed.
test workers="4":
    uv run {{extra}} pytest -n {{workers}} --dist loadfile

test-cpu workers="4":
    JAX_PLATFORMS=cpu uv run pytest -n {{workers}} --dist loadfile

# The certificates read on plants their models get wrong (ADR 0115). pyproject.toml's addopts
# deselects them from `test` and `test-cpu`; the `-m` here comes later and wins. CI runs them in a
# job of their own, on four workers.
misspecification workers="2":
    JAX_PLATFORMS=cpu uv run pytest -m misspecification -n {{workers}} --dist loadfile tests/misspecification

# The wrong certificates of tests/mutants/chc_mutants.py, each planted alone on the misspecification
# file that reads it, over `blocks` seed blocks: a file that fails on four in five of them or more
# caught its mutant. A file whose seeds no block moves runs once. About two hours at five blocks.
# Plants each wrong certificate on the misspecification file that reads it: caught, or MISSED.
mutants blocks="5":
    #!/usr/bin/env bash
    set -uo pipefail
    export JAX_PLATFORMS=cpu PYTHONPATH=tests/mutants
    for case in tube:0.85/tube channel:0.8/channel ope:0.8/ope regret:0.1/regret \
        dither:0.33/dither lag:0.33/lag lift:0.8/lift gate:4/gate barrier:1/barrier; do
      mutant=${case%/*} file=tests/misspecification/test_misspec_${case#*/}.py
      blocks={{blocks}}
      grep -q _misspec_counts "$file" || blocks=1
      failed=0
      for block in $(seq 0 $((blocks - 1))); do
        CHC_MUTANT=$mutant CHC_MISSPEC_BLOCK=$block uv run pytest -p chc_mutants -p no:cacheprovider \
          -m misspecification -x -q "$file" > /dev/null 2>&1
        status=$?
        [ $status -gt 1 ] && { echo "$mutant: pytest stopped with status $status on block $block"; exit 1; }
        failed=$((failed + status))
      done
      [ $((5 * failed)) -ge $((4 * blocks)) ] && verdict=caught || verdict=MISSED
      echo "$mutant on $file: failed on $failed of $blocks blocks, $verdict"
    done

# The commit `just wheel` builds; `just rev=v0.24.0 wheel` builds that tag.
rev := "HEAD"

# release.yml's `wheel` job, here: the wheel and sdist `uv build` makes of a commit, the wheel
# installed beside the locked test dependencies in a copy of that commit whose src/ is removed, and
# the suite run against it. A module or a file the build leaves out passes every run from the
# checkout and fails here, as on 0.24.0's tag. The copy and its environment, a few GB, go under
# TMPDIR, /var/tmp when that is unset: on a tmpfs /tmp they would sit in memory.
# Builds the commit `rev`, not the working tree, and hands its arguments to pytest.
[positional-arguments]
wheel *args:
    #!/usr/bin/env bash
    set -euo pipefail
    unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
    work=$(mktemp -d "${TMPDIR:-/var/tmp}/chc-wheel.XXXXXX"); trap 'rm -rf "$work"' EXIT
    git archive "{{rev}}" | tar -x -C "$work"
    cd "$work"
    uv build --python 3.11
    uv sync --frozen --python 3.14 --group dev --no-install-project
    uv pip install --no-deps dist/*.whl
    uv pip check
    rm -r src
    uv run --no-sync pytest "$@"

fix:
    uv run ruff check --fix .
    uv run ruff format .

# `just types` checks one interpreter, and that is not enough here. uv.lock resolves jax 0.10.2
# below Python 3.12 and 0.11 at or above it, and 0.11 declares `Config.jax_enable_x64` where
# 0.10 injects it -- so a green local `ty` on 3.14 was a red CI job on 3.11, with the failure
# living in a dependency's own class definition rather than in this code. This runs the same
# matrix ci.yml's `lint` does, each in its own environment so `.venv` is not swapped underneath
# you; a free-threaded build resolves as its twin with the GIL does, so it adds no leg here, and
# each leg asks for the twin with the GIL: uv may otherwise settle on the free-threaded one, which
# no abi3 wheel serves, and build polars from source. Minutes, not seconds: run it before pushing
# anything that touches a dependency's API.
types-matrix:
    #!/usr/bin/env bash
    set -euo pipefail
    for v in 3.11 3.12 3.13 3.14 3.15; do
      echo "== ty on $v =="
      UV_PROJECT_ENVIRONMENT=".venv-ty-$v" uv run --python "$v+gil" --group dev ty check --python-version "$v"
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

# ── Fuzzing ───────────────────────────────────────────────────────────────────

# The corpus and any crash stay in .fuzz/; atheris is hash-pinned in fuzz/requirements.txt.
# Fuzz one target, `graph` or `response`, for `seconds`.
fuzz target="graph" seconds="60":
    mkdir -p .fuzz/{{target}}
    uv run --with-requirements fuzz/requirements.txt python fuzz/fuzz_{{target}}.py .fuzz/{{target}} -max_total_time={{seconds}} -max_len=64 -len_control=0 -artifact_prefix=.fuzz/

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

# Execute every notebook in place, so the outputs GitHub shows are this checkout's. JAX_PLATFORMS
# stays unset: the pin each notebook makes is then what keeps JAX's accelerator probe out of them.
# Needs `uv sync --group notebooks`.
notebooks:
    #!/usr/bin/env bash
    set -euo pipefail
    unset JAX_PLATFORMS JAX_ENABLE_X64
    for notebook in notebooks/*.ipynb; do
      timeout 1800 uv run --group notebooks jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=1200 "$notebook"
    done
