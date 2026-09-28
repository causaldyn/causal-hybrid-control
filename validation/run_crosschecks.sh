#!/usr/bin/env bash
# Run the SMT, PARI/GP and Octave cross-checks, and fail if any did not do what it states.
#
# Each SMT file names its verdicts in one line, `; Expected output: unsat, unsat.`, and z3's must
# match them in order: a `sat` where `unsat` was expected is a counterexample, and a file that
# states no expectation fails, so a new check cannot land unread. cvc5 runs the same files where it
# is installed. The PARI/GP and Octave files print numbers, not verdicts, so for them, as for the
# Maxima derivations, the check is that they complete. PARI/GP reports an error with `***` and
# carries on, so its output is read rather than its exit code.
set -uo pipefail

cd "$(dirname "$0")/.."
fail=0
total=0

verdicts() { grep -xE 'sat|unsat|unknown' | paste -sd, - | sed 's/,/, /g'; }

for f in validation/*.smt2; do
  expected=$(sed -nE 's/^; Expected output: (.*)\.$/\1/p' "$f")
  for solver in z3 cvc5; do
    command -v "$solver" >/dev/null || continue
    total=$((total + 1))
    case $solver in
      z3) got=$(timeout 300 z3 -T:120 "$f" 2>&1 | verdicts) ;;
      cvc5) got=$(timeout 300 cvc5 --incremental --tlimit=120000 "$f" 2>&1 | verdicts) ;;
    esac
    if [ -z "$expected" ] || [ "$got" != "$expected" ]; then
      fail=$((fail + 1))
      echo "FAIL $f ($solver): expected '${expected:-no Expected output line}', got '$got'"
    fi
  done
done

for f in validation/*.gp; do
  total=$((total + 1))
  out=$(timeout 300 gp -q -f "$f" </dev/null 2>&1)
  status=$?
  if [ "$status" -ne 0 ] || grep -q '\*\*\*' <<<"$out"; then
    fail=$((fail + 1))
    echo "FAIL $f"
    grep -n -B2 -A2 '\*\*\*' <<<"$out" | head -10 | sed 's/^/       /'
  fi
done

for f in validation/*.m; do
  total=$((total + 1))
  out=$(timeout 300 octave-cli --quiet --no-window-system "$f" 2>&1)
  status=$?
  if [ "$status" -ne 0 ]; then
    fail=$((fail + 1))
    echo "FAIL $f"
    tail -5 <<<"$out" | sed 's/^/       /'
  fi
done

echo "-----------------------------------------------------"
echo "$((total - fail))/$total complete"
exit $((fail > 0))
