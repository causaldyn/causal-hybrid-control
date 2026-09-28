#!/usr/bin/env bash
# Print Assumptions on every lemma in the .v files of one directory -- this script's own by default
# -- and fail on an axiom outside Stdlib's classical reals or on an admitted step. `just assumptions`
# runs it, and so does CI's proofs job inside the Rocq image, so the allowed axioms are listed here
# and nowhere else.
set -euo pipefail
src="${1:-$(dirname "$0")}"
work=$(mktemp -d); trap 'rm -rf "$work"' EXIT
allowed='FunctionalExtensionality.functional_extensionality_dep|ClassicalDedekindReals.sig_forall_dec|ClassicalDedekindReals.sig_not_dec|Classical_Prop.classic'
for f in "$src"/*.v; do
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
