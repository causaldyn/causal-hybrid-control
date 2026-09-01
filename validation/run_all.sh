#!/usr/bin/env bash
# Run every Maxima derivation and fail if any of them did not complete.
#
# `maxima -b` exits 0 even when the batch aborts on a parse error, and exits 0 after an
# interactive question hits EOF -- it also still echoes the batch filename on the way out.
# So neither the exit code nor the last line is a check; the output is. Before this runner
# existed, three derivations had been dead since their first commit.
set -uo pipefail

cd "$(dirname "$0")/.."
outdir="${1:-$(mktemp -d)}"
mkdir -p "$outdir"

broken='incorrect syntax|-- an error\.|Lisp error|^MAXIMA>'
fail=0
total=0

for f in validation/*.mac; do
  total=$((total + 1))
  name=$(basename "$f" .mac)
  timeout 900 maxima --very-quiet -b "$f" >"$outdir/$name.out" 2>&1
  if grep -qE "$broken" "$outdir/$name.out"; then
    fail=$((fail + 1))
    echo "FAIL $f"
    grep -nE "$broken" "$outdir/$name.out" | head -3 | sed 's/^/       /'
  fi
done

echo "-----------------------------------------------------"
echo "$((total - fail))/$total complete    (output in $outdir)"
exit $((fail > 0))
