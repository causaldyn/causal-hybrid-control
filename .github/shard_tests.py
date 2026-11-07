"""Print the test files of one CI shard: ``python .github/shard_tests.py <index> <count>``.

Each file under ``tests/`` goes, longest first, to the shard with the least time so far, by the
seconds ``test-seconds.json`` records for it on a CI leg (0 for a file whose tests the leg
deselects or skips). A file it does not list counts as the median, so a new file runs somewhere;
the record only balances the shards and can lag the suite.
"""

import json
import statistics
import sys
from pathlib import Path

index, count = map(int, sys.argv[1:])
root = Path(__file__).resolve().parent.parent
seconds = json.loads((root / ".github" / "test-seconds.json").read_text())
default = statistics.median(seconds.values())
files = sorted(p.relative_to(root).as_posix() for p in (root / "tests").rglob("test_*.py"))
load = [0.0] * count
mine = []
for f in sorted(files, key=lambda f: (-seconds.get(f, default), f)):
    shard = min(range(count), key=load.__getitem__)
    load[shard] += seconds.get(f, default)
    if shard == index:
        mine.append(f)
if not mine:
    sys.exit(f"shard {index} of {count} has no test files")
print(" ".join(mine))
