"""Golden-trajectory parity: the Rust binary and ``chc`` must do identical arithmetic.

Without this the latency numbers are meaningless -- two programs solving different problems can be
timed against each other all day. Run it before quoting any measurement:

    cargo build --release --manifest-path runtime/Cargo.toml
    uv run python runtime/parity_check.py

Compares three arms on the same LQ instance: the Rust binary, ``chc.control`` as the library ships
it, and the same recursion compiled into one XLA program. All three must agree to 1e-9.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

RUNTIME = Path(__file__).resolve().parent
BINARY = RUNTIME / "target" / "release" / "mpc-latency"
REFERENCE = RUNTIME / "mpc_latency.py"
TOLERANCE = 1e-9


def _read(lines: list[str]) -> dict[str, float]:
    return {key: float(value) for key, value in (line.split() for line in lines if line)}


def _run(command: list[str]) -> dict[str, float]:
    done = subprocess.run(command, capture_output=True, text=True, check=True, timeout=900)
    return _read(done.stdout.strip().splitlines())


def main() -> int:
    if not BINARY.exists():
        print(f"build the binary first: cargo build --release --manifest-path {RUNTIME}/Cargo.toml")
        return 2
    arms = {
        "rust": _run([str(BINARY), "golden"]),
        "chc.control": _run([sys.executable, str(REFERENCE), "golden"]),
        "chc compiled": _run([sys.executable, str(REFERENCE), "golden-jit"]),
    }
    reference = arms["rust"]
    worst = 0.0
    for name, arm in arms.items():
        for key in ("cost", "u0"):
            gap = abs(arm[key] - reference[key])
            worst = max(worst, gap)
            print(f"{name:14s} {key:5s} {arm[key]:+.9f}  gap {gap:.2e}")
    print(f"worst gap {worst:.2e}  tolerance {TOLERANCE:.0e}")
    return 0 if worst <= TOLERANCE else 1


if __name__ == "__main__":
    raise SystemExit(main())
