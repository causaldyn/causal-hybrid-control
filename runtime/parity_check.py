"""Golden-parity check: the Rust runtime reproduces the Python ``chc`` controller exactly.

Build the extension and run this against the same golden LQ problem used by the Python and Rust tests:

    uvx maturin build --release --manifest-path runtime/Cargo.toml
    uv run --with runtime/target/wheels/chc_runtime-*.whl python runtime/parity_check.py

Expected: final cost 3.68619, first control -2.965208 (identical to ``chc``).
"""

from __future__ import annotations

import chc_runtime

A = [[0.0, 1.0], [-1.0, -0.2]]  # damped oscillator omega=1, zeta=0.1
B = [[0.0], [1.0]]
Q = [[1.0, 0.0], [0.0, 0.1]]
R = [[0.05]]
QF = [[5.0, 0.0], [0.0, 1.0]]


def main() -> None:
    us, history = chc_runtime.optimize_control(
        A, B, Q, R, QF, [0.0, 0.0], [1.0, 0.0], 0.1, -5.0, 5.0, 30, 400, 0.2
    )
    print(f"Rust runtime final cost = {history[-1]:.6f}   first u = {us[0][0]:.6f}")
    print("Python chc     final cost = 3.686190   first u = -2.965208")


if __name__ == "__main__":
    main()
