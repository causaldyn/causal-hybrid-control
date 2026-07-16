"""Confounding demo: the same offline data, two opposite conclusions about the action's effect.

Run: ``uv run python scripts/confounding_demo.py``. This is the text seed of the
"causal != predictive for control" figure (see ``plans/05``).
"""

from __future__ import annotations

import jax

from chc.causal import ConfoundedLinearSystem, estimate_control_effect


def main() -> None:
    system = ConfoundedLinearSystem()
    data = system.sample(20_000, jax.random.key(0))
    b_naive = float(estimate_control_effect(data, adjust_for=()))
    b_causal = float(estimate_control_effect(data, adjust_for=("z",)))

    print(f"true causal effect of u    : {system.b_true:+.3f}")
    print(f"naive fit (no adjustment)  : {b_naive:+.3f}   <- confounded by z")
    print(f"causal fit (adjust for z)  : {b_causal:+.3f}   <- recovers the truth")
    print()
    print(f"to raise the outcome you should {'raise' if b_causal > 0 else 'lower'} u (correct).")
    print(f"the naive model says {'raise' if b_naive > 0 else 'lower'} u  -> wrong direction.")


if __name__ == "__main__":
    main()
