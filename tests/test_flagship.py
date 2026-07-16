"""Flagship gate: causal control reaches the target; confounded predictive control diverges."""

from chc.flagship import run_flagship


def test_causal_control_reaches_target_naive_diverges() -> None:
    r = run_flagship()
    assert r["b_naive"] < 0.0 < r["b_causal"]  # the confounding sign flip
    assert abs(float(r["xs_causal"][-1]) - r["x_target"]) < 0.3  # causal reaches target
    assert abs(float(r["xs_naive"][-1]) - r["x_target"]) > 5.0  # naive diverges (catastrophe)
    # the naive controller pushes the state the wrong way relative to the causal one
    assert float(r["xs_naive"][-1]) < float(r["xs_causal"][-1])
