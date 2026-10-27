"""The projected Newton rounds' matrix, read on the code that builds it.

``validation/projected_newton_round.mac`` shows on a plant linear in the state and the action that
the cost's Gauss-Newton matrix is the cost's Hessian, that its diagonal is the planner's curvature
``r + B' P B``, and that the penalty's Hessian on affine columns is ``rho sum g g'``. The tests read
the same statements on :func:`chc.plan._round_terms` and :func:`chc.plan._round_metric`.
"""

from __future__ import annotations

FORMAL_LINKS = {
    "chc.plan._round_terms": {
        "cites": ("projected_newton_round.mac",),
        "tests": (
            "test_held_barrier_newton.py::test_the_round_matrix_is_the_round_s_hessian_on_a_linear_plant",
            "test_held_barrier_newton.py::test_the_round_matrix_s_diagonal_is_the_diagonal_metric_s_square",
        ),
    },
}
