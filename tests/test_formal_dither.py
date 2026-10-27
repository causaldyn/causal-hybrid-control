"""The dither check's closed forms and the dither covariance's, tied to the tests that read them.

``validation/dither_hausman.mac`` derives the adjusted channel's bias under an omitted fresh
confounder, the contrast's null variance and its noncentrality, and the void case;
``validation/dither_covariance.mac`` derives the matrix geometric mean, its value, the binding
case at forty digits and the white dither's search bound. The tests that read them on the float code
live beside the code they test.
"""

from __future__ import annotations

FORMAL_LINKS = {
    "chc.decision.DitherCheck": {
        "cites": ("confounded_log_equivalence.mac", "dither_hausman.mac"),
        "tests": (
            "test_dither_check.py::test_a_planted_confounder_is_rejected",
            "test_dither_check.py::test_a_logger_of_the_state_and_its_draw_alone_is_void",
            "misspecification/test_misspec_dither.py::test_size_on_a_graph_that_is_right",
            "misspecification/test_misspec_dither.py::test_power_on_a_confounder_the_graph_omits",
        ),
    },
    "chc.experiment.dither_covariance": {
        "cites": ("dither_covariance.mac",),
        "tests": (
            "test_dither_covariance.py::test_the_two_lever_example",
            "test_dither_covariance.py::test_a_prior_below_the_optimum_is_subtracted",
            "test_dither_covariance.py::test_a_binding_prior_matches_the_forty_digit_solution",
            "test_dither_covariance.py::test_the_diagonal_case_is_water_filling",
        ),
    },
    "chc.experiment._white_dither": {
        "cites": ("dither_covariance.mac",),
        "via": "dither_covariance",
        "tests": (
            "test_dither_covariance.py::"
            "test_a_prior_with_a_zero_eigenvalue_gives_the_white_dither_without_a_warning",
            "test_dither_covariance.py::test_one_lever_s_white_dither_is_the_search_s_lower_bound",
        ),
    },
}
