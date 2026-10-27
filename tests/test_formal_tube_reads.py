"""The tube's per-log spread, its field part at the plan's actions and the channel's grid region,
tied to their derivations.

``validation/tube_per_log_content.mac`` and ``tube_per_log_content.gp`` derive the spread over logs
of the share of paths a tube calibrated on average holds; ``validation/plan_action_tube.mac`` the
tube's field part carried through the plan's rollout as one vector;
``validation/channel_region_grid.mac`` the grid's members and radius of Simpson, Baumgartner,
Koehler and Diehl's Theorem 2; and ``validation/rk4_stage_term.mac`` the error each of RK4's
stages reads, which the tube carries by each stage's weight. The tests that check them on the
float code live beside the code's other tests; this file links them.
"""

from __future__ import annotations

FORMAL_LINKS = {
    "chc.decision._per_log_spread": {
        "cites": ("tube_per_log_content.mac", "tube_per_log_content.gp"),
        "tests": (
            "test_plan_action_tube.py::test_the_per_log_spread_is_its_closed_form",
            "test_plan_action_tube.py::test_the_per_log_spread_by_simulation",
        ),
    },
    "chc.decision._plan_field": {
        "cites": ("plan_action_tube.mac", "rk4_stage_term.mac"),
        "tests": (
            "test_rk4_stage_tube.py::test_the_stages_read_is_never_narrower_than_the_plan_s_form_under_rk4",
            "test_plan_action_tube.py::test_the_first_step_is_the_box_s_form_at_the_plan_s_action",
            "test_plan_action_tube.py::test_a_held_plan_from_rest_reads_close_to_the_box_s_form",
            "test_plan_action_tube.py::"
            "test_an_alternating_plan_is_covered_by_its_own_form_where_the_box_s_is_too_wide",
        ),
    },
    "chc.uncertainty._linear_tube": {
        "cites": ("rk4_stage_term.mac",),
        "tests": (
            "test_rk4_stage_tube.py::test_equal_stage_errors_carry_the_one_error_s_tube",
            "test_rk4_stage_tube.py::test_the_stages_read_is_never_narrower_than_the_plan_s_form_under_rk4",
        ),
    },
    "chc.dynamics_id._grid_members": {
        "cites": ("channel_region_grid.mac",),
        "tests": (
            "test_channel_region_grid.py::test_the_grid_s_members_are_their_closed_forms",
            "test_channel_region_grid.py::test_the_grid_s_radius_is_the_least_over_its_members",
        ),
    },
}
