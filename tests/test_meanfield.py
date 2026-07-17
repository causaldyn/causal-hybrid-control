"""Mean-field gate: planning that anticipates the migration lag beats a myopic controller."""

from chc.meanfield import MeanFieldControl


def test_meanfield_planning_beats_myopic_under_migration_lag() -> None:
    regrets = MeanFieldControl(horizon=14).regrets(steps=200)
    assert regrets["planned-CHC"] < regrets["myopic"]  # anticipating the lag beats reacting to it
    assert regrets["myopic"] < regrets["no-control"]  # some control beats none
