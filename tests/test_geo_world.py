"""chc.geo_world: a geo media-mix world drawn from a stated hierarchy.

Each geo's cells lift what its media lift at home, and a lift per head is the curve per head of
the spend per head, computed here apart. The geos' effects spread about the national median as
stated; national media is allotted by population; the log's spend rises with the season by the
policy; a geo's own media spill the stated share into its two neighbours and nowhere else; the
drift is a national random walk; the noise is each head's. One seed draws one world, and mixes that
differ in a parameter draw the same variates. The world's cells plan with its history.
"""

import numpy as np
import pytest

from chc.allocation import allocate_geos
from chc.geo_world import GeoMediaMix, MediaChannel
from chc.response import Hill, MichaelisMenten, Power, Tanh

CHANNELS = (
    MediaChannel("search", MichaelisMenten(2.0), 3.0, spread=0.3, retention=0.3, spend=1.5),
    MediaChannel("social", Hill(1.0, 2.0), 1.5, spread=0.5, retention=0.5, spend=0.8),
    MediaChannel("tv", Tanh(4.0), 2.0, spread=0.2, retention=0.7, spend=2.0, national=True),
)
GEOS, WEEKS = 12, 60
MIX = GeoMediaMix(CHANNELS, geos=GEOS, weeks=WEEKS, policy=0.4, spillover=0.3, drift=0.02)
SEASON = np.sin(2.0 * np.pi * np.arange(WEEKS) / 52.0)


@pytest.fixture(scope="module")
def world():
    return MIX.draw(7)


def test_one_seed_draws_one_world_and_another_seed_another(world):
    again = MIX.draw(7)
    for name in ("population", "effect", "drift", "spend", "base", "media", "spilled", "noise"):
        np.testing.assert_array_equal(getattr(again, name), getattr(world, name))
    assert not np.array_equal(MIX.draw(8).kpi, world.kpi)


def test_each_geo_s_cells_lift_what_its_media_lift_at_home(world):
    """Through `chc.response`'s own kernel and curve, at the drift of the week they are read at."""
    for week in (0, 30, WEEKS - 1):
        for g, row in enumerate(world.cells(week)):
            for c, cell in enumerate(row):
                lifted = np.asarray(cell(world.spend[g, :, c]))[week]
                assert lifted == pytest.approx(world.media[g, week, c], rel=1e-12)


def test_a_lift_per_head_is_the_curve_per_head_of_the_spend_per_head():
    """Michaelis-Menten at a scale of 2 a head lifts ``a / (2 + a)`` of its ceiling, ``a`` the
    spend per head through the normalised kernel: a geo's size cancels."""
    channel = MediaChannel("only", MichaelisMenten(2.0), 3.0, spread=0.4, retention=0.5)
    drawn = GeoMediaMix((channel,), geos=5, weeks=30, length=4).draw(3)
    weights = 0.5 ** np.arange(4) / np.sum(0.5 ** np.arange(4))
    for g in range(5):
        adstock = np.convolve(drawn.spend[g, :, 0] / drawn.population[g], weights)[:30]
        expected = drawn.effect[g, 0] * adstock / (2.0 + adstock)
        np.testing.assert_allclose(drawn.media[g, :, 0] / drawn.population[g], expected, rtol=1e-12)


def _log_normal_about(values, median, spread) -> bool:
    """Within five standard errors of a log-normal sample's median and log-SD."""
    logs = np.log(values)
    n = logs.size
    return bool(
        abs(logs.mean() - np.log(median)) < 5.0 * spread / np.sqrt(n)
        and abs(logs.std(ddof=1) - spread) < 5.0 * spread / np.sqrt(2.0 * n)
    )


def test_the_geos_spread_about_the_mix_s_medians_as_stated():
    """Each channel's effect about its national median, and a geo's population, base per head
    and spend per head on a channel it buys itself about the mix's, read in week 0, where the
    season is 0, with no jitter."""
    mix = GeoMediaMix(
        CHANNELS, geos=4000, weeks=2, population_spread=0.8, base_spread=0.3, spend_noise=0.0
    )
    drawn = mix.draw(11)
    for c, channel in enumerate(CHANNELS):
        assert _log_normal_about(drawn.effect[:, c], channel.effect, channel.spread)
    assert _log_normal_about(drawn.population, 1e5, 0.8)
    assert _log_normal_about(drawn.base[:, 0] / drawn.population, 1.0, 0.3)
    for c in (0, 1):
        per_head = drawn.spend[:, 0, c] / drawn.population
        assert _log_normal_about(per_head, CHANNELS[c].spend, 0.5)


def test_a_week_s_spend_jitters_about_its_plan_with_mean_one():
    drawn = GeoMediaMix(CHANNELS, geos=1, weeks=5000, spend_spread=0.0, spend_noise=0.4).draw(19)
    n = drawn.spend.shape[1]
    for c, channel in enumerate(CHANNELS):
        ratio = drawn.spend[0, :, c] / (drawn.population[0] * channel.spend)
        assert abs(ratio.mean() - 1.0) < 5.0 * np.sqrt(np.expm1(0.4**2) / n)
        assert abs(np.log(ratio).std(ddof=1) - 0.4) < 5.0 * 0.4 / np.sqrt(2.0 * n)


def test_the_base_moves_with_the_season(world):
    rise = np.broadcast_to(1.0 + 0.2 * SEASON, world.base.shape)
    np.testing.assert_allclose(world.base / world.base[:, :1], rise, rtol=1e-12)


def test_national_media_is_allotted_by_population(world):
    national = world.spend[:, :, 2]
    per_head = national.sum(axis=0) / world.population.sum()
    allotted = np.broadcast_to(per_head, national.shape)
    np.testing.assert_allclose(national / world.population[:, None], allotted, rtol=1e-12)
    assert world.shares.sum() == pytest.approx(1.0, rel=1e-14, abs=0.0)


def test_the_log_s_spend_rises_with_the_season_by_the_policy():
    drawn = GeoMediaMix(CHANNELS, geos=4, weeks=WEEKS, policy=0.4, spend_noise=0.0).draw(5)
    rise = (1.0 + 0.4 * SEASON)[None, :, None]
    np.testing.assert_allclose(
        drawn.spend / drawn.spend[:, :1, :], np.broadcast_to(rise, drawn.spend.shape), rtol=1e-12
    )


def test_a_geo_s_own_media_spill_half_the_share_into_each_neighbour(world):
    home = world.media[:, :, :2].sum(axis=2)  # the two channels the geos buy themselves
    for g in range(GEOS):
        expected = 0.15 * (home[g - 1] + home[(g + 1) % GEOS])
        np.testing.assert_allclose(world.spilled[g], expected, rtol=1e-12)
    assert not GeoMediaMix(CHANNELS, geos=GEOS, weeks=WEEKS).draw(7).spilled.any()


@pytest.mark.parametrize(("channel", "moved"), [(0, [3, 4, 5]), (2, [4])])
def test_a_geo_s_spend_lifts_only_it_and_its_neighbours(world, channel, moved):
    """Its own channels' lift spills into the two geos beside it; national media's does not."""
    spend = world.spend.copy()
    spend[4, :, channel] = 0.0
    changed = np.any(world.lift(spend) != world.lift(world.spend), axis=1)
    assert np.flatnonzero(changed).tolist() == moved


def test_the_drift_is_a_national_random_walk_in_the_log_of_each_lift():
    drawn = GeoMediaMix(CHANNELS, geos=3, weeks=4000, drift=0.05).draw(13)
    assert np.all(drawn.drift[0] == 1.0)
    steps = np.diff(np.log(drawn.drift), axis=0)
    n = steps.shape[0]
    assert np.all(np.abs(steps.mean(axis=0)) < 5.0 * 0.05 / np.sqrt(n))
    assert np.all(np.abs(steps.std(axis=0, ddof=1) - 0.05) < 5.0 * 0.05 / np.sqrt(2.0 * n))


def test_each_head_adds_noise_of_its_own():
    drawn = GeoMediaMix(CHANNELS, geos=6, weeks=3000, noise=0.7).draw(17)
    per_head = drawn.noise / np.sqrt(drawn.population)[:, None]
    n = per_head.shape[1]
    assert np.all(np.abs(per_head.std(axis=1, ddof=1) - 0.7) < 5.0 * 0.7 / np.sqrt(2.0 * n))


def test_mixes_that_differ_in_a_parameter_draw_the_same_variates():
    plain = GeoMediaMix(CHANNELS, geos=GEOS, weeks=WEEKS).draw(7)
    spilling = GeoMediaMix(CHANNELS, geos=GEOS, weeks=WEEKS, spillover=0.3).draw(7)
    chasing = GeoMediaMix(CHANNELS, geos=GEOS, weeks=WEEKS, policy=0.4).draw(7)
    np.testing.assert_array_equal(spilling.media, plain.media)
    for name in ("population", "effect", "base", "noise"):
        np.testing.assert_array_equal(getattr(chasing, name), getattr(plain, name))


def test_the_world_s_cells_plan_the_next_quarter_with_its_history(world):
    """National media held where it ran, each geo's own channels within half and twice their
    last quarter: the joint plan returns at least the last quarter's mix."""
    current = world.spend[:, -13:, :].mean(axis=1)
    lower, upper = 0.5 * current, 2.0 * current
    lower[:, 2] = upper[:, 2] = current[:, 2]
    arguments = {"history": np.transpose(world.spend, (1, 0, 2))}
    budget = 13.0 * float(current.sum())
    plan = allocate_geos(world.cells(), budget, 13, lower=lower, upper=upper, **arguments)
    held = allocate_geos(world.cells(), budget, 13, lower=current, upper=current, **arguments)
    assert plan.worth >= held.worth
    assert plan.bound >= plan.worth
    np.testing.assert_allclose(plan.spend[:, 2], current[:, 2], rtol=1e-12)


@pytest.mark.parametrize(
    ("build", "error", "match"),
    [
        (
            lambda: MediaChannel("x", Power(1.0, 0.5), 1.0),
            TypeError,
            "not a chc.response.Saturation",
        ),
        (lambda: MediaChannel("x", Tanh(1.0), 0.0), ValueError, "'x''s effect"),
        (lambda: MediaChannel("x", Tanh(1.0), 1.0, retention=1.0), ValueError, "retention"),
        (lambda: GeoMediaMix(()), ValueError, "one MediaChannel or more"),
        (lambda: GeoMediaMix((CHANNELS[0], CHANNELS[0])), ValueError, "names must differ"),
        (lambda: GeoMediaMix(CHANNELS, geos=0), ValueError, "geos must be a whole number"),
        (lambda: GeoMediaMix(CHANNELS, weeks=True), ValueError, "weeks must be a whole number"),
        (lambda: GeoMediaMix(CHANNELS, season=1.0), ValueError, "season"),
        (lambda: GeoMediaMix(CHANNELS, policy=-1.0), ValueError, "policy"),
        (lambda: GeoMediaMix(CHANNELS, noise=-1.0), ValueError, "noise"),
        (lambda: GeoMediaMix(CHANNELS, geos=2, spillover=0.1), ValueError, "needs three"),
        (lambda: MIX.draw(0).lift(np.zeros((GEOS, WEEKS))), ValueError, "geo by week by channel"),
    ],
)
def test_it_refuses_what_it_cannot_draw(build, error, match):
    with pytest.raises(error, match=match):
        build()
