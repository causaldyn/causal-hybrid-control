# ADR 0017 — The marketplace plant: zones over time, control-affine by construction

**Status:** proposed, 2026-09-28.

## Context

The 0.12.0 release (0.10.0 until D35 put the media-mix release first, and 0.11.0 until a planner over a posterior's draws shipped before it) pre-registers a marketplace study whose headline is that an off-policy estimate
of a plan predicts the plan's realised lift. The study needs a plant that every stage of the loop
can run on:

- the causal fit and the planner, which need a control-affine plant;
- `evaluate_plan`, which needs a `LinearGaussianPlant`;
- `design_experiment`, which needs a one-shot `ZoneDecision`;
- a switchback over zones and periods.

The library had two marketplaces, and neither fits. `chc.marketplace`'s `SharedStateMarket` is a
softmax equilibrium, so its actions do not enter affinely. `chc.matching` is a static transport
problem with no time. FleetPy, the environment CHC did not write, is the study's second plant
(MK3). It cannot be the first: its truth is not known in closed form, and whether it runs within
the budget is the open question.

## Decision

`chc.zones`: `ZoneMarketPlant`, an `equinox` vector field, and `ZoneMarketSystem`, its parameters
and log. The shape follows `chc.mmm`'s plant and system.

- **Per zone, three states and two levers.** The states are idle supply, open requests and an
  incentive stock; the levers are an incentive and a price. The levers are moves from the
  do-nothing point, on `[0, 1]`. The incentive recruits at once through the constant channel
  `(I - P) diag(recruit)`, and late through the stock. The price turns away `demand * elasticity`.
  The rest is drift, so the plant is control-affine with a constant channel.
- **The do-nothing point is the parameterisation.** The caller names each zone's idle supply and
  open requests with no lever moved. The supply level and the demand are derived so that this
  point is the steady state (`validation/zone_market.mac`). The numbers that set the plant are
  then ones an operator can read off a dashboard.
- **Matching is harmonic by default, `mu s q / (s + q)`.** The alternative is its tangent at the
  do-nothing point. The harmonic law is homogeneous of degree one, so the tangent passes through
  zero, and the two laws share the steady state. Under the tangent every closed form is exact;
  under the harmonic law a polynomial fit is a surrogate, as `chc.mmm`'s Hill curve makes it.
- **Interference on a ring.** A share `spill` of a zone's recruits come from its neighbours, so
  the city keeps `1 - spill` of them, and a test checks that conservation.
- **The delayed supply response is a mechanical stock.** `theta` is taken as known, and
  `stock_dynamics` hands the stock rows to a fit as `known=`, as `adstock_dynamics` does.
- **The confounding is placed where the mechanism is.** The logged operator moved both levers
  with each zone's own shock. That shock raises demand and takes supply off the road. It is
  logged, so it can be adjusted for, and `graph()` derives the adjustment. Shocks are independent
  across periods, so the stock's drift is not confounded; the module docstring says so.
- **Two bridges, each exact under linear matching.**
  - `zone_decision` gives the settled trips as a `ZoneDecision` over the incentives, with the
    steady-state response derived in Maxima.
  - `linear_gaussian` gives one RK4 period's Jacobians at the do-nothing point, and the noise the
    shock and the jitter carry.
- **Experimental** until the study it is the plant of is pre-registered.

## Consequences

- **Every number from this plant is by construction:** CHC wrote the plant it is scored on, and
  each such number says so.
- **Six logs were read before any threshold was written.**
  - A fit that ignores the shock reads the price at −0.71 to 0.21 of one period's response.
    That is a price rise raising demand in 18 of 24 zones.
  - It reads the incentive at 0.18–0.39 of it, averaged over the city.
  - Adjusted, it reads the price at 0.78–1.16 and the incentive at 0.90–1.08, zone by zone.
- **Under harmonic matching `zone_decision` is first order.** Its error, relative to the move it
  predicts, is 3.5% at an incentive of 0.025 held in three zones, and 12.8% at 0.1. It halves with
  the incentive, and a test holds that rate.
- **The tests fit under Euler** and compare with one period's response, which is what an Euler
  fit reads. At four zones the fit has 12 states and 8 levers, so 1404 coefficients against the
  marketing-mix case's 80. The RK4 fixed point's Newton matrix carries one forward-mode tangent per
  coefficient and is evaluated eagerly; a profile put most of the fit's time there. It is untimed:
  the machine was not quiet.
- **The tests read the fitted channel at the do-nothing point.** At `degree = 1` a fitted channel
  is affine in the state, `B(x) = C_0 + sum_l C_l x_l`, not constant.
- **The log's actions are correlated with its noise.** An off-policy evaluation that weights
  actions by `u | x` alone assumes they are not, so on these logs the study has to condition on the
  shock or evaluate from a randomised log. How far off the unconditioned evaluation is has not been
  measured.

## Alternatives considered

- **Extend `SharedStateMarket`.** Rejected: its equilibrium is not control-affine, and an affine
  surrogate of it would be a second plant with a different truth.
- **A discrete agent-based simulator.** Rejected: it has no closed forms, so nothing in it could be
  checked exactly. That is FleetPy's role, as the plant CHC did not write.
- **Linear matching by default.** Rejected: the fitted class would contain the truth, which
  flatters the fit. The harmonic default keeps the fit a surrogate, and the tangent stays one
  argument away for exact checks.
- **One shock for the whole city.** Rejected: every zone's levers would move together, so the fit
  could not tell one zone's lever from another's.
- **A price that pushes riders to neighbouring zones.** Not modelled: interference runs through
  supply only. A price spillover is the first extension if the study needs one.
