# Pessimism

A controller trained offline must not exploit its model where it was never trained. The objective
it optimises carries that explicitly:

```text
u* = argmin_u  J_task(u) + λ_unc·U(x,u) + λ_supp·D((x,u), 𝒟)
```

## The two penalties

- **`D` — distance from the data.** [`SupportModel`](../api/support.md) scores how far a
  state-action pair sits from the offline data cloud (a squared Mahalanobis distance);
  `pessimistic_control` penalises leaving that support.
- **`U` — calibrated predictive uncertainty.** [`chc.uncertainty`](../api/uncertainty.md) fits K
  residuals as a deep ensemble. Their disagreement is the epistemic uncertainty of the learned
  dynamics — large where members trained on the same data extrapolate apart — and split conformal
  turns it into interval widths with a finite-sample coverage guarantee. A time-consistent
  nested-CVaR aggregation keeps one very bad step from being averaged away.

`pessimistic_control` combines the two, and the uncertainty scorers plug into its penalty channel,
so model exploitation is bounded, not merely discouraged. A third scorer, `WassersteinPenalty`,
targets deployment shift rather than in-distribution spread: a Wasserstein-1
distributionally-robust margin that keeps control where a small shift of the state distribution
cannot move the learned dynamics much.

## What pessimism is not

It is not identification. Pessimism calibrated to sampling noise shrinks as the log grows, while
the bias from an unmeasured confounder does not — which is why the usual offline-RL recipe does not
close the gap on confounded logs. Confounding and off-support model error are different failure
modes, and each has its own safeguard here: [identification](identification.md) for the first,
this layer for the second.

When the effect is only partially identified, the two meet: `ConfoundingRobustPenalty` carries the
sensitivity radius of [Γ](gamma.md) into the same pessimistic-control stack.

## Before deploying

[`chc.offpolicy`](../api/offpolicy.md) is the pre-deployment check: from logs collected under a
behaviour policy, it estimates a candidate policy's value by inverse-propensity weighting (IPS and
SNIPS) and flags (`overlap_ok`) when the candidate's actions leave the logged support — no
overlap, no evidence. Overlap is summarised by the effective sample size. It does not refuse, and
what it estimates is the candidate's value **on the logger's states**, one step at a time: for a
feedback plan on a plant with memory that is not the value of deploying it, and on a loop where
the candidate spreads the state it was off by half with the flag set (see the module docstring).

## See it

- [Tutorial 4](../tutorials/04_epidemic_and_pessimism.md) — a greedy controller extrapolates off
  the logged support and stalls on the true plant; the pessimistic one stays in support.
- The `support-shift` and `model-uncertainty` tasks of the [benchmarks](../benchmarks.md).
