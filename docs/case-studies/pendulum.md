# A pendulum from a confounded log

Pendulum-v1 raised 0.30 rad from hanging and held there, planned by `prescribe` from a log whose
torque was confounded. While logging, an operator cancelled half of the wind torque it measured,
`u = -k w + e`, so the logged torque moves with a disturbance that enters `omega'` exactly where the
torque does: the confounding sits on the control channel itself.

```text
d(theta)/dt = omega
d(omega)/dt = 15 sin(theta) + 3 (u + w)
```

Gravity is first principles and goes to `prescribe` as `known=`, so what the log has to identify is
the actuator, the `3` on the torque. Fitted without the wind, that channel is
`b (k(k-1) s_w^2 + s_e^2) / (k^2 s_w^2 + s_e^2)` for white wind, which is negative for an operator
who cancels half of it: the log teaches that torque pushes backwards.

```bash
uv run python scripts/pendulum_demo.py
```

What it printed when this site was built:

```text
--8<-- "docs/output/pendulum_demo.txt"
```

The rows are the same planner reading the same log through different causal claims, every schedule
run on the **true** pendulum:

- `adjusted` — the graph derives the adjustment set `{wind}`;
- `asserted` — an empty set asserted, which is what fitting the log as it stands amounts to;
- `latent` — the same graph with the wind declared unobserved: nothing identifies the channel, and
  `prescribe` returns no schedule at all;
- `none` — never acting, the reference the angle errors are read against.

Scope, and it bounds what the numbers mean:

- The schedule is open loop on a frictionless plant, and it lands because the fitted model is
  right. Deployed, it would be re-planned from each measured state (`chc.RecedingHorizon`).
- `|omega| <= 1` is held in the solve as the barrier, at the library's class-K gain of 1 per second,
  so what binds is the approach to the speed limit, not the limit.
- The certificate's trusted prefix is short. Its tube compounds at the norm of the hanging
  pendulum's slope, 15 per second, a `local` rate, since gravity bends the field away from where
  the plan starts. It is fed the channel's standard error at the states the log visited. The run
  on the true plant shows the rest of the schedule holding; the certificate does not claim it.

Code:
[`scripts/pendulum_demo.py`](https://github.com/causaldyn/causal-hybrid-control/blob/main/scripts/pendulum_demo.py),
through [`prescribe`](../api/decision.md).
