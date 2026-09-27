# Driver supply, end to end

Every layer of the library on one decision, so the pieces are forced to compose:

1. **fit** — an incentive response estimated from logs whose behaviour policy chased a confounder,
   once naively and once with the backdoor adjustment;
2. **plan** — constrained optimal control with a certified Grönwall error tube (`causal_plan`);
3. **certify** — the plan priced against a partially identified effect (`certify_safety`);
4. **audit** — each plan then executed on the *true* plant, so the numbers a caller would have
   trusted offline can be compared with what actually happened.

The plant is two zones of a mobile driver pool. Incentivising zone A pulls drivers *out of* zone B —
the `[+b, -b]` control column is driver conservation, the interference channel — zone B drains on
its own, and the barrier is a supply floor there. So the objective and the constraint pull against
each other through the only lever available, which is what makes the certificate load-bearing
rather than decorative.

```bash
uv run python scripts/spine_demo.py
```

What it printed when this site was built:

```text
--8<-- "docs/output/spine_demo.txt"
```

Read the `Gamma*` column: it is computed offline, before either arm acts on the plant, and needs no
ground truth. See [certificates](../concepts/certificates.md) for what it does and does not
certify.

The plant is deliberately control-affine: `certify_safety` reads the channel off the Jacobian at
`u = 0`, which is exact for an affine plant and only a linearisation otherwise. Code:
[`chc.spine`](../api/spine.md).
