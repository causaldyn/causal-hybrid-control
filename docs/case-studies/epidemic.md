# Flatten the curve

Nonlinear known dynamics — a compartmental SIR model — with a non-pharmaceutical intervention `u`
that scales transmission, `beta -> beta*(1-u)`. Optimal control holds infections at a
hospital-capacity threshold with minimal intervention, the threshold a steep penalty rather than a
hard limit: optimal control on a classic nonlinear population model, differentiating through the
rollout. In observational logs the intervention's effect would be confounded, because policy reacts
to case counts; here the plant is the true system and control is planned against it.

```bash
uv run python scripts/epidemic_demo.py
```

What it printed when this site was built:

```text
--8<-- "docs/output/epidemic_demo.txt"
```

With matplotlib installed the script also writes `outputs/epidemic.png`;
[tutorial 4](../tutorials/04_epidemic_and_pessimism.md) draws the curves. Code:
[`chc.epidemic`](../api/epidemic.md).
