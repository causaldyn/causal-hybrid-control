# causal-hybrid-control

Physics-structured dynamics with a **learned causal residual**, controlled by **constrained optimal
control / MPC**, and made safe on offline, confounded data by an explicit **pessimism / support**
layer.

```text
ẋ = f_known(x, u, t; p) + r_θ(x, u, t)                         # known mechanism + learned residual
u* = argmin_u  J_task(u) + λ_unc·U(x,u) + λ_supp·D((x,u), 𝒟)   # pessimistic constrained control
```

Most data science stops at prediction. The value is in *decisions* — and a decision changes the
future, so it must be evaluated as an **intervention**, not a correlation, and chosen by **optimal
control**, not by argmax over a predictive score. `chc` is a small JAX library for that step.

## Install

```bash
pip install causal-hybrid-control    # or: uv add causal-hybrid-control
```

JAX, Diffrax, Equinox, Optax, NumPy and SciPy come with it; Python 3.11–3.14.

**GPU.** There is no `chc[cuda]` extra, on purpose: the project's lockfile pins the CPU `jaxlib`,
and a CUDA wheel there would install CUDA for every user. Add JAX's CUDA build to the environment
that uses chc — `pip install causal-hybrid-control "jax[cuda13]"`, or `cuda12`, whichever your
driver supports — and JAX places arrays on the GPU; chc has no GPU-specific code.

## Where to go next

- [Why, and when not to use it](why.md) — the gap this closes, where it sits among the tools you
  already know, and the problems it is the wrong tool for.
- [Quickstart](quickstart.md) — from a panel of logs to a certified schedule, in one call.
- **Concepts** — [identification](concepts/identification.md),
  [pessimism](concepts/pessimism.md), [certificates](concepts/certificates.md),
  [the sensitivity level Γ](concepts/gamma.md) and [the dtype policy](concepts/dtype-policy.md).
- [Tutorials](tutorials/index.md) — the eight notebooks, executed when this site was built.
- [Case studies](case-studies/index.md) — one command each, and what it printed.
- [API reference](api/index.md) — every public module, grouped by what a break would cost you.
- [Theory](theory.md), [Benchmarks](benchmarks.md), [Security](security.md), [Cite](cite.md).

The results on the quickstart, tutorial, case-study and benchmark pages are printed while the site
is built, by the committed notebook or script each page names. The release-by-release record,
scope corrections included, is the
[changelog](https://github.com/causaldyn/causal-hybrid-control/blob/main/CHANGELOG.md).
