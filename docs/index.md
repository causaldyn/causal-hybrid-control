# causal-hybrid-control

Physics-structured dynamics with a **learned causal residual**, controlled by **constrained optimal
control / MPC**, and hedged on offline, confounded data by an explicit **pessimism / support**
layer. Each certificate states what it bounds, on what and under which premises:
[the scopes](concepts/certificates.md#scopes).

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

JAX, Diffrax, Equinox, Optax, NumPy and SciPy come with it; Python 3.11–3.15, the free-threaded
3.14t and 3.15t included.

**GPU, TPU and other hardware.** chc has no device-specific code: install JAX's build for your
hardware through the extra of the same name — `pip install "causal-hybrid-control[cuda13]"` on an
NVIDIA GPU whose driver is 580 or newer — and JAX places the arrays there.
[Installation](installation.md) lists every extra, CUDA 12 and 13, TPU, ROCm and oneAPI, JAX's conda
and nightly builds, and what changes for chc on an accelerator.

## Where to go next

- [Why, and when not to use it](why.md) — the gap this closes, where it sits among the tools you
  already know, and the problems it is the wrong tool for.
- [Quickstart](quickstart.md) — from a panel of logs to a schedule and its certificate, in one
  call.
- [The decision lifecycle](lifecycle.md) — identify, plan, evaluate, experiment, deploy, adapt:
  what is built for each stage, where it is documented, and what is not built yet.
- [What's inside](modules.md) — every area of the library, what it does, and where its evidence
  is.
- **Concepts** — [identification](concepts/identification.md),
  [pessimism](concepts/pessimism.md), [certificates](concepts/certificates.md),
  [the sensitivity level Γ](concepts/gamma.md) and [the dtype policy](concepts/dtype-policy.md).
- [Tutorials](tutorials/index.md) — the notebooks, from a first decision to real data, executed
  when this site was built.
- [Case studies](case-studies/index.md) — one command each, and what it printed.
- [API reference](api/index.md) — every public module, grouped by what a break would cost you.
- [Theory](theory.md), [Benchmarks](benchmarks.md), [Security](security.md), [Cite](cite.md).

The results on the quickstart, tutorial, case-study and benchmark pages are printed while the site
is built, by the committed notebook or script each page names. The release-by-release record,
scope corrections included, is the
[changelog](https://github.com/causaldyn/causal-hybrid-control/blob/main/CHANGELOG.md).
