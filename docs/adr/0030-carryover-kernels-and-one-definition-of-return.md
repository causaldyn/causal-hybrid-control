# ADR 0030 — Carryover kernels and one definition of return

**Status:** proposed, 2026-09-30.

## Context

ADR 0029 gave `chc.response` its curves. A fitted media-mix model reads a channel's spend through a
carryover kernel before its curve, and the packages a model is ported from disagree on the kernel's
terms:

- whether its weights sum to one. Robyn's geometric kernel does not, PyMC-Marketing leaves it to a
  flag, and a coefficient fitted under one convention is rescaled under the other;
- whether its length is its own. Robyn's Weibull kernels take their scale as a quantile of the
  series' length, so logging another period moves every earlier adstock;
- how many weights a length buys. PyMC-Marketing's geometric and delayed kernels with `l_max` carry
  `l_max` weights, its CDF Weibull kernel `l_max + 1`.

A return is read several ways: what a decomposition attributes to the channel in a window, the
incremental return on a window's spend, the return on one more unit, the return at a steady level.
A report that does not say which, over what window, with what
carryover and in what units cannot be set beside another. The coefficient is none of them: through
a geometric kernel of retention `r` a unit of spend on a linear channel returns, in the long run,
`1 / (1 - r)` times what it returns in its own period, and on a saturating curve the return per
unit moves with the spend.

## Decision

- **Three kernels in `chc.response`** (*experimental*), beside the curves they feed. A kernel's
  parameters are positional; its structure, `length` and `normalized`, is keyword-only, and
  `normalized` has no default.
  - `GeometricAdstock(retention, length=, normalized=)`: `w_l = r^l`, `r` in `[0, 1]`, kept as a
    running product, since the slope of `r^0` at `r = 0` is `0 * inf`;
  - `DelayedAdstock(retention, delay, length=, normalized=)`: `w_l = r^{(l - d)^2}`, the carryover
    peaking `d` periods after the spend (Jin, Wang, Sun, Chan and Koehler 2017);
  - `WeibullAdstock(shape, scale, length=, normalized=)`: `w_0 = 1` and
    `w_l = prod_{j <= l} (1 - F(j))`, `F` the Weibull CDF with its scale in periods.

  A period's adstock is `a_t = sum_{l < L} w_l s_{t - l}`, with nothing spent before the series. It
  reads the `L` periods up to it and no others, so a longer series moves no earlier value, and
  normalising divides by the kernel's own sum. A `(T, C)` series is adstocked column by column.
- **`Channel(kernel, curve, coefficient)`**: a channel's return, `b h(kernel(spend))`, adstock
  first, in the outcome's units.
- **The return, read four ways**, each taking `revenue_per_kpi`, 1 when the outcome is revenue:
  - `contribution(channel, spend, window)`: the channel's return in the window's periods, carried
    over from all the spend before them, in the outcome's units; what a decomposition attributes to
    the channel there. The three readings below are per currency unit of spend.
  - `roi(channel, spend, window)`: the channel's return over the series less its return with the
    window's spend removed, over what the window spent. The carryover counted is what falls inside
    the series.
  - `marginal_roi(channel, spend, window)`: the derivative of the channel's return along the
    window's spend, over what the window spent: the limit of scaling the window's spend by `1 + e`.
  - `steady_state_marginal_roi(channel, level)`: `b h'(level sum(w)) sum(w)`, the slope of a
    period's return once the adstock of a constant `level` has settled; through a long unnormalised
    geometric kernel, `b h'(level / (1 - r)) / (1 - r)`.

  `contribution`, `roi` and `marginal_roi` take any function of a spend series, so a plant rolled
  out under a spend path reads the same way. A window that spends nothing has no return per unit, and asking raises.
- **Not offered: the Weibull PDF kernel scaled to `[0, 1]`**, PyMC-Marketing's default and one of
  Robyn's. Scaling by its own extremes sets its smallest weight to 0 and its largest to 1, and which
  weight is smallest depends on where the length ends, so the kernel's shape moves with its length.

## Consequences

- `tests/test_response.py` holds each kernel to a convolution summed by hand, a shorter series to a
  prefix of a longer one's adstock, and the weights to their definitions, the Weibull kernel's to
  SciPy's Weibull CDF. A linear channel returns `b sum(w)` on every reading per unit; a
contribution over the whole series is the ROI times what was spent, and a window that spends
nothing still carries over what came before it; the marginal return
  matches a finite difference of scaling the window's spend, and the steady state both a finite
  difference of a long rollout of `a_t = s_t + r a_{t-1}` and its closed form; every reading is
  unchanged by a change of currency; the geometric kernel's slope at `r = 0` is finite; and a
  channel fits through its kernel and curve by gradient, compiled.
- `scripts/pymc_marketing_reference.py` runs PyMC-Marketing 1.2.0 beside `chc.response` in a
  throwaway environment. Mapped as `chc.response`'s docstring says, every kernel, raw and
  normalised, agrees to `1.6e-16` of its largest value and every curve to `2.2e-16`, but
  `hill_function`, to `2.3e-9`. That gap is PyTensor's: it makes a Python float it holds exactly a
  float32 constant, so `kappa ** slope` is taken in single precision, and given float64 constants
  `hill_function` is `1.2e-16` from the Hill in 50 digits. The CDF Weibull kernel with
  `l_max = 12` carries 13 weights, and `WeibullAdstock` of length 13 matches it to `1.1e-16`.

## Not built

- **The plant's return read through `roi`.** `chc.mmm`'s plant carries over in continuous time,
  its adstock a state rather than a kernel, so a `Channel` does not describe it; `roi` reads it
  through a function that rolls the plant out under a spend path, once the plant takes a
  `Saturation` for its Hill.
- **Meridian's forms mapped** and run beside CHC's. Meridian is TensorFlow; a throwaway environment
  is the next check.
- **Saturation before adstock**, which some packages allow. No consumer asks for it.

## Alternatives considered

- **A kernel whose length is a share of the series.** Rejected: appending a period moves every
  earlier value, so last quarter's ROI changes when this quarter is logged.
- **Normalised by default, or not.** Rejected: either default is one package's convention, and a
  model ported from the other would come out with its coefficients silently rescaled.
- **`length` positional.** Rejected: `GeometricAdstock(0.6, 8)` and `GeometricAdstock(8, 0.6)` are
  one typo apart, and a kernel's structure reads better named.
- **The marginal return as that of a 1% rise in spend**, as some reports quote it. Rejected for the
  derivative, its limit: on a curved response the step's size moves the number, and the step is a
  choice the reader cannot see.
- **The coefficient as the return.** Rejected: it ignores carryover and is undefined under
  saturation.
- **The kernels in a module of their own.** Rejected: a `Channel` holds a kernel and a curve, and
  the return reads a `Channel`; one module keeps a channel's response in one place.
