# The dtype policy

JAX computes in single precision by default. `chc` does not change that for you, does not refuse
it, and does not hide it: it warns where precision is load-bearing and records the setting beside
every result that depends on it.

## Why precision is part of the result

- **A float32 run is a different computation, not a cheaper one.** How much that matters is
  plant-specific: it depends on how well conditioned the fit is.
- **A seed is not a dataset across dtypes.** JAX's `x64` flag changes how many bits a threefry key
  spends per element, so the *same* seed draws a *different* sample at the two settings. A run
  recorded without the flag cannot be repeated.

## What the library does

- **`prescribe` warns, and does not refuse.** A run without `x64` produces a `WARNING` record on
  the `chc.decision` logger — `chc_event="precision"` — and the fit proceeds. JAX is
  single-precision by default and the harm is plant-specific, so a caller who needs the guarantee
  asserts on the provenance rather than having a default chosen for them. The library installs no
  handler and sets no level; that decision belongs to the application.
- **`Provenance` records it.** Every `Panel` carries a `Provenance`: a sha256 over each column's
  name, dtype, shape and bytes, the library version, the row count, the column names, an optional
  `seed`, and `x64` — `jax_enable_x64` at the moment the panel was built. Two panels that agree
  numerically but differ in dtype hash differently, which is the honest answer, because they will
  not produce the same numbers. `Prescription.report()` prints it and `to_json()` carries it.
- **`RunProvenance` records the run.** A panel is built once and planned on many times, and a run
  need not match it: built with `x64` on and planned on with it off, the fit runs in float32 while
  the panel's record reads True. `Prescription.run` holds `x64` as `prescribe` ran, the device and
  the dtypes the fit and the solve ran in, why the solve stopped, the fit's seed and settings, the
  planner's, and the versions of the packages that computed the numbers. The report prints it under
  the panel's line, and `to_json()` carries it as `run`.

## What you do

Turn it on before anything creates an array, when the fit is load-bearing:

```bash
export JAX_ENABLE_X64=1
```

or, first thing in the program:

```python
import jax

jax.config.update("jax_enable_x64", True)
```

`out.provenance.x64` then says whether it was on when the panel was built, and `out.run.x64`
whether it was on when `prescribe` ran.

## On this site

The test suite runs in float64 — `tests/conftest.py` enables `x64` — while a standalone script runs
in whatever precision it sets, float32 unless it says otherwise. Numbers calibrated in one regime
can fail in the other. Every number on these pages is computed at the precision of the routine that
printed it: the [quickstart](../quickstart.md) and tutorials 1–4, 6 and 7 enable `x64` in their
first cell, and the robust-control tutorial 5 computes in NumPy;
[tutorial 5's scoreboard](../tutorials/05_benchmark_scoreboard.md) and the case-study and benchmark
scripts run at JAX's default precision, as the same commands do in a default environment.
