# ADR 0036 — Accelerator extras and the Python range

**Status:** accepted, 2026-09-30.

## Context

`chc` has no device-specific code: its arrays go wherever JAX puts them. Until now the installation
page told a user to install JAX's build for their hardware beside `chc`, and said there was no
`chc[cuda]` extra on purpose, since it would only repeat JAX's. The repository kept JAX's CUDA 13
wheels in a `cuda` dependency group, which the package does not publish, run from a second
environment, `.venv-cuda`, by `just test-gpu`.

Two costs showed. On a machine with an NVIDIA GPU, every process that started a JAX backend from
`.venv` printed "An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not
installed. Falling back to cpu." And testing on the GPU meant knowing about a second environment
and a second recipe.

`requires-python` was capped below 3.15 because diff-diff, behind the `did` extra, declares Python
below 3.15, and uv's universal lock must resolve every dependency across the whole range. Python
3.15 has reached its release candidates. On the free-threaded builds, NumPy, SciPy, pandas, jaxlib and the CUDA
plugins publish wheels for 3.14t and 3.15t; polars, whose runtime is abi3, and diff-diff publish
none, and CatBoost has none for 3.15 either.

## Decision

- **Extras under JAX's names**: `cpu`, `cuda13`, `cuda12`, `cuda13-local`, `cuda12-local`,
  `rocm7-local`, `tpu` and `oneapi`, each forwarding to JAX's extra of the same name, which pins
  `jaxlib` and the plugin to jax's own version. `oneapi` carries `python_version >= '3.12'`: the jax
  the lock resolves on 3.11, 0.10.2, has no oneAPI build. `[tool.uv] conflicts` makes the
  accelerator extras mutually exclusive, since two plugins would both claim the device; `cpu`, empty
  in JAX 0.11, conflicts with none. `tests/test_accelerator_extras.py` fails when an extra names one
  the installed jax does not provide, which pip and uv install with a warning alone.
- **The justfile picks the build from the machine.** `accelerator` is `cuda13` where `nvidia-smi`
  reports a driver of 580 or newer and a GPU of compute capability 7.5 or newer, `cuda12` from driver
  525, and empty otherwise; `just sync` and `just test` install it into `.venv`. `just test-cpu` pins
  the CPU and is the one `just check` runs, so a green check is still a green CI. The `cuda` group,
  `.venv-cuda` and `just test-gpu` are retired.
- **`tests/conftest.py` holds the suite to its CPU semantics on a GPU**: preallocation off, so the
  pytest-xdist workers share the card, and `jax_default_matmul_precision="highest"`, so a float32
  product is not computed in TensorFloat-32. Both are no-ops on the CPU.
- **Python 3.11 upward, uncapped.** diff-diff joins the `did` extra and the `dev` group behind
  `python_version < '3.15'`; on 3.15 the extra installs nothing, and `callaway_santanna_inference`
  says so in the error it raises. CI's test matrix gains 3.15, 3.14t and 3.15t, its lint matrix
  3.15. The free-threaded legs sync without polars and diff-diff, since PEP 508 has no marker for
  a free-threaded build and the lock cannot leave them out; the tests that need either skip. The
  classifiers add 3.15, `Free Threading :: 2 - Beta`, and NVIDIA CUDA 12 and 13.
- **`ty` checks each interpreter at its own version.** `[tool.ty.environment] python-version` was
  3.11, the floor, which made ty read the development environment's NumPy 2.5 stubs, written for
  3.12 and later, on a branch they were never written for; from ty 0.0.67 that rejected `.min()` on
  a float64 array, eight times. It is now 3.14, `.python-version`'s, and CI's lint legs and
  `just types-matrix` pass each leg's own version, the 3.11 leg holding the floor.

## Consequences

- An extra's name is published metadata, and a user's install line will depend on it. The names are
  JAX's, so they move when JAX's do, and the test above says when.
- With a CUDA build in `.venv`, every process loads its libraries, on the CPU too: a process that
  starts JAX's CPU backend held 0.60–0.62 GB against 0.23–0.25 GB without the build, measured
  2026-09-30. `just test-cpu` pays it per worker.
- A GPU reproduces a CPU result to rounding, not bit for bit. Results that must reproduce exactly
  pin `JAX_PLATFORMS=cpu`.
- On 3.15, `chc.did`'s inference waits for diff-diff to lift its cap; `trees` waits for CatBoost.
- On the free-threaded builds, `chc` is tested, not made safe for concurrent use of one object:
  hence Beta.

## Alternatives considered

- **Keep installing JAX's build beside `chc`.** Still works, and the installation page keeps the
  form for a project that must also resolve on macOS or Windows. As the only route, it leaves the
  one line a GPU user wants unwritten.
- **Keep the `cuda` dependency group.** Groups are not published, so it can serve development but
  never a user's install; and it kept the GPU in a second environment.
- **CUDA in `default-groups` for every environment.** CI's runners have no GPU, and every
  contributor would download the CUDA wheels whatever their hardware.
- **Choosing the build at install time from the hardware.** A wheel's metadata is static; neither pip
  nor uv can select a dependency by the GPU present. The justfile does it for the development
  environment only.
- **Keeping the cap below 3.15 until diff-diff lifts its own.** It would keep all of `chc` off 3.15
  for one optional extra.
