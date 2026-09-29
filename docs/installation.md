# Installation

```bash
pip install causal-hybrid-control    # or: uv add causal-hybrid-control
```

This brings JAX's CPU build with Diffrax, Equinox, Optax, NumPy and SciPy; Python 3.11–3.14.

`chc` has no device-specific code. Its arrays go wherever JAX puts them, so running it on an
accelerator means installing JAX's build for that hardware beside it. There is no `chc[cuda]`
extra, on purpose: every JAX build for an accelerator is installed through JAX's own extras, and
an extra here would only repeat one of them. This page follows JAX's
[installation guide](https://docs.jax.dev/en/latest/installation.html) as of JAX 0.11; where the two
differ, JAX's guide is the authority.

## Which build for which hardware

JAX's support table:

| | Linux x86_64 | Linux aarch64 | Mac aarch64 | Windows x86_64 | WSL2 x86_64 |
|---|---|---|---|---|---|
| CPU | yes | yes | yes | yes | yes |
| NVIDIA GPU | yes | yes | n/a | no | experimental |
| Google Cloud TPU | yes | n/a | n/a | n/a | n/a |
| AMD GPU | yes | no | n/a | no | experimental |
| Apple GPU | n/a | no | experimental | n/a | n/a |
| Intel GPU | experimental | n/a | n/a | no | no |

## CPU

Nothing to add: the install above is the CPU build. To move JAX forward on its own,
`pip install --upgrade jax`. JAX calls its Windows CPU wheels experimental; they may also need the
Microsoft Visual Studio 2019 Redistributable.

## NVIDIA GPU

### CUDA from pip wheels

The route JAX recommends: the wheels carry CUDA, cuDNN and NCCL themselves, and only the driver
comes from the system (`nvidia-smi` prints its version).

```bash
pip install causal-hybrid-control "jax[cuda13]"   # driver >= 580, GPUs of SM 7.5 or newer
pip install causal-hybrid-control "jax[cuda12]"   # driver >= 525, GPUs of SM 5.2 or newer
```

JAX recommends CUDA 13 and plans to drop CUDA 12. With uv, `uv add causal-hybrid-control "jax[cuda13]"`. A project that must also resolve on macOS or
Windows can keep the CUDA build to Linux with a marker:

```bash
uv add "jax[cuda13]; sys_platform == 'linux'"
```

- **Leave `LD_LIBRARY_PATH` unset.** It can override the CUDA libraries the wheels bring.
- **Upgrade through the extra**, `pip install --upgrade "jax[cuda13]"`. The extra pins the CUDA
  plugin and `jaxlib` to jax's own version, and upgrading `jax` alone leaves the plugin behind.

### CUDA installed on the system

```bash
pip install causal-hybrid-control "jax[cuda13-local]"   # CUDA >= 13.0, cuDNN >= 9.12 and < 10.0, NCCL >= 2.18
pip install causal-hybrid-control "jax[cuda12-local]"   # CUDA >= 12.1, cuDNN >= 9.10.2 and < 10.0, NCCL >= 2.18
```

Here JAX finds the CUDA libraries through `LD_LIBRARY_PATH` and `ptxas` and `nvlink` through
`PATH`, so both must point at the installation meant. It also needs `libdevice10.bc`, which usually
comes with `cuda-nvvm`, and a driver at least as new as the toolkit's. NCCL matters only on several
GPUs.

### Containers, WSL2 and Windows

NVIDIA's [JAX Toolbox](https://github.com/NVIDIA/JAX-Toolbox) containers carry nightly JAX: install
`chc` inside one with pip. WSL2 is experimental. On native Windows the CUDA wheels do not work, and
JAX's guide warns that they can fail silently.

## Google Cloud TPU

```bash
pip install causal-hybrid-control "jax[tpu]"
```

## AMD GPU

Install ROCm first, following
[AMD's guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html):
the extra brings only JAX's ROCm plugin, built against one ROCm version, and AMD's
[JAX on ROCm matrix](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/jax-install.html)
says which. Then:

```bash
pip install causal-hybrid-control "jax[rocm7-local]"
```

AMD also publishes a container, `docker pull rocm/jax:latest`. WSL2 is experimental.

## Apple silicon

Use the CPU build: JAX's guide says JAX does not support Mac GPUs, although its table still marks
the Apple GPU experimental.

## Intel GPU

Experimental, through Intel's OneAPI plugin: either its
[pip installation](https://github.com/intel/intel-extension-for-openxla/blob/main/docs/acc_jax.md)
or [Intel's XLA container](https://hub.docker.com/r/intel/intel-optimized-xla).

## conda-forge

Community-supported. On a machine with an NVIDIA GPU the first command should already bring a
CUDA-enabled `jaxlib`; the second insists on one:

```bash
conda install jax -c conda-forge
conda install "jaxlib=*=*cuda*" jax -c conda-forge
```

`chc` installs from PyPI: `pip install causal-hybrid-control` follows in the same environment.

## Nightly JAX, source builds, older wheels

```bash
# CPU
pip install -U --pre jax jaxlib -i https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/
# NVIDIA, CUDA 13 (for CUDA 12, replace cuda13 by cuda12 twice)
pip install -U --pre jax jaxlib "jax-cuda13-plugin[with-cuda]" jax-cuda13-pjrt \
    -i https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/
```

The TPU nightly, building from source and installing `jaxlib` wheels PyPI no longer carries are in
[JAX's guide](https://docs.jax.dev/en/latest/installation.html). `chc`'s CI tests the released
JAX versions its lockfile resolves, not nightlies.

## Checking the device

```bash
python -c "import jax; print(jax.devices())"    # [CudaDevice(id=0)] on a working CUDA build
```

- **`JAX_PLATFORMS=cpu`** runs on the CPU where an accelerator build is installed.
- **`JAX_PLATFORMS=cuda`** makes the GPU a requirement: JAX refuses to start without one. Without
  it, the CPU build on a machine with an NVIDIA GPU runs on the CPU with a warning, and a CUDA build
  on a machine without one runs on the CPU silently. A CUDA build whose GPU is present but fails to
  initialize is an error either way.

## What changes on an accelerator

- **Precision.** The fits want float64 ([the dtype policy](concepts/dtype-policy.md)). A GPU computes
  float64 as specified, but most consumer GPUs run it at a small fraction of their float32 rate.
- **float32 matrix products.** On NVIDIA GPUs from Ampere on, JAX's default lets a float32 matrix
  product run in TensorFloat-32, which keeps 10 of float32's 23 mantissa bits.
  `JAX_DEFAULT_MATMUL_PRECISION=float32`, or `jax.config.update("jax_default_matmul_precision",
  "float32")` before any computation, restores full float32 products. float64 is unaffected.
- **The same seed is not the same sample.** A key draws the same normal variates on the CPU and the
  GPU to rounding, not bit for bit, so a result recorded on one reproduces on the other to rounding
  only.
- **Some modules stay on the CPU.** `chc.dlm`, `chc.evaluation`, `chc.experiment`,
  `chc.switchback`, `chc.did` and `chc.scm` are NumPy and SciPy.
- **Speed is the workload's.** Small problems spend their time on dispatch and compilation, where
  a GPU does not help; large batched linear algebra is where it does. Time your own case.
- **Memory.** JAX preallocates 75% of the GPU's memory at its first operation, which leaves little
  for a second process on the same card. `XLA_PYTHON_CLIENT_PREALLOCATE=false` allocates on demand
  instead, and `XLA_PYTHON_CLIENT_MEM_FRACTION=.40` preallocates 40%; JAX's
  [GPU memory page](https://docs.jax.dev/en/latest/gpu_memory_allocation.html) has the rest.

## Developing `chc` on a GPU

The repository keeps JAX's CUDA 13 wheels in an opt-in dependency group, `cuda`, that the package
does not depend on and CI never installs. `just test-gpu` runs the test suite on the GPU from its own
environment, `.venv-cuda`; see
[CONTRIBUTING](https://github.com/causaldyn/causal-hybrid-control/blob/main/CONTRIBUTING.md).
