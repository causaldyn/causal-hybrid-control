# causal-hybrid-control

Physics-structured dynamics with a **learned causal residual**, controlled by **constrained optimal
control / MPC** — and made safe on offline, confounded data by an explicit pessimism/support layer.

```
ẋ = f_known(x, u, t; p) + r_θ(x, u, t)          # known mechanism + learned (causal) residual
u* = argmin_u  J_task(u) + λ_unc·U(x,u) + λ_supp·D((x,u), D)   # pessimistic constrained control
```

> Status: **early / experimental** (`v0.0.1`). Currently a working method spine — hybrid dynamics,
> RK4 rollout, a hand-written **discrete adjoint** (verified against autodiff and finite differences),
> and projected-gradient optimal control on a toy system. Causal identification, pessimism, MPC, and the
> benchmark are on the roadmap.

## Install (dev)

```bash
uv sync              # creates .venv, installs deps (JAX + Diffrax + Equinox + Optax)
uv run pytest        # run the test suite
```

## Minimal example

```python
import jax.numpy as jnp
from chc import DampedOscillator, HybridDynamics, ZeroResidual, QuadraticCost, projected_gradient_control

dyn = HybridDynamics(known=DampedOscillator(omega=1.0, zeta=0.1), residual=ZeroResidual(out_dim=2))
cost = QuadraticCost(Q=jnp.diag(jnp.array([1.0, 0.0])), R=jnp.array([[0.01]]),
                     Qf=jnp.diag(jnp.array([10.0, 1.0])), x_target=jnp.zeros(2))

x0 = jnp.array([1.0, 0.0])
us0 = jnp.zeros((50, 1))
us, history = projected_gradient_control(dyn, x0, us0, dt=0.1, cost=cost,
                                         u_lo=-5.0, u_hi=5.0)
print(float(history[0]), "->", float(history[-1]))  # cost decreases
```

## License

MIT
