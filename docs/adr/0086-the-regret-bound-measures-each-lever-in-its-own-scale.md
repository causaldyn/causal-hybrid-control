# ADR 0086 — The regret bound measures each lever in its own scale

**Status:** accepted, 2026-10-12.

## Context

`plan_regret_bound` certifies how far a finished plan is from the best one its box allows. Per
action it maximises the certified gain of a feasible move, `-g d - (mu/2) d^2`, over the box. The
modulus `mu` is the strong-convexity modulus of the planning objective `J`. Read off `J`'s Hessian,
it was the least eigenvalue of the Hessian over the actions the box lets move.

That least eigenvalue is the curvature in the plain norm `|d|`, and the plain norm adds moves of
different levers as if they were in one unit. With levers in different units, or of different
prices, the least eigenvalue is the cheapest lever's curvature. The bound then charges every
lever's moves at it, and it reads the levers' units. The bound stays valid in any units. It is
only loose, and how loose depends on units that carry no information.

On the decision tests' panel (`tests/test_cross_module_invariants.py`, two levers in `[-2, 2]`
priced 0.01 and 1, three steps), the `together` log's prescription read a regret bound of 0.0266.
With `u1` in units 1e3 times its own, its box and price with it, the same problem read 0.0576. With
`u1` in units 1e-3 times its own it read 0.0719. The `free` log's bound, at rounding, read 1.8e-16
in the levers' own units and 6.1e-9 with `u1` in thousandths and `u2` in thousands.

## Decision

- **The moves are measured lever by lever, each in its own scale.** Lever `j`'s scale is
  `s_j = (h / h_j)^(1/2)`. Here `h_j` is the mean of the Hessian's diagonal over lever `j`'s moving
  actions at the plan, and `h` is the largest `h_j` of a lever that moves. With `D` the scales
  along the actions, the modulus is the least eigenvalue of `D H D` over the moving actions. Along
  lever `j` the move's curvature is `mu / s_j^2`, and the bound is the same per-action maximum with
  that curvature. `validation/constrained_plan_regret.mac` STEP 8 derives it.
- **Why it is valid.** `D H D >= mu I` is `H >= mu D^-2` (STEP 8a): strong convexity in the norm
  `|D^-1 d|`. In the variables `V = D^-1 U` the box is a box, and the plain bound holds there word
  for word. Its terms, read back in `U`, are the per-action terms above (STEP 8b). Any positive
  scale gives a valid bound (STEP 8f). The choice of scale decides how tight the bound is, never
  whether it holds.
- **Why it reads no lever's units.** With lever `j` in units `c_j` times its own, `h_j` moves as
  `1 / c_j^2` and `s_j` as `c_j`, up to one factor common to all levers. `D H D` is the same matrix
  up to that factor, each term of the bound is the same number, and the factor cancels (STEP 8c).
- **One lever, or levers of equal curvature, keep the plain bound to the bit.** Their scales are
  exactly 1, so the modulus is the Hessian's least eigenvalue and every term is computed as before.
  The scale's reference is the stiffest lever that *moves*, so a lever that a box pins does not
  move the modulus of the levers beside it.
- **A lever with no positive curvature keeps the scale 1.** A diagonal entry of the moving block is
  then 0 or below, so no scale makes the modulus positive. The bound is the Frank-Wolfe gap or a
  refusal, in any units, as before.
- **A sampled modulus uses the plan's scales at every point drawn.** Any fixed positive scale is
  valid at every point, so the scales need not follow the sample.
- **A supplied modulus is the plain norm's.** The caller states it in the levers' own units, and
  the bound then reads those units. Read off the Hessian, the bound reads none.
- **`gradient_norm` is the dual norm's, `|D g|`,** so `unconstrained_bound` is
  `gradient_norm^2 / (2 mu)` in the same norm. Like the modulus, `gradient_norm` is in the
  stiffest moving lever's units; `unconstrained_bound` reads no lever's units.

## Evidence

All in float64. The decision tests' panel, the `prescribe` regret bound with each lever in units
`c` times its own (`diagnostic`, as the fitted channel reads the state):

| log | units `(c_1, c_2)` | before | after |
|---|---|---|---|
| `together` | `(1, 1)` | 0.026623 | 0.021377 |
| `together` | `(1e3, 1)` | 0.057590 | 0.021377 |
| `together` | `(1e-3, 1)` | 0.071944 | 0.021377 |
| `together` | `(1e-3, 1e3)` | 0.071950 | 0.021377 |
| `free` | `(1, 1)` | 1.77e-16 | 1.72e-16 |
| `free` | `(1e3, 1)` | 5.36e-12 | 1.72e-16 |
| `free` | `(1e-3, 1e3)` | 6.08e-9 | 1.72e-16 |
| `free`, a budget that binds | `(1, 1)` | 0.82447 | 0.76638 |
| `free`, a budget that binds | `(1e3, 1)` | 0.88919 | 0.76638 |
| `state`, one lever that moves | any | 3.5700e-20 | the same bits |

After, the units move `together`'s bound by at most 2e-15 of itself, where they moved it by up to
2.7 times. They move `free`'s, a bound at rounding, by 3e-8 of itself, where they moved it by 3.4e7
times.

Random linear plants with 2 or 3 levers, horizons of 2 or 3, prices log-uniform on `[1e-3, 10]`,
boxes of unlike widths with the upper side open on 30 % of levers, and a random feasible plan. The
exact regret is the box's quadratic programme solved by enumerating the active sets. On 300 such
plants (`numpy` seed 1):

| | bound over exact regret: median | largest | moved by units `1e-3`, `1e3` |
|---|---|---|---|
| plain norm, before | 2.18 | 4 714 | up to 1e12 times |
| each lever in its own scale | 1.50 | 39.4 | 0, to rounding |

The scaled bound is above the exact regret on every plant. It is not tighter on every plant: on 32
of the 300 it is looser than the plain one, by up to 3.0 times. On the 12 plants of
`tests/test_plan.py` it is looser on 2, by up to 4.9 times. It is looser where the levers'
curvatures are already close and the levers are coupled. On the 32 plants the stiffest lever's mean
curvature is a median 2.0 times the softest's, against 12 on the others. The largest correlation
between two levers' actions in the Hessian is a median 0.55, against 0.17. There the plain norm is
already near the levers' scales, and the scaled block's least eigenvalue charges some moves at
less than the plain one does.

The greedy selection's step bounds on the three named plants of ADR 0004
(`scripts/bench_max_levers.py`), each priced against every lever's box, before and after:

| plant | step 1 | step 2 | step 3 |
|---|---|---|---|
| saturation | 2.2816, 2.2816 | 0.38661, 0.38723 | 4.7e-12, 4.5e-12 |
| driver pool | 0.5124, 0.6037 | 3.88e-5, 8.56e-5 | 8.2e-13, 8.1e-13 |
| marketing mix | refused, refused | 2.3047, 2.5588 | 2.9e-11, 2.7e-11 |

Four of the eight finite bounds are looser after, by up to 2.2 times on the driver pool's second
step. No step changes between finite and refused: `D H D` and `H` have the same inertia, so the modulus
has the same sign at each point drawn.

## Consequences

- **A prescription with two or more levers reports a different regret bound, in any units.** On
  the panel above it is 20 % and 7 % lower in the levers' own units, and the same in all others.
  On most random problems it is lower. On some, where the levers' curvatures are close and the
  levers coupled, it is higher, as on four steps of ADR 0004's named plants.
- **`PlanRegretBound.modulus` is the curvature in the scaled norm.** Its units are those of the
  stiffest moving lever. One lever, or levers of equal curvature, read the plain Hessian's least
  eigenvalue as before.
- **A plan with one moving lever reports the same bits.** The quickstart, notebooks 00 and 09,
  and the BOPTEST case study have one lever, and their numbers do not move.
- No plan moves. The bound is read after the solve, and the lever selection does not rank on it.

## Alternatives

- **Box widths**, `s_j = hi_j - lo_j`. They move with each lever's units and need no Hessian
  diagonal. But a lever may leave a side open, `inf`, and its width then gives no scale. Box
  widths also say nothing about curvature. On 200 plants with closed boxes the box-width bound was
  looser than the plain one on 74 %, by up to 4.3 times. The Jacobi scale was looser on 8 %, by up
  to 1.09 times. Its median bound was 1.13 times the exact regret, where the box widths' was 2.09.
  The Hessian is computed for the modulus anyway, so its diagonal costs nothing.
- **A scale per action**, `s_i = H_ii^(-1/2)`. It also reads no units. On the same 200 plants it
  was as tight as the per-lever scale, with a median of 1.14. But it rescales one lever's steps
  against each other, which no change of units does, and one lever then loses the plain bound.
- **The best diagonal scale**, the `D` that minimises the bound. Every `D` gives a valid bound, so
  a search would be valid wherever it stopped. It would read no units if run in `log s` from the
  Jacobi scale. But it puts an optimiser inside a certificate, at one eigenvalue solve per probe
  per evaluation. It is left until a problem needs the factor the Jacobi scale leaves.
- **The least of the plain and the scaled bounds.** Both are valid, so the least is valid. But the
  plain bound reads the units, and so would the least.
- **Leave the plain norm, and document it.** The bound stays valid. But a certificate that a
  change of units moves by orders of magnitude cannot be compared across problems, or read
  without the units.

## References

- van der Sluis, A. (1969). Condition numbers and equilibration of matrices. *Numerische
  Mathematik* 14, 14–23.
- Nesterov, Y. (2018). *Lectures on Convex Optimization*, 2nd ed., Springer, §2.1 (strong
  convexity in a general norm).
