//! Rust runtime for Causal Hybrid Control (v2, plans/04).
//!
//! A deployable mirror of the proven Python control loop for a linear(-ised) known system: RK4
//! discretisation, a hand-written discrete adjoint, and projected-gradient optimal control with box
//! constraints. The numerical contracts match `chc` exactly (verified against golden trajectories),
//! so the same policy runs as a single fast, memory-safe binary. Stiff ODE / conic MPC backends
//! (diffsol + Enzyme, Clarabel) slot in later; this first slice is dependency-light on purpose.

use nalgebra::{DMatrix, DVector};

/// Linear vector field ẋ = A x + B u.
fn field(a: &DMatrix<f64>, b: &DMatrix<f64>, x: &DVector<f64>, u: &DVector<f64>) -> DVector<f64> {
    a * x + b * u
}

/// One RK4 step of the linear system with zero-order-hold control.
fn rk4_step(
    a: &DMatrix<f64>,
    b: &DMatrix<f64>,
    x: &DVector<f64>,
    u: &DVector<f64>,
    dt: f64,
) -> DVector<f64> {
    let k1 = field(a, b, x, u);
    let k2 = field(a, b, &(x + &(&k1 * (dt / 2.0))), u);
    let k3 = field(a, b, &(x + &(&k2 * (dt / 2.0))), u);
    let k4 = field(a, b, &(x + &(&k3 * dt)), u);
    x + &((k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6.0))
}

/// Discrete step matrices (A_d, B_d) so that x_{k+1} = A_d x_k + B_d u_k (RK4 is exact for a linear ODE).
pub fn discretize(a: &DMatrix<f64>, b: &DMatrix<f64>, dt: f64) -> (DMatrix<f64>, DMatrix<f64>) {
    let n = a.nrows();
    let m = b.ncols();
    let zero_u = DVector::zeros(m);
    let zero_x = DVector::zeros(n);
    let mut a_d = DMatrix::zeros(n, n);
    for i in 0..n {
        let mut e = DVector::zeros(n);
        e[i] = 1.0;
        a_d.set_column(i, &rk4_step(a, b, &e, &zero_u, dt));
    }
    let mut b_d = DMatrix::zeros(n, m);
    for j in 0..m {
        let mut e = DVector::zeros(m);
        e[j] = 1.0;
        b_d.set_column(j, &rk4_step(a, b, &zero_x, &e, dt));
    }
    (a_d, b_d)
}

fn rollout(
    a_d: &DMatrix<f64>,
    b_d: &DMatrix<f64>,
    x0: &DVector<f64>,
    us: &[DVector<f64>],
) -> Vec<DVector<f64>> {
    let mut xs = Vec::with_capacity(us.len() + 1);
    xs.push(x0.clone());
    for u in us {
        let x = xs.last().unwrap();
        xs.push(a_d * x + b_d * u);
    }
    xs
}

fn quad(m: &DMatrix<f64>, v: &DVector<f64>) -> f64 {
    v.dot(&(m * v))
}

fn total_cost(
    xs: &[DVector<f64>],
    us: &[DVector<f64>],
    q: &DMatrix<f64>,
    r: &DMatrix<f64>,
    qf: &DMatrix<f64>,
    x_target: &DVector<f64>,
) -> f64 {
    let h = us.len();
    let mut cost = 0.0;
    for k in 0..h {
        cost += 0.5 * quad(q, &(&xs[k] - x_target)) + 0.5 * quad(r, &us[k]);
    }
    cost + 0.5 * quad(qf, &(&xs[h] - x_target))
}

/// Discrete adjoint: gradient of the Bolza cost w.r.t. each control (mirrors chc.adjoint).
#[allow(clippy::too_many_arguments)]
fn control_gradient(
    a_d: &DMatrix<f64>,
    b_d: &DMatrix<f64>,
    q: &DMatrix<f64>,
    r: &DMatrix<f64>,
    qf: &DMatrix<f64>,
    x_target: &DVector<f64>,
    xs: &[DVector<f64>],
    us: &[DVector<f64>],
) -> Vec<DVector<f64>> {
    let h = us.len();
    let a_t = a_d.transpose();
    let b_t = b_d.transpose();
    let mut lam = qf * (&xs[h] - x_target);
    let mut grads = vec![DVector::zeros(us[0].len()); h];
    for k in (0..h).rev() {
        grads[k] = r * &us[k] + &b_t * &lam;
        lam = q * (&xs[k] - x_target) + &a_t * &lam;
    }
    grads
}

fn clip(v: &DVector<f64>, lo: f64, hi: f64) -> DVector<f64> {
    v.map(|x| x.clamp(lo, hi))
}

/// Projected-gradient optimal control with backtracking (monotone descent). Returns the optimised
/// control sequence and the cost history.
#[allow(clippy::too_many_arguments)]
pub fn optimize_control(
    a: &DMatrix<f64>,
    b: &DMatrix<f64>,
    q: &DMatrix<f64>,
    r: &DMatrix<f64>,
    qf: &DMatrix<f64>,
    x_target: &DVector<f64>,
    x0: &DVector<f64>,
    dt: f64,
    u_lo: f64,
    u_hi: f64,
    horizon: usize,
    steps: usize,
    lr0: f64,
) -> (Vec<DVector<f64>>, Vec<f64>) {
    let (a_d, b_d) = discretize(a, b, dt);
    let m = b.ncols();
    let mut us: Vec<DVector<f64>> = (0..horizon).map(|_| DVector::zeros(m)).collect();
    let mut current = total_cost(&rollout(&a_d, &b_d, x0, &us), &us, q, r, qf, x_target);
    let mut history = vec![current];

    for _ in 0..steps {
        let xs = rollout(&a_d, &b_d, x0, &us);
        let grad = control_gradient(&a_d, &b_d, q, r, qf, x_target, &xs, &us);
        let mut lr = lr0;
        let mut improved = false;
        for _ in 0..40 {
            let cand: Vec<DVector<f64>> = us
                .iter()
                .zip(&grad)
                .map(|(u, g)| clip(&(u - &(g * lr)), u_lo, u_hi))
                .collect();
            let cand_cost = total_cost(&rollout(&a_d, &b_d, x0, &cand), &cand, q, r, qf, x_target);
            if cand_cost < current - 1e-12 {
                us = cand;
                current = cand_cost;
                history.push(current);
                improved = true;
                break;
            }
            lr *= 0.5;
        }
        if !improved {
            break;
        }
    }
    (us, history)
}

#[cfg(feature = "python")]
mod python {
    use crate::optimize_control as core_optimize_control;
    use nalgebra::{DMatrix, DVector};
    use pyo3::prelude::*;

    fn matrix(rows: &[Vec<f64>]) -> DMatrix<f64> {
        DMatrix::from_fn(rows.len(), rows[0].len(), |i, j| rows[i][j])
    }

    /// Solve the constrained LQ optimal-control problem; returns (controls, cost_history).
    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn optimize_control(
        a: Vec<Vec<f64>>,
        b: Vec<Vec<f64>>,
        q: Vec<Vec<f64>>,
        r: Vec<Vec<f64>>,
        qf: Vec<Vec<f64>>,
        x_target: Vec<f64>,
        x0: Vec<f64>,
        dt: f64,
        u_lo: f64,
        u_hi: f64,
        horizon: usize,
        steps: usize,
        lr0: f64,
    ) -> (Vec<Vec<f64>>, Vec<f64>) {
        let (us, history) = core_optimize_control(
            &matrix(&a),
            &matrix(&b),
            &matrix(&q),
            &matrix(&r),
            &matrix(&qf),
            &DVector::from_vec(x_target),
            &DVector::from_vec(x0),
            dt,
            u_lo,
            u_hi,
            horizon,
            steps,
            lr0,
        );
        (
            us.into_iter()
                .map(|u| u.iter().copied().collect())
                .collect(),
            history,
        )
    }

    #[pymodule]
    fn chc_runtime(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(optimize_control, m)?)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{dmatrix, dvector};

    // Damped oscillator omega=1, zeta=0.1 (matches the Python golden problem).
    #[allow(clippy::type_complexity)]
    fn problem() -> (
        DMatrix<f64>,
        DMatrix<f64>,
        DMatrix<f64>,
        DMatrix<f64>,
        DMatrix<f64>,
    ) {
        let a = dmatrix![0.0, 1.0; -1.0, -0.2];
        let b = dmatrix![0.0; 1.0];
        let q = dmatrix![1.0, 0.0; 0.0, 0.1];
        let r = dmatrix![0.05];
        let qf = dmatrix![5.0, 0.0; 0.0, 1.0];
        (a, b, q, r, qf)
    }

    #[test]
    fn discretize_matches_python_golden() {
        let (a, b, ..) = problem();
        let (a_d, b_d) = discretize(&a, &b, 0.1);
        // golden values from chc.lqr.linearize_discrete (float64)
        assert!((a_d[(0, 0)] - 0.9950373333).abs() < 1e-6);
        assert!((a_d[(0, 1)] - 0.0988416333).abs() < 1e-6);
        assert!((a_d[(1, 0)] + 0.0988416333).abs() < 1e-6);
        assert!((a_d[(1, 1)] - 0.9752690067).abs() < 1e-6);
        assert!((b_d[(0, 0)] - 0.0049626667).abs() < 1e-6);
        assert!((b_d[(1, 0)] - 0.0988416333).abs() < 1e-6);
    }

    #[test]
    fn adjoint_matches_finite_difference() {
        let (a, b, q, r, qf) = problem();
        let (a_d, b_d) = discretize(&a, &b, 0.1);
        let x0 = dvector![1.0, 0.0];
        let xt = dvector![0.0, 0.0];
        let us: Vec<DVector<f64>> = (0..12).map(|k| dvector![0.1 * (k as f64) - 0.5]).collect();
        let xs = rollout(&a_d, &b_d, &x0, &us);
        let grad = control_gradient(&a_d, &b_d, &q, &r, &qf, &xt, &xs, &us);
        let eps = 1e-6;
        for k in [0usize, 5, 11] {
            let mut up = us.clone();
            let mut um = us.clone();
            up[k][0] += eps;
            um[k][0] -= eps;
            let jp = total_cost(&rollout(&a_d, &b_d, &x0, &up), &up, &q, &r, &qf, &xt);
            let jm = total_cost(&rollout(&a_d, &b_d, &x0, &um), &um, &q, &r, &qf, &xt);
            let fd = (jp - jm) / (2.0 * eps);
            assert!(
                (grad[k][0] - fd).abs() < 1e-5,
                "k={k} adjoint={} fd={fd}",
                grad[k][0]
            );
        }
    }

    #[test]
    fn optimize_control_reaches_python_optimum() {
        let (a, b, q, r, qf) = problem();
        let x0 = dvector![1.0, 0.0];
        let xt = dvector![0.0, 0.0];
        let (_us, history) =
            optimize_control(&a, &b, &q, &r, &qf, &xt, &x0, 0.1, -5.0, 5.0, 30, 400, 0.2);
        let final_cost = *history.last().unwrap();
        // Python DLQR optimum = 3.68618109; PG-OC converges to ~3.6862.
        assert!(
            final_cost >= 3.68618109 - 1e-4,
            "cost {final_cost} below optimum"
        );
        assert!(final_cost < 3.70, "cost {final_cost} not converged");
        assert!(
            history.first().unwrap() > &(2.0 * final_cost),
            "cost did not decrease enough"
        );
    }
}
