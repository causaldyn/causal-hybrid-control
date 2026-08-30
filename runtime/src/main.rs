//! The measurement harness: one MPC solve per process, or a timed loop inside one.
//!
//! `hyperfine` times whole processes, which is the right unit for "a single-binary MPC step" but
//! charges each run for process start. So the binary reports both: run it bare and `hyperfine`
//! measures the deployable cold path, run it with `steady <n>` and it reports the per-solve time
//! after the first solve has warmed the allocator -- the number a control loop actually pays.

use chc_runtime::optimize_control;
use nalgebra::{dmatrix, dvector, DMatrix, DVector};
use std::time::Instant;

/// The golden LQ instance: a damped oscillator, shared verbatim with `parity_check.py`.
struct Problem {
    a: DMatrix<f64>,
    b: DMatrix<f64>,
    q: DMatrix<f64>,
    r: DMatrix<f64>,
    qf: DMatrix<f64>,
    x_target: DVector<f64>,
    x0: DVector<f64>,
}

fn problem() -> Problem {
    Problem {
        a: dmatrix![0.0, 1.0; -1.0, -0.2],
        b: dmatrix![0.0; 1.0],
        q: dmatrix![1.0, 0.0; 0.0, 0.1],
        r: dmatrix![0.05],
        qf: dmatrix![5.0, 0.0; 0.0, 1.0],
        x_target: dvector![0.0, 0.0],
        x0: dvector![1.0, 0.0],
    }
}

fn solve() -> (Vec<DVector<f64>>, Vec<f64>) {
    let p = problem();
    optimize_control(
        &p.a,
        &p.b,
        &p.q,
        &p.r,
        &p.qf,
        &p.x_target,
        &p.x0,
        0.1,
        -5.0,
        5.0,
        30,
        400,
        0.2,
        1e-9,
    )
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("golden") => {
            let (us, history) = solve();
            println!("cost {:.9}", history.last().unwrap());
            println!("u0 {:.9}", us[0][0]);
            println!("accepted {}", history.len() - 1);
        }
        Some("steady") => {
            let repeats: usize = args.get(2).and_then(|n| n.parse().ok()).unwrap_or(50);
            let (_, warm) = solve(); // first solve warms the allocator; never timed
            let mut times = Vec::with_capacity(repeats);
            for _ in 0..repeats {
                let start = Instant::now();
                let (_us, history) = solve();
                times.push(start.elapsed().as_secs_f64());
                std::hint::black_box(history.len());
            }
            times.sort_by(f64::total_cmp);
            println!("solves {repeats}");
            println!("best_ms {:.4}", times[0] * 1e3);
            println!("median_ms {:.4}", times[repeats / 2] * 1e3);
            std::hint::black_box(warm.len());
        }
        _ => {
            let (_us, history) = solve();
            std::hint::black_box(history.len());
        }
    }
}
