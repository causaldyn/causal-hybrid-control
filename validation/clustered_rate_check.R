# Independent (R, base only) cross-check of the C2 clustered cross-fit DML rate theorem (Result 26):
#   ||B_hat - B|| = O_p(G^{-1/2} + nuisance products).  At tiny nuisance error the sampling term
# dominates, so ||B_hat - B|| ~ G^{-1/2}: the log-log slope of RMSE vs G should be about -0.5.
# This reproduces, in a separate language/stack, what chc.regret.clustered_lower_bound_certificate shows.

set.seed(20260722)

## cross-fit two-channel Robinson DML for the clustered PLR  Y = b_d U + b_s Gexp + gamma Z + a_g + eps
dml_two_channel <- function(z, u, g, y, fold, delta = 0.0) {
  resid <- function(t) {
    r <- t
    for (f in 0:1) {
      tr <- fold != f; te <- fold == f
      fit <- lm(t[tr] ~ z[tr])                       # cross-fit nuisance E[t|Z]
      b0 <- coef(fit)[1]; b1 <- coef(fit)[2] + delta # + delta*z systematic nuisance error
      r[te] <- t[te] - (b1 * z[te] + b0)
    }
    r
  }
  ut <- resid(u); gt <- resid(g); yt <- resid(y)
  beta <- coef(lm(yt ~ ut + gt - 1))                 # joint Robinson (both channels orthogonal)
  unname(beta[1] + beta[2])                          # total effect B = b_d + b_s
}

simulate_cluster <- function(G, m = 10, b_d = 1.0, b_s = 0.6,
                             au = 1.0, ag = 0.8, gamma = 1.0, tau = 0.5, noise = 0.5) {
  n <- G * m; cid <- rep(1:G, each = m)
  z <- rnorm(n)
  a <- rnorm(G, sd = tau)[cid]                       # within-cluster random effect (dependence)
  u <- au * z + 0.7 * rnorm(n)
  g <- ag * z + 0.7 * rnorm(n)
  y <- b_d * u + b_s * g + gamma * z + a + noise * rnorm(n)
  list(z = z, u = u, g = g, y = y, fold = (cid %% 2))   # A8: whole clusters held out
}

Gs <- c(20, 40, 80, 160, 320); n_seed <- 80; b_total <- 1.6
rmse <- sapply(Gs, function(G) {
  errs <- sapply(1:n_seed, function(s) {
    d <- simulate_cluster(G)
    dml_two_channel(d$z, d$u, d$g, d$y, d$fold, delta = 0.002) - b_total
  })
  sqrt(mean(errs^2))
})

slope <- coef(lm(log(rmse) ~ log(Gs)))[2]
cat(sprintf("G        : %s\n", paste(Gs, collapse = "  ")))
cat(sprintf("RMSE     : %s\n", paste(sprintf("%.4f", rmse), collapse = "  ")))
cat(sprintf("sqrt(G)*RMSE (flat if rate is G^-1/2): %s\n",
            paste(sprintf("%.4f", sqrt(Gs) * rmse), collapse = "  ")))
cat(sprintf("log-log slope of RMSE vs G = %.3f  (theory: -0.5)\n", slope))
stopifnot(slope > -0.62, slope < -0.38)
cat("PASS: clustered cross-fit DML total-effect error scales as G^{-1/2} (Result 26, T1 term).\n")
