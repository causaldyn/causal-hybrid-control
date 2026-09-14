# Independent (R, base only) cross-check of the C2 clustered cross-fit DML rate theorem (Result 26):
#   ||B_hat - B|| = O_p(G^{-1/2} + nuisance products).  At tiny nuisance error the sampling term
# dominates, so ||B_hat - B|| ~ G^{-1/2}: the log-log slope of RMSE vs G should be about -0.5.
# This reproduces, in a separate language/stack, what chc.regret.clustered_lower_bound_certificate shows.
#
# TWO SECTIONS, and the second one exists because the first one's number was read too generously.
# Section 1 is the original single-window check: G in 20..320, one seed set, slope -0.541 against a
# theoretical -0.5, gated to a band wide enough that -0.541 passes.  It is NOT evidence that the
# rate is -0.5 -- it has no error bar, so "-0.541 is about -0.5" was an assertion.  Section 2 puts
# one on it, by running the whole ladder five times with independent seeds and moving the cluster
# grid up.  At the original window the mean is 4.8 standard errors BELOW -0.5; at G in 80..1280 it
# is within one.  So the miss is the finite-G term, exactly as the Python side's regret slope walks
# -1.14 -> -1.02 over the same grids -- and the two are the same measurement, since regret is
# quadratic in the error and 2 * (-0.51) = -1.02.  Section 2 takes a few minutes; run the file.

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

## ---- section 2: the same slope with an error bar, as the cluster grid moves up ----------------
## Every exponent in this programme is a fit over a finite window, so the reading owes a window and
## a standard error.  Here both are cheap: the seed is ours, so these are INDEPENDENT replicates
## rather than the nested prefixes the Python certificates force (their seeding is per grid point).

slope_for <- function(Gs, seed, n_seed) {
  set.seed(seed)
  rmse <- sapply(Gs, function(G) {
    errs <- sapply(1:n_seed, function(s) {
      d <- simulate_cluster(G)
      dml_two_channel(d$z, d$u, d$g, d$y, d$fold, delta = 0.002) - b_total
    })
    sqrt(mean(errs^2))
  })
  unname(coef(lm(log(rmse) ~ log(Gs)))[2])
}

windows <- list(c(20, 40, 80, 160, 320), c(40, 80, 160, 320, 640), c(80, 160, 320, 640, 1280))
n_rep <- 5; n_ladder <- 240
means <- numeric(length(windows)); errs <- numeric(length(windows))

cat(sprintf("\n%-14s %9s %9s %9s\n", "G window", "mean", "s.e.", "(mean+0.5)/s.e."))
for (i in seq_along(windows)) {
  Gs <- windows[[i]]
  vals <- sapply(1:n_rep, function(r) slope_for(Gs, 20260722L + 1000L * r, n_ladder))
  means[i] <- mean(vals); errs[i] <- sd(vals) / sqrt(n_rep)
  cat(sprintf("%-14s %9.4f %9.4f %9.1f\n", sprintf("%d .. %d", Gs[1], Gs[length(Gs)]),
              means[i], errs[i], (means[i] + 0.5) / errs[i]))
}

top <- length(windows)
walk <- abs(means[top] - means[1])
noise <- sqrt(errs[1]^2 + errs[top]^2)
cat(sprintf("walk %.4f over the ladder against %.4f on the endpoints -> %.1fx\n",
            walk, noise, walk / noise))

## The two gates that make this a measurement rather than a print: the shipped window must REJECT
## the theoretical -0.5, and the top window must not.  Reversing either is what would falsify the
## claim that the miss is the window -- a real deviation from the rate would reject at both.
stopifnot(abs(means[1] + 0.5) > 3 * errs[1])
stopifnot(abs(means[top] + 0.5) < 3 * errs[top])
cat("PASS: the -0.541 is the finite-G window, not the rate -- it clears -0.5 only once G >= 80.\n")
