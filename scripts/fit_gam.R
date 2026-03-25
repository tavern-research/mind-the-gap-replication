#!/usr/bin/env Rscript
# Fit GAM dose-response model via mgcv and write predictions to CSV.
# Called from plot_dose_response.py with args: sample_csv grid_csv output_csv

library(data.table)
library(mgcv)

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) >= 3)
sample_path <- args[1]
grid_path <- args[2]
output_path <- args[3]
dv_col <- if (length(args) >= 4) args[4] else "dv_binary"

# Remaining args: optional weights_col, optional deriv_output_path, optional family_name
# Detect by checking if arg looks like a file path (contains / or .csv)
# or a known family name
weights_col <- NULL
deriv_output_path <- NULL
family_name <- "binomial"
if (length(args) >= 5) {
  for (i in 5:length(args)) {
    if (args[i] %in% c("binomial", "gaussian")) {
      family_name <- args[i]
    } else if (grepl("[/\\\\]|\\.csv$", args[i])) {
      deriv_output_path <- args[i]
    } else {
      weights_col <- args[i]
    }
  }
}

gam_family <- if (family_name == "gaussian") gaussian else binomial
use_logit_link <- family_name == "binomial"
cat("GAM family:", family_name, "\n")

dt <- fread(sample_path)
grid <- fread(grid_path)

cat_vars <- c(
  "potus_2024", "gender",
  "education", "race", "income", "registered_voter"
)

dt[, (cat_vars) := lapply(.SD, as.factor), .SDcols = cat_vars]

# Set grid factors with same levels as training data
for (v in cat_vars) {
  if (v %in% names(grid)) {
    grid[[v]] <- factor(grid[[v]], levels = levels(dt[[v]]))
  }
}

gam_formula <- as.formula(paste(
  dv_col, "~ s(total_duration) +",
  "potus_2024 + gender +",
  "education + race + income + registered_voter +",
  "age + age_missing"
))

gam_args <- list(formula = gam_formula, family = gam_family, data = dt)
if (!is.null(weights_col) && weights_col %in% names(dt)) {
  gam_args$weights <- dt[[weights_col]]
}
fit <- do.call(gam, gam_args)

cat("\nGAM summary:\n")
print(summary(fit))

if ("_grid_idx" %in% names(grid)) {
  # G-computation: predict per-observation, apply inverse link, then average.
  # Avoids Jensen's inequality bias from averaging on the link scale.
  Lp <- predict(fit, newdata = grid, type = "lpmatrix")
  grid_idx <- grid[["_grid_idx"]]
  unique_idx <- sort(unique(grid_idx))
  n_grid <- length(unique_idx)

  beta <- coef(fit)
  V <- vcov(fit)

  dur_vals <- numeric(n_grid)
  means <- numeric(n_grid)
  se <- numeric(n_grid)

  for (i in seq_along(unique_idx)) {
    rows <- which(grid_idx == unique_idx[i])
    dur_vals[i] <- grid[["total_duration"]][rows[1]]
    Lp_i <- Lp[rows, , drop = FALSE]

    # Per-observation predictions on response scale
    eta_i <- as.numeric(Lp_i %*% beta)
    if (use_logit_link) {
      p_i <- plogis(eta_i)
      means[i] <- mean(p_i)
      # Delta method: gradient = mean(p*(1-p)*X)
      w_i <- p_i * (1 - p_i)
      grad_i <- colMeans(w_i * Lp_i)
    } else {
      # Identity link: predictions are already on response scale
      means[i] <- mean(eta_i)
      grad_i <- colMeans(Lp_i)
    }
    se[i] <- sqrt(as.numeric(t(grad_i) %*% V %*% grad_i))
  }

  result <- data.table(
    `_grid_idx` = unique_idx,
    total_duration = dur_vals,
    mean = means,
    ci_low = means - 1.96 * se,
    ci_high = means + 1.96 * se
  )
  fwrite(result, output_path)
} else {
  preds <- predict(fit, newdata = grid, type = "link", se.fit = TRUE)
  if (use_logit_link) {
    grid[, `:=`(
      mean = plogis(preds$fit),
      ci_low = plogis(preds$fit - 1.96 * preds$se.fit),
      ci_high = plogis(preds$fit + 1.96 * preds$se.fit)
    )]
  } else {
    grid[, `:=`(
      mean = preds$fit,
      ci_low = preds$fit - 1.96 * preds$se.fit,
      ci_high = preds$fit + 1.96 * preds$se.fit
    )]
  }
  fwrite(grid[, .(total_duration, mean, ci_low, ci_high)], output_path)
}
cat("GAM predictions written to", output_path, "\n")

# Derivative of marginal means via finite differences
# When n_boot > 0, also compute bootstrap CIs
if (!is.null(deriv_output_path) && "_grid_idx" %in% names(grid)) {
  n_boot <- 0

  beta <- coef(fit)
  Lp <- predict(fit, newdata = grid, type = "lpmatrix")
  grid_idx <- grid[["_grid_idx"]]
  unique_idx <- sort(unique(grid_idx))
  n_grid <- length(unique_idx)

  # Pre-compute row indices for each grid point
  row_lists <- lapply(unique_idx, function(idx) which(grid_idx == idx))
  dur_vals <- sapply(row_lists, function(rows) grid[["total_duration"]][rows[1]])
  h <- diff(dur_vals)

  # Helper: compute marginal means and finite-difference derivative
  compute_deriv <- function(coefs) {
    marginal <- numeric(n_grid)
    for (i in seq_along(unique_idx)) {
      rows <- row_lists[[i]]
      Lp_i <- Lp[rows, , drop = FALSE]
      eta_i <- as.numeric(Lp_i %*% coefs)
      if (use_logit_link) {
        marginal[i] <- mean(plogis(eta_i))
      } else {
        marginal[i] <- mean(eta_i)
      }
    }
    deriv <- numeric(n_grid)
    deriv[1] <- (marginal[2] - marginal[1]) / h[1]
    for (j in 2:(n_grid - 1)) {
      deriv[j] <- (marginal[j + 1] - marginal[j - 1]) / (dur_vals[j + 1] - dur_vals[j - 1])
    }
    deriv[n_grid] <- (marginal[n_grid] - marginal[n_grid - 1]) / h[n_grid - 1]
    deriv
  }

  # Point estimate derivative from fitted coefficients
  point_deriv <- compute_deriv(beta)

  if (n_boot > 0) {
    set.seed(42)
    V_mat <- vcov(fit)
    draws <- MASS::mvrnorm(n_boot, beta, V_mat)
    deriv_mat <- matrix(NA_real_, nrow = n_boot, ncol = n_grid)
    for (b in seq_len(n_boot)) {
      deriv_mat[b, ] <- compute_deriv(draws[b, ])
    }
    deriv_result <- data.table(
      total_duration = dur_vals,
      deriv_mean = point_deriv,
      deriv_ci_low = apply(deriv_mat, 2, quantile, probs = 0.025),
      deriv_ci_high = apply(deriv_mat, 2, quantile, probs = 0.975)
    )
  } else {
    deriv_result <- data.table(
      total_duration = dur_vals,
      deriv_mean = point_deriv
    )
  }

  fwrite(deriv_result, deriv_output_path)
  cat("GAM derivatives written to", deriv_output_path, "\n")
}
