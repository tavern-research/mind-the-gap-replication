"""Dose-response model fitting functions.

Contains utilities for covariate formula construction, g-computation marginal
predictions, adjusted cell means, equivalence tests, and parametric/GAM/GP
curve fitting.
"""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import build_design_matrices, dmatrices
from scipy.optimize import curve_fit
from scipy.special import expit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

logger = logging.getLogger(__name__)

CATEGORICAL_COVARIATES = ["potus_2024", "gender", "education", "race", "income", "registered_voter"]

PHASE2_DUR: dict[str, float] = {
    "path_a_novel": 15.0,
    "path_a_repeat": 15.0,
    "path_b1_dtc": 15.0,
    "path_b2_trad": 15.0,
    "path_b3_alt_30s": 30.0,
    "none": 0.0,
}


def build_covariate_formula(
    sample: pd.DataFrame,
    categorical_covariates: list[str] | None = None,
    continuous_covariates: list[str] | None = None,
) -> str:
    """Build covariate formula fragment, dropping single-level categoricals."""
    if categorical_covariates is None:
        categorical_covariates = CATEGORICAL_COVARIATES
    if continuous_covariates is None:
        continuous_covariates = ["age"]

    parts: list[str] = []
    for col in categorical_covariates:
        if col in sample.columns and sample[col].nunique() > 1:
            parts.append(f"C({col})")
    parts.extend(continuous_covariates)
    if "age_missing" in sample.columns and sample["age_missing"].sum() > 0:
        parts.append("age_missing")
    return " + ".join(parts)


def marginal_prediction_frame(
    sample: pd.DataFrame,
    duration_values: np.ndarray,
    overrides: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build prediction DataFrame for marginal means (g-computation).

    For each duration value, replicates the full sample with total_duration
    set to that value. Predictions averaged over this frame yield
    population-average (marginal) effects.

    Parameters
    ----------
    overrides : optional dict mapping column names to values. The special
        sentinel ``"_dose_"`` means "set to the current grid value *d*".
    """
    frames = []
    for i, d in enumerate(duration_values):
        frame = sample.copy()
        frame["total_duration"] = d
        frame["log_total_duration"] = np.log(max(d, 0.1))
        frame["_grid_idx"] = i
        if overrides:
            for col, val in overrides.items():
                frame[col] = d if val == "_dose_" else val
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _marginal_mean_predictions(
    model: object,
    pred_df: pd.DataFrame,
    n_grid: int,
    n_sample: int,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compute marginal mean predictions with CIs.

    For GLM (logit link): g-computation with correct averaging order.
    Predicts per-observation on link scale, applies inverse link, then
    averages. Uses delta method for SEs.

    For OLS/WLS (identity link): averages design matrix then uses t_test.
    """
    try:
        design_info = model.model.data.orig_exog.design_info
    except AttributeError:
        _, X_rebuild = dmatrices(model.model.formula, model.model.data.frame)
        design_info = X_rebuild.design_info

    (X,) = build_design_matrices([design_info], pred_df)
    X_all = np.asarray(X).reshape(n_grid, n_sample, -1)

    is_glm = hasattr(model.model, 'family')

    if is_glm:
        beta = np.asarray(model.params)
        V = np.asarray(model.cov_params())

        # Per-observation linear predictors: (n_grid, n_sample)
        eta_all = X_all @ beta
        p_all = expit(eta_all)
        mean_vals = p_all.mean(axis=1)

        # Delta method: d(mean_p)/d(beta) = mean(p*(1-p)*X)
        weights = p_all * (1 - p_all)  # (n_grid, n_sample)
        grad = (weights[:, :, np.newaxis] * X_all).mean(axis=1)  # (n_grid, n_params)
        se = np.sqrt(np.einsum('gp,pq,gq->g', grad, V, grad))

        z = 1.96
        ci_low = mean_vals - z * se
        ci_high = mean_vals + z * se
    else:
        X_avg = X_all.mean(axis=1)
        result = model.t_test(X_avg)
        ci = result.conf_int(alpha=alpha)
        mean_vals = result.effect
        ci_low = ci[:, 0]
        ci_high = ci[:, 1]

    return pd.DataFrame({
        "mean": mean_vals,
        "ci_low": ci_low,
        "ci_high": ci_high,
    })


def adjusted_cell_means(
    sample: pd.DataFrame,
    dv_col: str = "dv_binary",
    categorical_covariates: list[str] | None = None,
    weights: str | None = "quality_weight",
    by_path: bool = False,
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """Compute covariate-adjusted cell means with 95% CIs via saturated WLS.

    When by_path=True, returns a dict mapping each var_phase2_path level to a
    DataFrame of cell means for durations observed under that path.
    """
    covariates = build_covariate_formula(
        sample, categorical_covariates=categorical_covariates
    )

    if by_path:
        # Approach B: model phase1_dur conditional on path
        formula = f"{dv_col} ~ C(phase1_dur) + C(var_phase2_path) + {covariates}"
        if weights and weights in sample.columns:
            model = smf.wls(formula, data=sample, weights=sample[weights]).fit(cov_type="HC2")
        else:
            model = smf.ols(formula, data=sample).fit(cov_type="HC2")

        path_levels = sorted(sample["var_phase2_path"].unique())
        result: dict[str, pd.DataFrame] = {}
        for path in path_levels:
            path_mask = sample["var_phase2_path"] == path
            phase1_vals = np.sort(sample.loc[path_mask, "phase1_dur"].unique())
            phase2_dur = PHASE2_DUR.get(path, 0.0)
            pred_df = marginal_prediction_frame(
                sample, phase1_vals,
                overrides={"var_phase2_path": path, "phase1_dur": "_dose_"},
            )
            averaged = _marginal_mean_predictions(model, pred_df, len(phase1_vals), len(sample))
            cell_sizes = sample[path_mask].groupby("phase1_dur").size()
            total_durations = phase1_vals + phase2_dur
            result[path] = pd.DataFrame({
                "duration": total_durations,
                "mean": averaged["mean"].values,
                "ci_low": averaged["ci_low"].values,
                "ci_high": averaged["ci_high"].values,
                "n": [cell_sizes[p] for p in phase1_vals],
            })
        return result

    # Approach A: marginal dose-response (no path, no phase1_dur)
    formula = f"{dv_col} ~ C(total_duration) + {covariates}"
    if weights and weights in sample.columns:
        model = smf.wls(formula, data=sample, weights=sample[weights]).fit(cov_type="HC2")
    else:
        model = smf.ols(formula, data=sample).fit(cov_type="HC2")

    durations = np.sort(sample["total_duration"].unique())
    pred_df = marginal_prediction_frame(sample, durations)
    averaged = _marginal_mean_predictions(model, pred_df, len(durations), len(sample))

    cell_sizes = sample.groupby("total_duration").size()

    return pd.DataFrame({
        "duration": durations,
        "mean": averaged["mean"].values,
        "ci_low": averaged["ci_low"].values,
        "ci_high": averaged["ci_high"].values,
        "n": [cell_sizes[d] for d in durations],
    })


def fit_model(
    formula: str,
    sample: pd.DataFrame,
    weights: str | None = "quality_weight",
    use_glm: bool = True,
) -> object:
    """Fit GLM (binomial) or WLS/OLS with optional quality weights."""
    use_weights = weights and weights in sample.columns
    if use_glm:
        kwargs: dict = {"formula": formula, "data": sample, "family": sm.families.Binomial()}
        if use_weights:
            kwargs["var_weights"] = sample[weights]
        return smf.glm(**kwargs).fit()
    if use_weights:
        return smf.wls(formula, data=sample, weights=sample[weights]).fit(cov_type="HC2")
    return smf.ols(formula, data=sample).fit(cov_type="HC2")


def fit_parametric_curves(
    sample: pd.DataFrame,
    grid: np.ndarray,
    dv_col: str = "dv_binary",
    categorical_covariates: list[str] | None = None,
    weights: str | None = "quality_weight",
    use_glm: bool = True,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """Fit parametric dose-response models (Logit-log).

    When use_glm=True (default), fits GLM with binomial family.
    When use_glm=False, fits WLS with HC2 (identity link, for continuous DVs).

    Returns (predictions, models) where predictions maps model name to DataFrame
    with columns: total_duration, mean, ci_low, ci_high; and models maps name
    to the fitted statsmodels GLM result.
    """
    covariates = build_covariate_formula(
        sample, categorical_covariates=categorical_covariates
    )
    specs = {
        "Logit (log)": f"{dv_col} ~ log_total_duration + {covariates}",
    }
    pred_df = marginal_prediction_frame(sample, grid)
    n_sample = len(sample)
    n_grid = len(grid)
    results: dict[str, pd.DataFrame] = {}
    models: dict[str, object] = {}
    for name, formula in specs.items():
        try:
            model = fit_model(formula, sample, weights=weights, use_glm=use_glm)
            averaged = _marginal_mean_predictions(model, pred_df, n_grid, n_sample)
            results[name] = pd.DataFrame({
                "total_duration": grid,
                "mean": averaged["mean"].values,
                "ci_low": averaged["ci_low"].values,
                "ci_high": averaged["ci_high"].values,
            })
            models[name] = model
        except Exception as e:
            logger.warning("%s model failed: %s", name, e)
    return results, models


def path_conditional_predictions(
    sample: pd.DataFrame,
    grid: np.ndarray,
    dv_col: str = "dv_binary",
    categorical_covariates: list[str] | None = None,
    weights: str | None = "quality_weight",
    use_glm: bool = True,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Fit path-conditional models and generate per-path predictions (Approach B).

    Fits additive models on phase1_dur conditional on var_phase2_path,
    then generates predictions for each path by varying phase1_dur over
    its observed range. Returns total_duration on the x-axis (phase1_dur +
    phase2_dur(path)) for display consistency.

    Returns nested dict: {model_name: {path_level: prediction_df}}.
    Each prediction_df has columns: total_duration, mean, ci_low, ci_high.
    """
    covariates = build_covariate_formula(
        sample, categorical_covariates=categorical_covariates
    )
    specs = {
        "Logit (log)": f"{dv_col} ~ log_phase1_dur + C(var_phase2_path) + {covariates}",
    }

    n_sample = len(sample)

    sample = sample.copy()
    sample["log_phase1_dur"] = np.log(np.clip(sample["phase1_dur"], 0.1, None))

    fitted: dict[str, object] = {}
    for name, formula in specs.items():
        try:
            fitted[name] = fit_model(formula, sample, weights=weights, use_glm=use_glm)
        except Exception as e:
            logger.warning("Path-conditional %s model failed: %s", name, e)

    path_levels = sorted(sample["var_phase2_path"].unique())
    result: dict[str, dict[str, pd.DataFrame]] = {}

    for name, model in fitted.items():
        result[name] = {}
        for path in path_levels:
            try:
                phase2_dur = PHASE2_DUR.get(path, 0.0)
                path_mask = sample["var_phase2_path"] == path
                p1_min = sample.loc[path_mask, "phase1_dur"].min()
                p1_max = sample.loc[path_mask, "phase1_dur"].max()
                p1_grid = np.linspace(p1_min, p1_max, len(grid))

                pred_df = marginal_prediction_frame(
                    sample, p1_grid,
                    overrides={
                        "var_phase2_path": path,
                        "phase1_dur": "_dose_",
                        "log_phase1_dur": "_dose_log_",
                    },
                )
                # Fix log_phase1_dur: marginal_prediction_frame doesn't handle _dose_log_
                pred_df["log_phase1_dur"] = np.log(np.clip(pred_df["phase1_dur"], 0.1, None))

                n_grid = len(p1_grid)
                averaged = _marginal_mean_predictions(model, pred_df, n_grid, n_sample)
                total_dur = p1_grid + phase2_dur
                result[name][path] = pd.DataFrame({
                    "total_duration": total_dur,
                    "mean": averaged["mean"].values,
                    "ci_low": averaged["ci_low"].values,
                    "ci_high": averaged["ci_high"].values,
                })
            except Exception as e:
                logger.warning("%s path-conditional prediction failed for %s: %s", name, path, e)
    return result


def fit_gam(
    sample: pd.DataFrame,
    grid: np.ndarray,
    dv_col: str = "dv_binary",
    categorical_covariates: list[str] | None = None,
    weights: str | None = "quality_weight",
    overrides: dict[str, Any] | None = None,
    family: str = "binomial",
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Fit GAM via R/mgcv subprocess.

    Returns (prediction_df, derivative_df) where derivative_df has columns
    total_duration, deriv_mean, deriv_ci_low, deriv_ci_high. Either may be None.

    overrides : passed to marginal_prediction_frame for path-conditional predictions.
    family : "binomial" (default) or "gaussian" for continuous DVs.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        sample_csv = tmp / "sample.csv"
        grid_csv = tmp / "grid.csv"
        output_csv = tmp / "gam_preds.csv"
        deriv_csv = tmp / "gam_deriv.csv"

        sample.to_csv(sample_csv, index=False)
        pred_df = marginal_prediction_frame(sample, grid, overrides=overrides)
        pred_df.to_csv(grid_csv, index=False)

        cmd = [
                "Rscript",
                str(Path(__file__).parent / "fit_gam.R"),
                str(sample_csv),
                str(grid_csv),
                str(output_csv),
                dv_col,
        ]
        if weights and weights in sample.columns:
            cmd.append(weights)
        cmd.append(str(deriv_csv))
        cmd.append(family)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        pred_result = None
        deriv_result = None

        if result.returncode == 0 and output_csv.exists():
            raw = pd.read_csv(output_csv)
            pred_result = pd.DataFrame({
                "total_duration": grid,
                "mean": raw["mean"].values,
                "ci_low": raw["ci_low"].values,
                "ci_high": raw["ci_high"].values,
            })
            if deriv_csv.exists():
                deriv_raw = pd.read_csv(deriv_csv)
                deriv_dict: dict[str, np.ndarray] = {
                    "total_duration": deriv_raw["total_duration"].values,
                    "deriv_mean": deriv_raw["deriv_mean"].values,
                }
                if "deriv_ci_low" in deriv_raw.columns:
                    deriv_dict["deriv_ci_low"] = deriv_raw["deriv_ci_low"].values
                    deriv_dict["deriv_ci_high"] = deriv_raw["deriv_ci_high"].values
                deriv_result = pd.DataFrame(deriv_dict)
        else:
            logger.warning("GAM fitting failed:\n%s", result.stderr)

        return pred_result, deriv_result


def _emax_func(
    dose: np.ndarray, e0: float, emax: float, ec50: float, n: float,
) -> np.ndarray:
    """Sigmoid Emax (Hill) model."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = e0 + emax * dose**n / (ec50**n + dose**n)
    # dose=0 produces 0/0 -> nan; the limit is e0
    result = np.where(np.isfinite(result), result, e0)
    return result


def fit_emax(
    cells: pd.DataFrame, grid: np.ndarray,
) -> tuple[pd.DataFrame | None, dict[str, float], pd.DataFrame | None]:
    """Fit Emax (Hill) model on adjusted cell means, weighted by cell size.

    Returns (prediction_df, params_dict, derivative_df). prediction_df and
    derivative_df are None on failure.
    """
    try:
        durations = cells["duration"].values
        means = cells["mean"].values
        weights = np.sqrt(cells["n"].values)

        popt, _ = curve_fit(
            _emax_func,
            durations,
            means,
            p0=[means[0], means[-1] - means[0], np.median(durations), 1.0],
            sigma=1.0 / weights,
            maxfev=20000,
        )
        e0, emax, ec50, n = popt
        params = {"E0": e0, "Emax": emax, "EC50": ec50, "n": n}

        pred = _emax_func(grid, *popt)
        pred_df = pd.DataFrame({"total_duration": grid, "mean": pred})

        x = np.clip(grid, 0.1, None)
        ec50n = ec50 ** n
        xn = x ** n
        deriv = emax * n * ec50n * x ** (n - 1) / (ec50n + xn) ** 2
        deriv_df = pd.DataFrame({"total_duration": grid, "deriv_mean": deriv})

        return pred_df, params, deriv_df
    except Exception as e:
        logger.warning("Emax fit failed: %s", e)
        return None, {}, None


def fit_gp(
    cells: pd.DataFrame,
    grid: np.ndarray,
    duration_col: str = "duration",
    clip_bounds: tuple[float, float] | None = (0.0, 1.0),
) -> tuple[pd.DataFrame | None, dict | None]:
    """Fit Gaussian Process (Matern kernel) on adjusted cell means.

    Returns (prediction_df, kernel_params_dict) or (None, None) on failure.
    prediction_df has columns: total_duration, mean, ci_low, ci_high.
    """
    try:
        durations = cells[duration_col].values.reshape(-1, 1)
        means = cells["mean"].values

        kernel = Matern(length_scale=10.0, nu=2.5, length_scale_bounds=(5, 60)) + WhiteKernel(noise_level=1e-3)
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            random_state=42,
        )
        gp.fit(durations, means)

        grid_2d = grid.reshape(-1, 1)
        pred_mean, pred_std = gp.predict(grid_2d, return_std=True)

        if clip_bounds is not None:
            lo, hi = clip_bounds
            pred_mean = np.clip(pred_mean, lo, hi)
            ci_low = np.clip(pred_mean - 1.96 * pred_std, lo, hi)
            ci_high = np.clip(pred_mean + 1.96 * pred_std, lo, hi)
        else:
            ci_low = pred_mean - 1.96 * pred_std
            ci_high = pred_mean + 1.96 * pred_std

        optimized = gp.kernel_
        kernel_params = {
            "length_scale": float(optimized.get_params()["k1__length_scale"]),
            "noise_level": float(optimized.get_params()["k2__noise_level"]),
        }
        logger.info(
            "GP kernel params: length_scale=%.2f, noise_level=%.6f",
            kernel_params["length_scale"],
            kernel_params["noise_level"],
        )

        pred_df = pd.DataFrame({
            "total_duration": grid,
            "mean": pred_mean,
            "ci_low": ci_low,
            "ci_high": ci_high,
        })
        return pred_df, kernel_params
    except Exception as e:
        logger.warning("GP fit failed: %s", e)
        return None, None
