#!/usr/bin/env python
"""Dose-response analysis pipeline for release package.

Reads the analysis-ready dataset from 1_prepare_data.py, fits dose-response
models, runs equivalence tests, and generates all visualization outputs.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from statsmodels.tools.sm_exceptions import ValueWarning

from test_equivalence import run_balance_tests, run_dose_response_equivalence
from estimate_dose_response import (
    CATEGORICAL_COVARIATES,
    adjusted_cell_means,
    fit_emax,
    fit_gam,
    fit_gp,
    fit_parametric_curves,
    path_conditional_predictions,
)
from visualize_results import (
    plot_cell_means,
    plot_derivatives,
    plot_dose_response,
    plot_dose_response_by_path,
    plot_emax_derivative,
)

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message="covariance of constraints does not have full rank",
        category=ValueWarning,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    joined = pl.read_parquet(DATA_DIR / "analysis_ready.parquet")
    sample = joined.to_pandas()
    if "var_phase2_path" not in sample.columns:
        sample["var_phase2_path"] = "none"
    if "phase1_dur" not in sample.columns:
        sample["phase1_dur"] = sample["total_duration"]
    sample["log_total_duration"] = np.log(np.clip(sample["total_duration"], 0.1, None))

    n = len(sample)
    dv_col = "dv_binary"
    ylabel = "Pr(Vote for Tom Martin)"
    title_suffix = f"Vote Choice (Combined N={n:,})"
    prefix = "combined_dose_response"

    cells = adjusted_cell_means(sample, dv_col=dv_col)
    grid = np.linspace(0, sample["total_duration"].max(), 200)

    # Fit models
    curves: dict[str, pd.DataFrame] = {}

    parametric_preds, _ = fit_parametric_curves(sample, grid, dv_col=dv_col, use_glm=True)
    curves.update(parametric_preds)

    gam_pred, gam_deriv = fit_gam(sample, grid, dv_col=dv_col, family="binomial")
    if gam_pred is not None:
        curves["GAM (mgcv)"] = gam_pred

    emax_pred, _, emax_deriv_df = fit_emax(cells, grid)
    if emax_pred is not None:
        curves["Emax (Hill)"] = emax_pred

    gp_pred, _ = fit_gp(cells, grid, clip_bounds=(0.0, 1.0))
    if gp_pred is not None:
        curves["GP (Matern)"] = gp_pred

    # Collect analytic derivatives for derivative plots
    deriv_cis: dict[str, pd.DataFrame] = {}
    if gam_deriv is not None:
        deriv_cis["GAM (mgcv)"] = gam_deriv
    if emax_deriv_df is not None:
        deriv_cis["Emax (Hill)"] = emax_deriv_df

    # Overall plots
    plot_cell_means(
        cells, OUTPUT_DIR / f"{prefix}_cell_means.png",
        ylabel=ylabel,
        title=f"Adjusted Cell Means: Ad Duration vs {title_suffix}",
    )
    plot_dose_response(
        cells, curves, OUTPUT_DIR / f"{prefix}.png",
        ylabel=ylabel,
        title=f"Dose-Response: Ad Duration vs {title_suffix}",
    )
    plot_derivatives(
        curves, OUTPUT_DIR / f"{prefix}_derivative.png",
        order=1, analytic_derivatives=deriv_cis,
        ylabel="dPr/d(second)",
        title=f"Marginal Effect of Additional Exposure ({title_suffix})",
    )
    if emax_deriv_df is not None:
        plot_emax_derivative(
            emax_deriv_df, OUTPUT_DIR / f"{prefix}_derivative_emax.png",
            deriv_ylabel="dPr/d(second)",
        )
    plot_derivatives(
        curves, OUTPUT_DIR / f"{prefix}_second_derivative.png",
        order=2, analytic_derivatives=deriv_cis,
        title="Rate of Change in Marginal Effect (Second Derivative)",
    )

    # Path-conditional analysis
    path_cells = adjusted_cell_means(sample, dv_col=dv_col, by_path=True)

    all_path_curves: dict[str, dict[str, pd.DataFrame]] = {}
    parametric_path = path_conditional_predictions(sample, grid, dv_col=dv_col, use_glm=True)
    all_path_curves.update(parametric_path)

    emax_path_preds: dict[str, pd.DataFrame] = {}
    for path, pc in path_cells.items():
        emax_p, _, _ = fit_emax(pc, grid)
        if emax_p is not None:
            emax_path_preds[path] = emax_p
    if emax_path_preds:
        all_path_curves["Emax (Hill)"] = emax_path_preds

    plot_dose_response_by_path(
        path_cells, all_path_curves, sample,
        OUTPUT_DIR / f"{prefix}_by_path.png",
        ylabel=ylabel,
        title=f"Path-Conditional Dose-Response ({title_suffix})",
    )

    # Equivalence tests
    balance = run_balance_tests(
        sample, arm_col="treated", arms=(0, 1),
        categorical_covariates=CATEGORICAL_COVARIATES,
        continuous_covariates=["age", "attention_score"],
    )
    n_equiv = sum(1 for _, _, _, _, r in balance["rows"] if r == "EQUIV")
    print(f"Balance: {n_equiv}/{len(balance['rows'])} covariates equivalent")

    equiv = run_dose_response_equivalence(
        sample, dv_col=dv_col, reference_duration=15.0, margin=0.04,
        categorical_covariates=CATEGORICAL_COVARIATES, weights="quality_weight",
    )
    n_equiv = sum(1 for r in equiv if r["result"] == "EQUIV")
    print(f"Dose-response: {n_equiv}/{len(equiv)} durations equivalent to 15s")


if __name__ == "__main__":
    main()
