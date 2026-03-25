"""TOST equivalence tests for balance and dose-response flatness."""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.weightstats import ttost_ind

from estimate_dose_response import build_covariate_formula


def run_balance_tests(
    pdf: pd.DataFrame,
    arm_col: str,
    arms: tuple[str, str],
    categorical_covariates: list[str],
    continuous_covariates: list[str],
    margin_sd: float = 0.1,
) -> dict[str, object]:
    """TOST equivalence balance tests (Hartman & Hidalgo 2018).

    Returns dict with keys:
    - rows: list of (variable, diff, smd, p_equiv, result) tuples
    - margin_sd: the margin used
    - f_value: joint F-test statistic (or None)
    - f_pvalue: joint F-test p-value (or None)
    """
    g0 = pdf[pdf[arm_col] == arms[0]]
    g1 = pdf[pdf[arm_col] == arms[1]]

    rows: list[tuple[str, str, str, float, str]] = []

    for col in continuous_covariates:
        if col not in pdf.columns:
            continue
        x0 = g0[col].dropna().values
        x1 = g1[col].dropna().values
        pooled_sd = np.sqrt(
            ((len(x0) - 1) * np.var(x0, ddof=1)
             + (len(x1) - 1) * np.var(x1, ddof=1))
            / (len(x0) + len(x1) - 2)
        )
        if pooled_sd == 0:
            rows.append((col, "0.000", "0.000", 0.0, "EQUIV"))
            continue
        bound = margin_sd * pooled_sd
        raw_diff = float(np.mean(x1) - np.mean(x0))
        smd = raw_diff / pooled_sd
        p_equiv, _, _ = ttost_ind(x0, x1, low=-bound, upp=bound, usevar="unequal")
        result = "EQUIV" if p_equiv < 0.05 else "NOT EQUIV"
        rows.append((col, f"{raw_diff:.3f}", f"{smd:.3f}", p_equiv, result))

    for col in categorical_covariates:
        if col not in pdf.columns:
            continue
        dummies = pd.get_dummies(pdf[col], prefix=col, dtype=float)
        max_p = 0.0
        max_abs_smd = 0.0
        for dcol in dummies.columns:
            x0 = dummies.loc[g0.index, dcol].values
            x1 = dummies.loc[g1.index, dcol].values
            pooled_sd = np.sqrt(
                ((len(x0) - 1) * np.var(x0, ddof=1)
                 + (len(x1) - 1) * np.var(x1, ddof=1))
                / (len(x0) + len(x1) - 2)
            )
            if pooled_sd == 0:
                continue
            bound = margin_sd * pooled_sd
            smd_d = (np.mean(x1) - np.mean(x0)) / pooled_sd
            p_val, _, _ = ttost_ind(x0, x1, low=-bound, upp=bound, usevar="unequal")
            max_p = max(max_p, p_val)
            max_abs_smd = max(max_abs_smd, abs(smd_d))
        result = "EQUIV" if max_p < 0.05 else "NOT EQUIV"
        rows.append((col, "\u2014", f"{max_abs_smd:.3f}", max_p, result))

    # Joint F-test -- drop rows with categorical levels that have < 3 obs
    # per arm to avoid rank-deficient design matrices from sparse cells.
    min_cell = 3
    all_covariates = [c for c in continuous_covariates if c in pdf.columns]
    ftest_mask = pd.Series(True, index=pdf.index)
    for col in categorical_covariates:
        if col not in pdf.columns or pdf[col].nunique() <= 1:
            continue
        all_covariates.append(f"C({col})")
        counts = pdf.groupby([arm_col, col]).size().unstack(fill_value=0)
        sparse_levels = counts.columns[counts.min(axis=0) < min_cell].tolist()
        if sparse_levels:
            ftest_mask &= ~pdf[col].isin(sparse_levels)

    f_value = None
    f_pvalue = None
    if all_covariates:
        tmp = pdf.loc[ftest_mask].copy()
        tmp["_arm_binary"] = (tmp[arm_col] == arms[1]).astype(int)
        formula = "_arm_binary ~ " + " + ".join(all_covariates)
        joint_model = smf.ols(formula, data=tmp).fit()
        f_value = joint_model.fvalue
        f_pvalue = joint_model.f_pvalue

    return {
        "rows": rows,
        "margin_sd": margin_sd,
        "f_value": f_value,
        "f_pvalue": f_pvalue,
    }


def run_dose_response_equivalence(
    sample: pd.DataFrame,
    dv_col: str = "dv_binary",
    reference_duration: float = 15.0,
    margin: float = 0.02,
    categorical_covariates: list[str] | None = None,
    weights: str | None = "quality_weight",
) -> list[dict]:
    """TOST equivalence tests for dose-response flatness.

    Tests whether the adjusted mean at each duration differs from the
    reference duration by more than +/- margin.

    Returns a list of dicts with keys: duration, diff, se, ci_low, ci_high, tost_p, result.
    """
    covariates = build_covariate_formula(
        sample, categorical_covariates=categorical_covariates
    )
    formula = f"{dv_col} ~ C(total_duration) + {covariates}"

    if weights and weights in sample.columns:
        model = smf.wls(formula, data=sample, weights=sample[weights]).fit(cov_type="HC2")
    else:
        model = smf.ols(formula, data=sample).fit(cov_type="HC2")

    param_names = model.params.index.tolist()
    durations = sorted(d for d in sample["total_duration"].unique() if d > reference_duration)
    df_resid = model.df_resid

    ref_label = None
    for name in param_names:
        if name.startswith("C(total_duration)[T.") and str(reference_duration) in name:
            ref_label = name
            break

    results: list[dict] = []
    for d in durations:
        dur_label = None
        for name in param_names:
            if name.startswith("C(total_duration)[T.") and str(d) in name:
                dur_label = name
                break
        if dur_label is None:
            continue

        if ref_label is None:
            contrast = dur_label
        else:
            contrast = f"{dur_label} - {ref_label}"

        t_result = model.t_test(contrast)
        estimate = float(np.squeeze(t_result.effect))
        se = float(np.squeeze(t_result.sd))

        t_crit = stats.t.ppf(0.95, df_resid)
        ci_low = estimate - t_crit * se
        ci_high = estimate + t_crit * se

        p_upper = stats.t.cdf((estimate - margin) / se, df_resid)
        p_lower = 1.0 - stats.t.cdf((estimate + margin) / se, df_resid)
        tost_p = max(p_upper, p_lower)

        result = "EQUIV" if tost_p < 0.05 else "NOT EQUIV"
        results.append({
            "duration": d,
            "diff": estimate,
            "se": se,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "tost_p": tost_p,
            "result": result,
        })

    return results
