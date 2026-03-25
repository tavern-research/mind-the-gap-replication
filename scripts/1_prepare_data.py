#!/usr/bin/env python
"""Prepare analysis-ready dataset from exported parquet files.

Reads 5 parquet files (one per survey version), harmonizes treatment
variables, scores attention, prepares covariates, joins all sources,
computes quality weights, and outputs a single analysis-ready dataframe.

"""

from pathlib import Path

import polars as pl

DATA_DIR = Path(__file__).parent.parent / "data"

# Attention check definitions: (column, fail_value, negate?)
# fail if equals value (negate=False), fail if NOT equals value (negate=True)
ATTENTION_CHECKS: list[tuple[str, str, bool]] = [
    ("diet_control_fake_ballot_measure", "yes", False),
    ("potholes_fake_policy", "support_policy", False),
    ("patterson_favorability", "Not sure/never heard of", True),
    ("what_do__wrestled_bear", "true", False),
    ("what_do__ate_crickets", "true", False),
    ("what_do__slept_2_hours", "true", True),
    ("what_do__walked_10_steps", "true", True),
]

CATEGORICAL_COVARIATES = ["potus_2024", "gender", "race", "income"]

BATCH_SOURCES = [
    ("2026-02-25.parquet", "exp1_pilot"),
    ("2026-02-27.parquet", "exp1_prev"),
    ("2026-03-02.parquet", "exp1_current_pre"),
    ("2026-03-03.parquet", "exp3"),
]
EXP1_MAIN_FILE = "2026-03-05.parquet"


def score_attention(df: pl.DataFrame, n_checks: int) -> pl.DataFrame:
    """Add attention_score column (proportion of checks failed, 0=best 1=worst).

    Scores items defined in ATTENTION_CHECKS. Items not present in the
    DataFrame are skipped.
    """
    cols = df.columns
    fail_exprs: list[pl.Expr] = []

    for col, value, negate in ATTENTION_CHECKS:
        if col not in cols:
            continue
        check = pl.col(col).is_not_null() & (pl.col(col) != value if negate else pl.col(col) == value)
        fail_exprs.append(check.cast(pl.Int32))

    return df.with_columns(
        (pl.sum_horizontal(fail_exprs) / n_checks).alias("attention_score")
    )


def prepare_standard_covariates(df: pl.DataFrame) -> pl.DataFrame:
    """Prepare standard pretreatment covariates.

    - Age from birth_year with age_missing indicator
    - Categorical NaN filled with "missing"
    - Attention score NaN filled with 0.0
    """
    birth_year = pl.col("birth_year").cast(pl.Float64, strict=False)
    age = pl.lit(2026) - birth_year
    age_median = df.select(age.median()).item()

    exprs: list[pl.Expr] = [
        birth_year.is_null().cast(pl.Int32).alias("age_missing"),
        (age - age_median).fill_null(0.0).alias("age"),
    ]

    for col in CATEGORICAL_COVARIATES:
        if col in df.columns:
            exprs.append(
                pl.col(col).fill_null("missing").cast(pl.Utf8).alias(col)
            )
        else:
            exprs.append(pl.lit("missing").alias(col))

    if "attention_score" in df.columns:
        exprs.append(
            pl.col("attention_score").fill_null(0.0).alias("attention_score")
        )
    else:
        exprs.append(pl.lit(0.0).alias("attention_score"))

    return df.with_columns(exprs)


def build_dataset() -> pl.DataFrame:
    """Read all parquet sources, harmonize, and produce analysis-ready dataset."""
    frames: list[pl.DataFrame] = []
    for filename, label in BATCH_SOURCES:
        lf = pl.scan_parquet(DATA_DIR / filename)
        schema = lf.collect_schema()
        lf = lf.with_columns(pl.lit(label).alias("source"))

        if label == "exp3":
            duration = (
                pl.when(pl.col("var_arm") == "short").then(6.0)
                .when(pl.col("var_arm") == "full").then(30.0)
                .otherwise(0.0)
            )
            is_treatment = pl.lit(1).cast(pl.Int32)
            ad_format = pl.lit(None)
            condition_label = pl.col("var_arm")
        else:
            phase1_dur = (
                pl.col("var_phase1_duration")
                .cast(pl.Float64, strict=False)
                .fill_null(0.0)
            )
            phase1_dur = (
                pl.when(pl.col("var_phase1_condition") == "placebo")
                .then(15.0)
                .otherwise(phase1_dur)
            )
            if label == "exp1_current_pre":
                duration = phase1_dur
            elif "var_phase2_path" in schema:
                phase2_dur = (
                    pl.when(pl.col("var_phase2_path") == "path_b3_alt_30s")
                    .then(30.0)
                    .when(pl.col("var_phase2_path").is_not_null())
                    .then(15.0)
                    .otherwise(0.0)
                )
                duration = phase1_dur + phase2_dur
            else:
                duration = phase1_dur

            is_treatment = (
                (pl.col("var_is_treatment").fill_null("no") == "yes")
                .cast(pl.Int32)
            )
            ad_format = pl.col("var_ad_format") if "var_ad_format" in schema else pl.lit(None)
            condition_label = pl.col("var_phase1_condition")

        lf = lf.with_columns(
            duration.alias("duration"),
            is_treatment.alias("is_treatment"),
            ad_format.alias("ad_format"),
            condition_label.alias("condition_label"),
        )

        frames.append(lf.collect())

    batch = pl.concat(frames, how="diagonal")
    batch = score_attention(batch, n_checks=7)
    batch = prepare_standard_covariates(batch)
    batch = batch.filter(pl.col("condition_label").is_not_null())

    has_dv2 = (
        pl.col("outcome_dv").is_null()
        & pl.col("outcome_dv_2").is_not_null()
    )
    raw_dv = pl.when(has_dv2).then(pl.col("outcome_dv_2")).otherwise(pl.col("outcome_dv"))
    raw_push = pl.when(has_dv2).then(pl.col("outcome_dv_2_push")).otherwise(pl.col("outcome_dv_push"))
    dv_resolved = (
        pl.when(raw_dv == "not_sure").then(raw_push).otherwise(raw_dv)
    )
    batch = batch.with_columns(
        dv_resolved.alias("dv_resolved"),
        (dv_resolved == "democrat_tom_martin").cast(pl.Int32).alias("dv_binary"),
    )

    exp1 = pl.scan_parquet(DATA_DIR / EXP1_MAIN_FILE).collect()
    exp1 = score_attention(exp1, n_checks=6)

    # Resolve DVs based on has_phase2 gating
    has_phase2 = pl.col("var_has_phase2").fill_null("no") == "yes"

    def _push_resolve(dv: str, push: str) -> pl.Expr:
        base = pl.col(dv)
        return pl.when(base == "not_sure").then(pl.col(push)).otherwise(base)

    non_p2_dv = _push_resolve("outcome_dv", "outcome_dv_push")
    p2_dv = _push_resolve("outcome_dv_2", "outcome_dv_2_push")
    dv_resolved = pl.when(has_phase2).then(p2_dv).otherwise(non_p2_dv)

    phase1_dur = (
        pl.col("var_phase1_duration")
        .cast(pl.Float64, strict=False)
        .fill_null(0.0)
    )
    phase1_dur = (
        pl.when(pl.col("var_phase1_condition") == "placebo")
        .then(15.0)
        .otherwise(phase1_dur)
    )
    phase2_dur = (
        pl.when(pl.col("var_phase2_path") == "path_b3_alt_30s")
        .then(30.0)
        .when(pl.col("var_phase2_path").is_not_null())
        .then(15.0)
        .otherwise(0.0)
    )
    exp1 = exp1.with_columns(
        (phase1_dur + phase2_dur).alias("duration"),
        (dv_resolved == "democrat_tom_martin").cast(pl.Int32).alias("dv_binary"),
    )

    exp1 = prepare_standard_covariates(exp1)
    exp1 = exp1.with_columns(pl.lit("exp1_main").alias("source"))

    join_cols = [
        "source", "duration", "dv_binary",
        "gender", "race", "income", "education",
        "potus_2024", "registered_voter",
        "attention_score", "age", "age_missing",
    ]

    exp1_subset = exp1.select([c for c in join_cols if c in exp1.columns])
    batch_subset = batch.select([c for c in join_cols if c in batch.columns])
    joined = pl.concat([exp1_subset, batch_subset], how="diagonal")

    joined = joined.with_columns(
        ((1 - pl.col("attention_score")) ** 2).clip(lower_bound=1e-10).alias("quality_weight")
    )

    joined = joined.filter(
        pl.col("dv_binary").is_not_null()
        & pl.col("education").is_not_null() & (pl.col("education") != "missing")
        & pl.col("race").is_not_null() & (pl.col("race") != "missing")
        & pl.col("income").is_not_null() & (pl.col("income") != "missing")
        & pl.col("registered_voter").is_not_null() & (pl.col("registered_voter") != "missing")
    )

    # Bin near-identical durations
    joined = joined.with_columns(
        pl.when(pl.col("duration") == 20.0).then(21.0)
        .when(pl.col("duration") == 36.0).then(35.0)
        .otherwise(pl.col("duration"))
        .alias("duration"),
    )

    # Aliases for pipeline compatibility
    joined = joined.with_columns(
        pl.col("duration").alias("total_duration"),
        (pl.col("duration") > 0).cast(pl.Int32).alias("treated"),
        (pl.col("duration") == 0).cast(pl.Int32).alias("is_control"),
    )

    return joined


def main() -> None:
    joined = build_dataset()
    output_path = DATA_DIR / "analysis_ready.parquet"
    joined.write_parquet(output_path)
    print(f"Wrote {output_path} ({joined.height} rows)")


if __name__ == "__main__":
    main()
