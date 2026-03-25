# Mind the Gap

How does the length of a political ad affect whether it changes someone's vote? In a survey experiment with 20,148 respondents, we find that nearly all persuasion happens within the first 15 seconds of exposure. Additional seconds of advertising produce sharply diminishing returns.

## Quick start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). R with `mgcv` is optional (GAM curves are skipped without it).

```bash
uv sync
uv run python scripts/1_prepare_data.py
uv run python scripts/2_run_analysis.py
```

All plots are written to `outputs/`.

## Pipeline

The analysis runs in two steps. Step 1 produces `data/analysis_ready.parquet` from the raw survey exports. Step 2 reads that file, fits four dose-response models, tests for equivalence across duration levels, and writes the plots.

`1_prepare_data.py` harmonizes treatment assignment, scores attention checks, computes quality weights, and outputs one row per respondent with duration (seconds of ad exposure), binary outcome (vote choice), and covariates.

`2_run_analysis.py` estimates the dose-response curve four ways (Logit-log GLM, GAM via R/mgcv, Emax/Hill, Gaussian Process), runs TOST equivalence tests comparing each duration to the 15-second reference, and generates six plots.

## Data

Five parquet files in `data/`, one per survey version (CINT panel, February-March 2026), totaling 20,148 respondents after exclusions.
