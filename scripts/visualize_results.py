"""Dose-response visualization functions."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter
from statsmodels.nonparametric.smoothers_lowess import lowess


OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

CURVE_COLORS: dict[str, str] = {
    "Logit (log)": "steelblue",
    "GAM (mgcv)": "purple",
    "Emax (Hill)": "red",
    "GP (Matern)": "darkorange",
    "Local Linear": "teal",
}

MIN_OBSERVED_DOSE = 6


def dur_col(df: pd.DataFrame) -> str:
    """Return the duration column name present in df."""
    return "total_duration" if "total_duration" in df.columns else "duration"


def save_plot(fig: plt.Figure, path: Path) -> None:
    """Save figure and close."""
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fit_loess(
    cells: pd.DataFrame,
    grid: np.ndarray,
    frac: float = 0.4,
) -> pd.DataFrame:
    """Fit a LOWESS smoother through adjusted cell means, weighted by cell size."""
    weights = np.sqrt(cells["n"].values)
    # statsmodels lowess doesn't accept weights directly; weight via
    # repeated sampling proportional to sqrt(n)
    dur = cells["duration"].values
    mean = cells["mean"].values

    # Weighted LOWESS: replicate points proportional to weight
    reps = np.round(weights / weights.min()).astype(int)
    dur_rep = np.repeat(dur, reps)
    mean_rep = np.repeat(mean, reps)

    smoothed = lowess(mean_rep, dur_rep, frac=frac, return_sorted=True)
    # Interpolate onto the grid
    loess_on_grid = np.interp(grid, smoothed[:, 0], smoothed[:, 1])
    return pd.DataFrame({"total_duration": grid, "mean": loess_on_grid})


def plot_cell_means(
    cells: pd.DataFrame,
    output_path: Path,
    ylabel: str = "Pr(Vote for Tom Martin)",
    title: str = "Adjusted Cell Means: Ad Duration vs Vote Choice",
) -> None:
    """Plot adjusted cell means with error bars and sample-size annotations."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(
        cells["duration"],
        cells["mean"],
        yerr=[
            cells["mean"] - cells["ci_low"],
            cells["ci_high"] - cells["mean"],
        ],
        fmt="o",
        color="black",
        markersize=6,
        capsize=4,
        zorder=10,
    )

    # Annotate with sample sizes
    for _, row in cells.iterrows():
        ax.annotate(
            f"n={row['n']:.0f}",
            (row["duration"], row["ci_high"]),
            textcoords="offset points",
            xytext=(0, 6),
            fontsize=6,
            ha="center",
            color="grey",
        )

    ax.set_xlabel("Total Ad Duration (seconds)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    save_plot(fig, output_path)


def plot_dose_response(
    cells: pd.DataFrame,
    curves: dict[str, pd.DataFrame],
    output_path: Path,
    ylabel: str = "Pr(Vote for Tom Martin)",
    title: str = "Dose-Response: Ad Duration vs Vote Choice",
) -> None:
    """Plot model curves with LOESS smoother prominent and parametric models faded."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = CURVE_COLORS
    grid_max = 0.0

    # Parametric/semiparametric model curves (faded)
    for name, pred_df in curves.items():
        color = colors.get(name, "grey")
        col_name = dur_col(pred_df)
        grid_max = max(grid_max, pred_df[col_name].max())
        ax.plot(
            pred_df[col_name],
            pred_df["mean"],
            label=name,
            color=color,
            linewidth=1.2,
            alpha=0.4,
        )
        if "ci_low" in pred_df.columns:
            ax.fill_between(
                pred_df[col_name],
                pred_df["ci_low"],
                pred_df["ci_high"],
                alpha=0.05,
                color=color,
            )

    # LOESS smoother through cell means (prominent)
    grid = np.linspace(0, grid_max, 200) if grid_max > 0 else np.linspace(0, 60, 200)
    loess_df = fit_loess(cells, grid)
    ax.plot(
        loess_df["total_duration"],
        loess_df["mean"],
        color="black",
        linewidth=2.5,
        label="LOESS (cell means)",
        zorder=8,
    )

    # Light cell-mean markers for reference (no error bars)
    ax.scatter(
        cells["duration"],
        cells["mean"],
        color="black",
        s=20,
        zorder=9,
        alpha=0.5,
    )

    ax.set_xlabel("Total Ad Duration (seconds)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    save_plot(fig, output_path)


def plot_derivatives(
    curves: dict[str, pd.DataFrame],
    output_path: Path,
    order: int = 1,
    analytic_derivatives: dict[str, pd.DataFrame] | None = None,
    ylabel: str = "dPr/d(second)",
    title: str = "",
) -> None:
    """Plot numerical derivatives of fitted dose-response curves.

    order=1 plots first derivative (marginal effect).
    order=2 plots second derivative (rate of change in marginal effect).
    """
    if analytic_derivatives is None:
        analytic_derivatives = {}

    x_min = MIN_OBSERVED_DOSE
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")

    for name, pred_df in curves.items():
        if name == "Emax (Hill)":
            continue
        color = CURVE_COLORS.get(name, "grey")

        if name in analytic_derivatives:
            ad = analytic_derivatives[name]
            d_vals = ad[dur_col(ad)].values
            mask = d_vals >= x_min
            d_masked = d_vals[mask]
            first_deriv = ad["deriv_mean"].values[mask]
            if order == 2:
                first_deriv = savgol_filter(first_deriv, window_length=21, polyorder=3)
                deriv = np.gradient(first_deriv, d_masked)
            else:
                deriv = first_deriv
        else:
            duration = pred_df[dur_col(pred_df)].values
            mask = duration >= x_min
            d_masked = duration[mask]
            first_deriv = np.gradient(pred_df["mean"].values[mask], d_masked)
            if order == 2:
                deriv = np.gradient(first_deriv, d_masked)
            else:
                deriv = first_deriv

        ax.plot(d_masked, deriv, label=name, color=color, linewidth=1.5)

    ax.set_xlim(x_min, None)
    ax.axvline(x=x_min, color="grey", linewidth=0.5, linestyle=":", alpha=0.7)
    ax.annotate("Lowest observed dose", xy=(x_min + 0.5, ax.get_ylim()[1] * 0.9),
                fontsize=7, color="grey", va="top")

    if order == 2:
        ylabel = "d\u00b2Pr/d(second)\u00b2"
    ax.set_xlabel("Total Ad Duration (seconds)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    save_plot(fig, output_path)


def plot_emax_derivative(
    emax_deriv_df: pd.DataFrame,
    output_path: Path,
    deriv_ylabel: str = "dPr/d(second)",
) -> None:
    """Plot Emax (Hill) first and second derivative on its own axes."""
    x_min = MIN_OBSERVED_DOSE
    color = CURVE_COLORS.get("Emax (Hill)", "red")

    d_vals = emax_deriv_df["total_duration"].values
    mask = d_vals >= x_min
    d_masked = d_vals[mask]
    deriv = emax_deriv_df["deriv_mean"].values[mask]
    second_deriv = np.gradient(deriv, d_masked)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # First derivative
    ax1.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    ax1.plot(d_masked, deriv, color=color, linewidth=1.5)
    ax1.set_ylabel(deriv_ylabel)
    ax1.set_title("Emax (Hill) Derivatives")
    ax1.grid(True, alpha=0.3)
    ax1.annotate("First derivative", xy=(0.98, 0.95), xycoords="axes fraction",
                 ha="right", va="top", fontsize=9, color="grey")

    # Second derivative
    ax2.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    ax2.plot(d_masked, second_deriv, color=color, linewidth=1.5)
    ax2.set_xlabel("Total Ad Duration (seconds)")
    ax2.set_ylabel("d\u00b2Pr/d(second)\u00b2")
    ax2.grid(True, alpha=0.3)
    ax2.annotate("Second derivative", xy=(0.98, 0.95), xycoords="axes fraction",
                 ha="right", va="top", fontsize=9, color="grey")

    ax2.set_xlim(x_min, None)

    save_plot(fig, output_path)


def plot_dose_by_phase1(
    sample: pd.DataFrame,
    output_path: Path,
) -> None:
    """Show dose-response decomposed by phase1 duration and phase2 path."""
    # Short labels for phase2 paths
    path_labels: dict[str, str] = {
        "path_a_novel": "novel",
        "path_b1_identical": "ident",
        "path_b2_alt_15s": "alt15",
        "path_b3_alt_30s": "alt30",
    }
    # Marker shapes by phase2 path type
    path_markers: dict[str, str] = {
        "path_a_novel": "o",
        "path_b1_identical": "o",
        "path_b2_alt_15s": "o",
        "path_b3_alt_30s": "^",
    }

    # Fill NaN phase2 path for solo/control so groupby works
    sample = sample.copy()
    sample["var_phase2_path"] = sample["var_phase2_path"].fillna("_solo_or_control")

    grouped = (
        sample.groupby(["total_duration", "phase1_dur", "var_phase2_path"])
        .agg(mean=("dv_binary", "mean"), n=("dv_binary", "size"))
        .reset_index()
    )
    # Tag solo cells (phase1 only, no phase2)
    grouped["is_solo"] = (
        (grouped["var_phase2_path"] == "_solo_or_control") & (grouped["phase1_dur"] > 0)
    )
    grouped["is_control"] = grouped["phase1_dur"] == 0

    fig, ax = plt.subplots(figsize=(10, 6))

    phase1_colors = {0.0: "#888888", 6.0: "#e41a1c", 10.0: "#ff7f00", 15.0: "#4daf4a",
                     20.0: "#377eb8", 25.0: "#984ea3", 30.0: "#a65628"}

    placed: list[tuple[float, float]] = []

    def _offset(x: float, y: float) -> tuple[int, int]:
        dy = 8
        for px, py in placed:
            if abs(px - x) < 1.5 and abs(py - y) < 0.015:
                dy += 10
        placed.append((x, y))
        return (0, dy)

    grouped = grouped.sort_values(["total_duration", "mean"], ascending=[True, False])

    for _, row in grouped.iterrows():
        p1 = row["phase1_dur"]
        color = phase1_colors.get(p1, "grey")

        if row["is_control"]:
            marker, label = "o", "0s"
        elif row["is_solo"]:
            marker, label = "D", f"{p1:.0f}s solo"
        else:
            marker = path_markers.get(row["var_phase2_path"], "o")
            label = f"{p1:.0f}+{path_labels.get(row['var_phase2_path'], row['var_phase2_path'])}"

        ax.scatter(
            row["total_duration"], row["mean"],
            s=max(row["n"] * 0.4, 30),
            color=color, marker=marker, zorder=5,
            edgecolors="black", linewidths=0.5,
        )
        dx, dy = _offset(row["total_duration"], row["mean"])
        ax.annotate(
            label, (row["total_duration"], row["mean"]),
            textcoords="offset points", xytext=(dx, dy),
            fontsize=6, ha="center", color=color,
        )

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
               markeredgecolor="black", markersize=8,
               label="Control (0s)" if p1 == 0 else f"{p1:.0f}s phase1")
        for p1, c in sorted(phase1_colors.items())
    ] + [
        Line2D([0], [0], marker=m, color="w", markerfacecolor="grey",
               markeredgecolor="black", markersize=8, label=lab)
        for m, lab in [("D", "Solo (no phase2)"), ("o", "15s path"), ("^", "30s path")]
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=7)

    ax.set_xlabel("Total Ad Duration (seconds)")
    ax.set_ylabel("Pr(Vote for Tom Martin) - Raw Mean")
    ax.set_title("Dose-Response by Phase 1 Duration (Compositional Structure)")
    ax.grid(True, alpha=0.3)

    save_plot(fig, output_path)


PATH_COLORS: dict[str, str] = {
    "path_a_novel": "#D55E00",       # Vermillion (Okabe-Ito)
    "path_b1_identical": "#0072B2",  # Blue (Okabe-Ito)
    "path_b2_alt_15s": "#009E73",    # Bluish green (Okabe-Ito)
    "path_b3_alt_30s": "#E69F00",    # Orange (Okabe-Ito)
    "none": "#888888",
}

PATH_MARKERS: dict[str, str] = {
    "path_a_novel": "o",
    "path_b1_identical": "s",
    "path_b2_alt_15s": "^",
    "path_b3_alt_30s": "D",
    "none": "X",
}

PATH_LABELS: dict[str, str] = {
    "path_a_novel": "A: novel",
    "path_b1_identical": "B1: identical",
    "path_b2_alt_15s": "B2: alt 15s",
    "path_b3_alt_30s": "B3: alt 30s",
    "none": "Solo/control",
}

MODEL_LINESTYLES: dict[str, str] = {
    "Logit (log)": "--",
    "GAM (mgcv)": ":",
    "Emax (Hill)": "-.",
}


def plot_dose_response_by_path(
    path_cells: dict[str, pd.DataFrame],
    path_curves: dict[str, dict[str, pd.DataFrame]],
    sample: pd.DataFrame,
    output_path: Path,
    ylabel: str = "Pr(Vote for Tom Martin)",
    title: str = "Path-Conditional Dose-Response Curves",
) -> None:
    """Plot path-conditional dose-response curves with per-path cell means."""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Observed duration ranges per path for clipping
    path_ranges: dict[str, tuple[float, float]] = {}
    for path in sample["var_phase2_path"].unique():
        durs = sample.loc[sample["var_phase2_path"] == path, "total_duration"]
        path_ranges[path] = (durs.min(), durs.max())

    # Cell means as colored dots with error bars (marker shape encodes path)
    for path, cells in path_cells.items():
        color = PATH_COLORS.get(path, "grey")
        marker = PATH_MARKERS.get(path, "o")
        ax.errorbar(
            cells["duration"],
            cells["mean"],
            yerr=[
                cells["mean"] - cells["ci_low"],
                cells["ci_high"] - cells["mean"],
            ],
            fmt=marker,
            color=color,
            markersize=5,
            capsize=3,
            zorder=10,
        )

    # Fitted curves: color + marker by path, linestyle by model

    # Track rightmost curve endpoint per path for label placement
    curve_endpoints: dict[str, tuple[float, float]] = {}

    legend_models_added: set[str] = set()
    for model_name, per_path in path_curves.items():
        ls = MODEL_LINESTYLES.get(model_name, "-")
        for path, pred_df in per_path.items():
            color = PATH_COLORS.get(path, "grey")
            marker = PATH_MARKERS.get(path, "o")
            ms = 5 if path == "none" else 4
            d_min, d_max = path_ranges.get(path, (0, 60))
            mask = (pred_df["total_duration"] >= d_min) & (pred_df["total_duration"] <= d_max)
            clipped = pred_df[mask]
            ax.plot(
                clipped["total_duration"],
                clipped["mean"],
                color=color,
                linestyle=ls,
                linewidth=1.2,
                alpha=0.8,
                marker=marker,
                markevery=20,
                markersize=ms,
            )
            if model_name not in legend_models_added:
                legend_models_added.add(model_name)
            # Update rightmost curve point for this path
            if len(clipped) > 0:
                last_x = clipped["total_duration"].iloc[-1]
                last_y = clipped["mean"].iloc[-1]
                if path not in curve_endpoints or last_x > curve_endpoints[path][0]:
                    curve_endpoints[path] = (last_x, last_y)

    # Direct labels at rightmost curve endpoint per path, with vertical stagger
    label_items: list[tuple[float, float, str, str]] = []
    for path in path_cells:
        color = PATH_COLORS.get(path, "grey")
        label = PATH_LABELS.get(path, path)
        if path in curve_endpoints:
            x, y = curve_endpoints[path]
        else:
            cells = path_cells[path]
            x, y = cells["duration"].iloc[-1], cells["mean"].iloc[-1]
        label_items.append((x, y, label, color))

    # Sort by y so we can stagger overlapping labels
    label_items.sort(key=lambda t: t[1])
    min_gap_points = 10  # minimum vertical separation in offset points
    placed_offsets: list[float] = []
    for x, y, label, color in label_items:
        dy = 0.0
        for prev_dy in placed_offsets:
            if abs(dy - prev_dy) < min_gap_points:
                dy = prev_dy + min_gap_points
        placed_offsets.append(dy)
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(6, dy),
            textcoords="offset points",
            fontsize=7,
            color=color,
            va="center",
            fontweight="bold",
        )

    # Legend: model linestyles only (paths are directly labeled)
    model_handles = []
    for model_name in legend_models_added:
        ls = MODEL_LINESTYLES.get(model_name, "-")
        model_handles.append(
            Line2D([0], [0], color="black", linestyle=ls, linewidth=1.2,
                   label=model_name)
        )

    # Extend x-axis for label room
    x_max = max(x for x, _, _, _ in label_items)
    ax.set_xlim(None, x_max + 8)

    ax.set_xlabel("Total Ad Duration (seconds)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(handles=model_handles, loc="best", fontsize=7)
    ax.grid(True, alpha=0.3)

    save_plot(fig, output_path)
