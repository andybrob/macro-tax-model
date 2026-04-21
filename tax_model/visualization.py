"""
Visualization module.

All functions return matplotlib Figure objects — they never call plt.show()
or plt.savefig(). Let the caller (notebook or Streamlit app) decide.

Standard chart set for a PolicyComparison:
  1. Macro bars: GDP, capital, labor, wage % change
  2. Revenue waterfall: baseline vs reform by instrument
  3. Distributional incidence: burden change by income group
  4. Sensitivity tornado (for sweep results)
  5. Monte Carlo credible interval strip chart
  6. Transition path: GDP, capital, wage over time (from transition.py)
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; works in both Streamlit and Jupyter
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

from .calibration import GROUPS
from .results import PolicyComparison, DistributionalIncidence


# ---------------------------------------------------------------------------
# Color palette (accessible, works in print)
# ---------------------------------------------------------------------------

_BLUE   = "#2166AC"
_RED    = "#D6604D"
_GREEN  = "#4DAC26"
_GREY   = "#808080"
_ORANGE = "#F4A582"
_DARK   = "#1A1A1A"

_GROUP_LABELS = {
    "Q1": "Q1\n(bottom 20%)",
    "Q2": "Q2",
    "Q3": "Q3\n(middle 20%)",
    "Q4": "Q4",
    "Q5_bottom": "Q5b\n(80–99%)",
    "Q5_top": "Q5t\n(top 1%)",
}


# ---------------------------------------------------------------------------
# 1. Macro bar chart
# ---------------------------------------------------------------------------

def plot_macro_bars(comparison: PolicyComparison) -> plt.Figure:
    """
    Four horizontal bars showing % change in GDP, capital stock, labor supply,
    and real wage from baseline to reform.
    """
    metrics = {
        "GDP": comparison.gdp_change_pct,
        "Capital stock": comparison.capital_change_pct,
        "Labor supply": comparison.labor_change_pct,
        "Real wage": comparison.wage_change_pct,
    }

    fig, ax = plt.subplots(figsize=(7, 4))
    labels = list(metrics.keys())
    values = list(metrics.values())
    colors = [_GREEN if v >= 0 else _RED for v in values]

    bars = ax.barh(labels, values, color=colors, height=0.5)
    ax.axvline(0, color=_DARK, linewidth=0.8)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    ax.set_xlabel("Change relative to baseline (%)")
    ax.set_title(
        f"{comparison.baseline.policy_label}  →  {comparison.reform.policy_label}\n"
        "Macroeconomic effects (long-run steady state)",
        fontsize=10,
    )

    for bar, val in zip(bars, values):
        pad = 0.05
        ha = "left" if val >= 0 else "right"
        ax.text(
            val + (pad if val >= 0 else -pad),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.2f}%",
            va="center",
            ha=ha,
            fontsize=9,
        )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Revenue waterfall
# ---------------------------------------------------------------------------

def plot_revenue(comparison: PolicyComparison) -> plt.Figure:
    """
    Side-by-side bar chart of tax revenue by instrument for baseline and reform.
    Revenue expressed as % of GDP.
    """
    instruments = [
        ("Labor income", "labor_income_tax"),
        ("Payroll", "payroll_tax"),
        ("Consumption", "consumption_tax"),
        ("Land value", "land_value_tax"),
        ("Corporate", "corporate_tax"),
        ("Pigouvian", "pigouvian_tax"),
        ("Capital gains", "capital_gains_tax"),
        ("Estate", "estate_tax"),
    ]

    base_rev = comparison.baseline.revenue
    ref_rev  = comparison.reform.revenue

    labels = [i[0] for i in instruments]
    base_vals = [getattr(base_rev, i[1]) * 100 for i in instruments]
    ref_vals  = [getattr(ref_rev,  i[1]) * 100 for i in instruments]

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, base_vals, w, label=comparison.baseline.policy_label,
           color=_BLUE, alpha=0.85)
    ax.bar(x + w / 2, ref_vals,  w, label=comparison.reform.policy_label,
           color=_ORANGE, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=1))
    ax.set_ylabel("Revenue as % of GDP")
    ax.set_title("Tax revenue by instrument", fontsize=10)
    ax.legend(fontsize=9)

    # Add total revenue annotation
    total_base = base_rev.total * 100
    total_ref  = ref_rev.total * 100
    ax.axhline(total_base, color=_BLUE, linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axhline(total_ref,  color=_ORANGE, linestyle="--", linewidth=0.8, alpha=0.6)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Distributional incidence
# ---------------------------------------------------------------------------

def plot_incidence(comparison: PolicyComparison) -> plt.Figure:
    """
    Bar chart of change in tax burden by income group (pp of pre-tax income).
    Negative bars = reform reduces burden on that group (good for that group).
    Color: green if burden falls, red if it rises.
    """
    ic = comparison.incidence_change
    groups = list(GROUPS)
    values = [ic[g] * 100 for g in groups]
    labels = [_GROUP_LABELS[g] for g in groups]
    colors = [_GREEN if v <= 0 else _RED for v in values]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, width=0.6)
    ax.axhline(0, color=_DARK, linewidth=0.8)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=1))
    ax.set_ylabel("Change in tax burden (pp of pre-tax income)")
    ax.set_title(
        f"Distributional incidence\n"
        f"{comparison.baseline.policy_label}  →  {comparison.reform.policy_label}",
        fontsize=10,
    )

    for bar, val in zip(bars, values):
        va = "bottom" if val >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val,
            f"{val:+.1f}pp",
            ha="center",
            va=va,
            fontsize=8,
        )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Combined dashboard (all three panels)
# ---------------------------------------------------------------------------

def plot_dashboard(comparison: PolicyComparison) -> plt.Figure:
    """
    Three-panel dashboard: macro bars | revenue | distributional incidence.
    """
    fig = plt.figure(figsize=(16, 5))

    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    _draw_macro_bars_on(ax1, comparison)
    _draw_revenue_on(ax2, comparison)
    _draw_incidence_on(ax3, comparison)

    fig.suptitle(
        f"Policy comparison:  {comparison.baseline.policy_label}  →  {comparison.reform.policy_label}",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# ---------------------------------------------------------------------------
# 5. Sensitivity tornado
# ---------------------------------------------------------------------------

def plot_sensitivity_tornado(
    parameter_labels: List[str],
    comparisons_by_param: Dict[str, List[PolicyComparison]],
    metric: str = "gdp_change_pct",
    metric_label: str = "GDP change (%)",
) -> plt.Figure:
    """
    Tornado chart showing sensitivity of a metric to each swept parameter.
    For each parameter, shows the range [min, max] of the metric across its sweep.
    """
    ranges = {}
    for label, comps in comparisons_by_param.items():
        vals = [getattr(c, metric) for c in comps]
        ranges[label] = (min(vals), max(vals))

    # Sort by range width (largest first)
    sorted_params = sorted(ranges.keys(), key=lambda k: ranges[k][1] - ranges[k][0], reverse=True)

    fig, ax = plt.subplots(figsize=(8, max(3, len(sorted_params) * 0.7 + 1)))

    for i, param in enumerate(sorted_params):
        lo, hi = ranges[param]
        ax.barh(i, hi - lo, left=lo, color=_BLUE, alpha=0.7, height=0.5)
        ax.text(lo - 0.02, i, f"{lo:+.2f}", va="center", ha="right", fontsize=8)
        ax.text(hi + 0.02, i, f"{hi:+.2f}", va="center", ha="left",  fontsize=8)

    ax.set_yticks(range(len(sorted_params)))
    ax.set_yticklabels(sorted_params, fontsize=9)
    ax.axvline(0, color=_DARK, linewidth=0.8)
    ax.set_xlabel(metric_label)
    ax.set_title("Sensitivity analysis — parameter uncertainty ranges", fontsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. Monte Carlo credible interval strip
# ---------------------------------------------------------------------------

def plot_monte_carlo(
    comparisons: List[PolicyComparison],
    metrics: Optional[List[str]] = None,
    metric_labels: Optional[Dict[str, str]] = None,
) -> plt.Figure:
    """
    Strip chart showing the distribution of outcomes across Monte Carlo draws.
    Shows median, 50% interval (IQR), and 90% interval.
    """
    if metrics is None:
        metrics = ["gdp_change_pct", "capital_change_pct", "wage_change_pct", "budget_balance_change"]
    if metric_labels is None:
        metric_labels = {
            "gdp_change_pct": "GDP (%)",
            "capital_change_pct": "Capital (%)",
            "wage_change_pct": "Wage (%)",
            "budget_balance_change": "Budget balance (pp GDP)",
        }

    fig, ax = plt.subplots(figsize=(8, max(3, len(metrics) * 0.9 + 1)))

    for i, metric in enumerate(metrics):
        vals = np.array([getattr(c, metric) for c in comparisons])
        p5, p25, p50, p75, p95 = np.percentile(vals, [5, 25, 50, 75, 95])

        ax.plot([p5, p95], [i, i], color=_GREY, linewidth=2, zorder=1)
        ax.fill_betweenx([i - 0.25, i + 0.25], p25, p75,
                         color=_BLUE, alpha=0.5, zorder=2)
        ax.scatter(p50, i, color=_DARK, zorder=3, s=40)

        label = metric_labels.get(metric, metric)
        ax.text(-0.5, i, label, va="center", ha="right", fontsize=9,
                transform=ax.get_yaxis_transform())

    ax.axvline(0, color=_DARK, linewidth=0.8)
    ax.set_yticks([])
    ax.set_xlabel("Value")
    ax.set_title(
        "Monte Carlo uncertainty (90% CI = line, 50% CI = bar, dot = median)",
        fontsize=9,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. Transition path chart
# ---------------------------------------------------------------------------

def plot_transition(transition_result) -> plt.Figure:
    """
    Three-panel chart showing the transition path from baseline to reform SS.

      Panel 1: GDP over time (normalized to baseline = 1.0)
      Panel 2: Capital stock over time
      Panel 3: Tax burden change by income group over time

    The vertical dashed line at t=0 marks the policy change.
    The horizontal dashed lines show the reform steady state.
    """
    tr = transition_result
    t  = tr.period_numbers

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    # --- Panel 1: GDP ---
    ax = axes[0, 0]
    ax.plot(t, tr.gdp_path, color=_BLUE, linewidth=2)
    ax.axhline(tr.reform_ss.gdp,   color=_GREEN, linestyle="--", linewidth=1,
               label=f"Reform SS ({tr.reform_ss.gdp:.3f})")
    ax.axhline(tr.baseline_ss.gdp, color=_GREY,  linestyle=":",  linewidth=1,
               label=f"Baseline SS ({tr.baseline_ss.gdp:.3f})")
    ax.set_xlabel("Period")
    ax.set_ylabel("GDP (baseline = 1.0)")
    ax.set_title("GDP transition path")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    # --- Panel 2: Capital ---
    ax = axes[0, 1]
    ax.plot(t, tr.capital_path, color=_ORANGE, linewidth=2)
    ax.axhline(tr.reform_ss.capital_stock,   color=_GREEN, linestyle="--", linewidth=1)
    ax.axhline(tr.baseline_ss.capital_stock, color=_GREY,  linestyle=":",  linewidth=1)
    ax.set_xlabel("Period")
    ax.set_ylabel("Capital stock (baseline = 1.0)")
    ax.set_title("Capital stock transition")
    ax.spines[["top", "right"]].set_visible(False)

    # --- Panel 3: Incidence over time for key groups ---
    ax = axes[1, 0]
    colors_by_group = {"Q1": _GREEN, "Q3": _BLUE, "Q5_top": _RED}
    for group, color in colors_by_group.items():
        burden_path = tr.incidence_path(group)
        ax.plot(t, [b * 100 for b in burden_path], color=color, linewidth=2, label=group)
    ax.axhline(0, color=_DARK, linewidth=0.8)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=1))
    ax.set_xlabel("Period")
    ax.set_ylabel("Tax burden (% of pre-tax income)")
    ax.set_title("Incidence over transition (Q1, Q3, Q5_top)")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    # --- Panel 4: Cumulative debt path ---
    ax = axes[1, 1]
    budget_path = tr.series("budget_balance")
    # Cumulative debt = initial debt + cumsum of deficits (negative budget balance adds to debt)
    # Start from 0 (relative change vs. baseline budget balance at t=0)
    baseline_balance = tr.baseline_ss.budget_balance
    cumulative_debt  = np.cumsum([-(b - baseline_balance) for b in budget_path])
    ax.plot(t, cumulative_debt * 100, color=_RED, linewidth=2)
    ax.axhline(0, color=_DARK, linewidth=0.8)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=1))
    ax.set_xlabel("Period")
    ax.set_ylabel("Cumulative debt change (pp of GDP)")
    ax.set_title("Fiscal path\n(cumulative debt vs. baseline)")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Transition path: [{tr.baseline_ss.policy_label}] → [{tr.reform_ss.policy_label}]\n"
        f"Long-run GDP change: {tr.long_run_gdp_change():+.2f}%  |  "
        f"Approx. convergence: period {tr.years_to_convergence()}",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# ---------------------------------------------------------------------------
# Axes-level helpers (used by plot_dashboard)
# ---------------------------------------------------------------------------

def _draw_macro_bars_on(ax: plt.Axes, comparison: PolicyComparison) -> None:
    metrics = {
        "GDP": comparison.gdp_change_pct,
        "Capital": comparison.capital_change_pct,
        "Labor": comparison.labor_change_pct,
        "Wage": comparison.wage_change_pct,
    }
    labels = list(metrics.keys())
    values = list(metrics.values())
    colors = [_GREEN if v >= 0 else _RED for v in values]
    ax.barh(labels, values, color=colors, height=0.5)
    ax.axvline(0, color=_DARK, linewidth=0.8)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    ax.set_title("Macro effects", fontsize=9)

def _draw_revenue_on(ax: plt.Axes, comparison: PolicyComparison) -> None:
    instruments = [
        ("Labor", "labor_income_tax"),
        ("Payroll", "payroll_tax"),
        ("Consump.", "consumption_tax"),
        ("Land", "land_value_tax"),
        ("Corp.", "corporate_tax"),
        ("Pig.", "pigouvian_tax"),
        ("Cap. gains", "capital_gains_tax"),
        ("Estate", "estate_tax"),
    ]
    x = np.arange(len(instruments))
    w = 0.35
    base_vals = [getattr(comparison.baseline.revenue, i[1]) * 100 for i in instruments]
    ref_vals  = [getattr(comparison.reform.revenue,   i[1]) * 100 for i in instruments]
    ax.bar(x - w / 2, base_vals, w, color=_BLUE, alpha=0.85)
    ax.bar(x + w / 2, ref_vals,  w, color=_ORANGE, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([i[0] for i in instruments], rotation=30, ha="right", fontsize=7)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=1))
    ax.set_title("Revenue (% GDP)", fontsize=9)

def _draw_incidence_on(ax: plt.Axes, comparison: PolicyComparison) -> None:
    ic = comparison.incidence_change
    groups = list(GROUPS)
    values = [ic[g] * 100 for g in groups]
    labels = [g.replace("_", "\n") for g in groups]
    colors = [_GREEN if v <= 0 else _RED for v in values]
    ax.bar(labels, values, color=colors, width=0.6)
    ax.axhline(0, color=_DARK, linewidth=0.8)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=1))
    ax.set_title("Burden change by group (pp)", fontsize=9)


# ---------------------------------------------------------------------------
# Per-instrument incidence stacked bar chart
# ---------------------------------------------------------------------------

def plot_instrument_incidence(
    baseline_breakdown: Dict[str, Dict[str, float]],
    reform_breakdown: Dict[str, Dict[str, float]],
    baseline_label: str = "Baseline",
    reform_label: str = "Reform",
) -> plt.Figure:
    """
    Stacked bar chart of tax burden by instrument for each income group.
    Side-by-side: baseline (left) and reform (right).

    breakdown dicts: Dict[instrument_name, Dict[group, burden_fraction_of_group_income]]
    """
    instruments = list(baseline_breakdown.keys())
    # Use a distinct color palette
    palette = [
        "#2166AC", "#D6604D", "#4DAC26", "#F4A582",
        "#762A83", "#AF8DC3", "#1B7837", "#D9EF8B",
    ]
    colors = {inst: palette[i % len(palette)] for i, inst in enumerate(instruments)}

    x = np.arange(len(GROUPS))
    group_labels = [g.replace("_", "\n") for g in GROUPS]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, breakdown, title in [(ax1, baseline_breakdown, baseline_label),
                                  (ax2, reform_breakdown, reform_label)]:
        bottoms = np.zeros(len(GROUPS))
        for inst in instruments:
            vals = np.array([max(0.0, breakdown.get(inst, {}).get(g, 0.0)) * 100 for g in GROUPS])
            ax.bar(x, vals, bottom=bottoms, color=colors[inst], label=inst, width=0.65)
            bottoms += vals

        ax.set_xticks(x)
        ax.set_xticklabels(group_labels, fontsize=9)
        if ax is ax1:
            ax.set_ylabel("Effective burden (% of group income)", fontsize=9)
        ax.set_title(f"{title}", fontsize=10, fontweight="bold")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=0))
        ax.spines[["top", "right"]].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=colors[i]) for i in instruments]
    fig.legend(handles, instruments, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.suptitle("Tax burden by instrument and income group", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Pareto frontier scatter
# ---------------------------------------------------------------------------

def plot_pareto_frontier(
    frontier_data: List[Tuple[str, float, float]],
) -> plt.Figure:
    """
    Scatter plot of GDP change (x) vs Q1 burden change (y, inverted: positive = Q1 better off).
    Each point = one named preset vs Current Law.

    frontier_data: list of (label, gdp_change_pct, q1_burden_change_pp)
      where q1_burden_change_pp < 0 means Q1 burden FELL (reform helped Q1).
    """
    if not frontier_data:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.text(0.5, 0.5, "No frontier data available", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    labels, gdp_changes, q1_changes = zip(*frontier_data)

    # Flip Q1 sign: positive on chart = Q1 better off
    q1_display = [-v for v in q1_changes]

    scatter_colors = [_GREEN if y >= 0 else _RED for y in q1_display]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(gdp_changes, q1_display, c=scatter_colors, s=140, zorder=5,
               edgecolors="white", linewidths=1.5)

    for i, label in enumerate(labels):
        ax.annotate(
            label,
            (gdp_changes[i], q1_display[i]),
            textcoords="offset points",
            xytext=(7, 3),
            fontsize=7.5,
            color=_DARK,
        )

    ax.axhline(0, color=_GREY, linewidth=0.8, linestyle="--")
    ax.axvline(0, color=_GREY, linewidth=0.8, linestyle="--")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=1))
    ax.set_xlabel("Long-run GDP change vs Current Law (%)", fontsize=10)
    ax.set_ylabel("Q1 burden change (pp of income)\npositive = Q1 better off", fontsize=10)
    ax.set_title("Growth–Equity Frontier: All Presets vs Current Law", fontsize=11, fontweight="bold")
    ax.text(0.97, 0.97, "↗ better on both", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color=_GREEN, alpha=0.8)
    ax.text(0.97, 0.03, "↘ growth only", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, color=_RED, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig
