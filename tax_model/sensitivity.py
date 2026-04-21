"""
Sensitivity and uncertainty analysis.

Two modes:
  1. Parameter sweep: vary one calibration parameter across a fixed grid.
  2. Monte Carlo: draw parameters from distributions, report credible intervals.

Both return lists of PolicyComparison objects that visualization.py can plot.
"""

from __future__ import annotations
import copy
from dataclasses import replace
from typing import Dict, List, Optional, Callable
import numpy as np

from .calibration import Calibration
from .policy import TaxPolicy
from .results import PolicyComparison
from .economy import Economy


# ---------------------------------------------------------------------------
# Single-parameter sweep
# ---------------------------------------------------------------------------

def sweep(
    base_calibration: Calibration,
    baseline_policy: TaxPolicy,
    reform_policy: TaxPolicy,
    parameter_path: str,
    values: List[float],
) -> List[PolicyComparison]:
    """
    Vary one calibration parameter and return a comparison for each value.

    parameter_path uses dot notation, e.g.:
        "households.frisch_elasticity"
        "macro.capital_mobility"
        "production.capital_share"

    Returns a list of PolicyComparison objects (one per value in `values`).
    """
    comparisons = []
    for val in values:
        cal = _set_nested(base_calibration, parameter_path, val)
        economy = Economy(cal)
        baseline = economy.solve(baseline_policy)
        reform = economy.solve(reform_policy)
        comparisons.append(PolicyComparison(baseline=baseline, reform=reform))
    return comparisons


# ---------------------------------------------------------------------------
# Monte Carlo over parameter uncertainty
# ---------------------------------------------------------------------------

def monte_carlo(
    base_calibration: Calibration,
    baseline_policy: TaxPolicy,
    reform_policy: TaxPolicy,
    n_draws: int = 500,
    seed: Optional[int] = 42,
    param_distributions: Optional[Dict[str, Callable]] = None,
) -> List[PolicyComparison]:
    """
    Draw calibration parameters from uncertainty distributions and return
    a PolicyComparison for each draw.

    param_distributions: dict mapping parameter_path → zero-argument callable
        that returns a float sample. Defaults to the built-in prior below.

    The resulting list can be used to compute credible intervals over any
    output metric.

    Example
    -------
        results = monte_carlo(cal, baseline, reform, n_draws=1000)
        gdp_changes = [r.gdp_change_pct for r in results]
        print(f"GDP effect: {np.percentile(gdp_changes, 5):.2f}% – {np.percentile(gdp_changes, 95):.2f}%")
    """
    rng = np.random.default_rng(seed)

    if param_distributions is None:
        param_distributions = _default_distributions(rng)

    comparisons = []
    for _ in range(n_draws):
        cal = copy.deepcopy(base_calibration)
        for path, sampler in param_distributions.items():
            val = sampler()
            cal = _set_nested(cal, path, val)
        economy = Economy(cal)
        baseline = economy.solve(baseline_policy)
        reform = economy.solve(reform_policy)
        comparisons.append(PolicyComparison(baseline=baseline, reform=reform))
    return comparisons


def credible_interval(
    comparisons: List[PolicyComparison],
    metric: str = "gdp_change_pct",
    lower: float = 5.0,
    upper: float = 95.0,
) -> Dict[str, float]:
    """
    Extract a credible interval for a scalar metric across Monte Carlo draws.

    metric: any attribute of PolicyComparison (e.g. "gdp_change_pct",
            "capital_change_pct", "budget_balance_change").

    Returns {"mean": ..., "lower": ..., "upper": ...}.
    """
    values = [getattr(c, metric) for c in comparisons]
    return {
        "mean": float(np.mean(values)),
        "lower": float(np.percentile(values, lower)),
        "upper": float(np.percentile(values, upper)),
    }


# ---------------------------------------------------------------------------
# Default parameter distributions (calibrated to empirical uncertainty)
# ---------------------------------------------------------------------------

def _default_distributions(rng: np.random.Generator) -> Dict[str, Callable]:
    """
    Conservative uncertainty bounds around consensus parameter values.
    Based on ranges reported in the empirical literature.
    """
    def truncated_normal(mean, std, lo, hi):
        def _sample():
            while True:
                v = rng.normal(mean, std)
                if lo <= v <= hi:
                    return float(v)
        return _sample

    return {
        "households.frisch_elasticity":       truncated_normal(0.50, 0.10, 0.20, 0.90),
        "households.ies":                     truncated_normal(0.50, 0.10, 0.20, 1.00),
        "households.saving_rate_sensitivity": truncated_normal(0.30, 0.08, 0.05, 0.60),
        "production.capital_share":           truncated_normal(0.35, 0.02, 0.28, 0.42),
        "macro.capital_mobility":             truncated_normal(0.50, 0.15, 0.00, 1.00),
        "macro.elasticity_of_taxable_income": truncated_normal(0.25, 0.08, 0.05, 0.50),
    }


# ---------------------------------------------------------------------------
# Internal helper: set a nested attribute by dot-path
# ---------------------------------------------------------------------------

def _set_nested(obj, path: str, value: float):
    """
    Return a (shallow-ish) copy of `obj` with the attribute at `path` set to `value`.
    Path is dot-separated, e.g. "households.frisch_elasticity".
    """
    parts = path.split(".")
    obj = copy.deepcopy(obj)
    target = obj
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)
    return obj
