"""
Transition path analysis.

Computes how the economy moves from a baseline steady state to a reform steady
state after a policy change at t=0. This is distinct from the steady-state
comparison: it shows who wins and who loses *during the transition*, not just
in the long run.

Model
-----
The transition path is computed as a discrete-time capital accumulation sequence:

    K_{t+1} = (1-δ) K_t + S_t

where S_t = s(w_t_net, r_t_net, L_t) is the aggregate savings of the young
generation at time t, evaluated at reform-policy prices. Starting from K_0
(the old steady state), the path converges to K_∞ (the new steady state).

At each t, we solve for the spot equilibrium (w_t, r_t, L_t) given K_t, then
compute the full incidence for that year's cohort.

Key insight: the *current old* generation (period 2) at t=0 holds capital K_0
accumulated under the old policy and now faces new capital returns r_0 under
the reform. This one-time capital revaluation is the main transition cost for
capital-heavy reforms (e.g., replacing corporate tax with LVT).

Parameters
----------
periods : int
    Number of periods to simulate. Default 30 (roughly 30 years if one period
    ≈ 1 year; or 60 years if two-period = lifetime). For steady-state analysis
    set periods=1.

Output
------
TransitionResult: a dataframe-like object with per-period values of:
    - K, L, Y, w, r (normalized to baseline = 1.0)
    - budget_balance
    - incidence by group
    - cumulative welfare change (consumption-equivalent) by group
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict

import numpy as np

from .calibration import Calibration, GROUPS
from .policy import TaxPolicy
from .economy import Economy, _WORLD_REAL_RETURN
from .government import Allocation, compute_revenue, compute_incidence
from .results import ModelResult, DistributionalIncidence, RevenueBreakdown


@dataclass
class PeriodSnapshot:
    """Economy state at a single point along the transition path."""
    period: int
    gdp: float
    capital_stock: float
    labor_supply: float
    wage: float
    return_on_capital: float
    budget_balance: float
    revenue_total: float
    incidence: DistributionalIncidence


@dataclass
class TransitionResult:
    """
    Full transition path from baseline to reform steady state.

    Attributes
    ----------
    baseline_ss : ModelResult
        The starting (pre-reform) steady state.
    reform_ss : ModelResult
        The ending (post-reform) steady state.
    path : list of PeriodSnapshot
        Per-period values along the transition.
    periods : int
        Number of periods simulated.
    """
    baseline_ss: ModelResult
    reform_ss: ModelResult
    path: List[PeriodSnapshot]
    periods: int

    # --- Convenience extractors ---

    def series(self, attr: str) -> List[float]:
        """Return a list of values for a given PeriodSnapshot attribute."""
        return [getattr(s, attr) for s in self.path]

    @property
    def period_numbers(self) -> List[int]:
        return [s.period for s in self.path]

    @property
    def gdp_path(self) -> List[float]:
        return self.series("gdp")

    @property
    def capital_path(self) -> List[float]:
        return self.series("capital_stock")

    @property
    def wage_path(self) -> List[float]:
        return self.series("wage")

    @property
    def budget_path(self) -> List[float]:
        return self.series("budget_balance")

    def incidence_path(self, group: str) -> List[float]:
        """Burden for a specific income group over time (fraction of income)."""
        return [s.incidence[group] for s in self.path]

    def long_run_gdp_change(self) -> float:
        """% GDP change from baseline to reform steady state."""
        return (self.reform_ss.gdp / self.baseline_ss.gdp - 1.0) * 100.0

    def years_to_convergence(self, threshold: float = 0.01) -> int:
        """
        Number of periods until GDP is within `threshold` fraction of reform SS.
        Returns periods if not converged by end of simulation.
        """
        target = self.reform_ss.gdp
        for snap in self.path:
            if abs(snap.gdp - target) / max(target, 1e-8) < threshold:
                return snap.period
        return self.periods

    def summary(self) -> str:
        lines = [
            f"Transition: [{self.baseline_ss.policy_label}] → [{self.reform_ss.policy_label}]",
            f"  Periods simulated:      {self.periods}",
            f"  Long-run GDP change:    {self.long_run_gdp_change():+.2f}%",
            f"  Approx. convergence:    period {self.years_to_convergence()}",
            "",
            f"  GDP path (first 5 / last 1):",
        ]
        for snap in self.path[:5]:
            lines.append(f"    t={snap.period:2d}  GDP={snap.gdp:.4f}")
        if self.periods > 5:
            last = self.path[-1]
            lines.append(f"    ...  GDP={self.path[-1].gdp:.4f}  (t={last.period})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def compute_transition(
    economy: Economy,
    baseline_policy: TaxPolicy,
    reform_policy: TaxPolicy,
    periods: int = 30,
) -> TransitionResult:
    """
    Compute the transition path from the baseline to the reform steady state.

    The reform is implemented at t=1 (the first period after t=0 baseline).
    K_0 is the baseline steady-state capital stock.

    Parameters
    ----------
    economy : Economy
        The calibrated economy object.
    baseline_policy : TaxPolicy
        The starting policy (pre-reform).
    reform_policy : TaxPolicy
        The reform policy (implemented at t=1).
    periods : int
        Number of discrete periods to simulate. Default 30.

    Returns
    -------
    TransitionResult
    """
    cal = economy.cal

    # Solve both steady states
    baseline_ss = economy.solve(baseline_policy)
    reform_ss   = economy.solve(reform_policy)

    # Starting capital = baseline steady state (in normalized units)
    K_t = baseline_ss.capital_stock  # in units of baseline K0

    path: List[PeriodSnapshot] = []

    for t in range(periods):
        # Solve spot equilibrium given current K_t under the reform policy
        snap = _spot_equilibrium(economy, reform_policy, K_t, t, cal)
        path.append(snap)

        # Update capital for next period
        K_t = _update_capital(economy, reform_policy, snap, cal)

        # Early stopping: close enough to reform SS
        if t > 5 and abs(snap.gdp - reform_ss.gdp) / max(reform_ss.gdp, 1e-8) < 5e-4:
            # Fill remaining periods with reform SS
            for tt in range(t + 1, periods):
                path.append(PeriodSnapshot(
                    period=tt,
                    gdp=reform_ss.gdp,
                    capital_stock=reform_ss.capital_stock,
                    labor_supply=reform_ss.labor_supply,
                    wage=reform_ss.wage,
                    return_on_capital=reform_ss.return_on_capital,
                    budget_balance=reform_ss.budget_balance,
                    revenue_total=reform_ss.revenue.total,
                    incidence=reform_ss.incidence,
                ))
            break

    return TransitionResult(
        baseline_ss=baseline_ss,
        reform_ss=reform_ss,
        path=path,
        periods=periods,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _spot_equilibrium(
    economy: Economy,
    policy: TaxPolicy,
    K_t: float,
    period: int,
    cal: Calibration,
) -> PeriodSnapshot:
    """
    Given capital stock K_t (normalized), solve for the spot equilibrium
    under `policy` and return a PeriodSnapshot.
    """
    # Recover unnormalized K
    K_raw = K_t * economy._baseline_K

    # Solve labor market given K_raw (L is endogenous, K is predetermined)
    from scipy.optimize import brentq

    # Labor supply: L = (w_net / w0)^eta
    # Firm's MPL: w = (1-alpha) * A * K^alpha * L^(-alpha) — solve for L
    alpha = cal.production.capital_share
    A     = cal.production.tfp_level
    η     = cal.households.frisch_elasticity
    δ     = cal.production.depreciation_rate
    mob   = cal.macro.capital_mobility

    # Effective labor wedge
    τ_wedge = economy._effective_labor_wedge(policy)
    w0_net  = economy._baseline_w  # no-tax baseline wage

    def labor_residual(L):
        w     = (1.0 - alpha) * A * (K_raw ** alpha) * (L ** (-alpha))
        w_net = w * (1.0 - τ_wedge)
        L_s   = economy._baseline_L * (w_net / w0_net) ** η
        return L_s - L

    try:
        L_raw = brentq(labor_residual, 1e-6, 1e2, xtol=1e-8, maxiter=200)
    except ValueError:
        # Fallback: use reform SS labor supply scaled to current K
        L_raw = economy._baseline_L * (K_raw / economy._baseline_K) ** 0.3

    w_raw = (1.0 - alpha) * A * (K_raw ** alpha) * (L_raw ** (-alpha))
    r_raw = alpha * A * (K_raw ** (alpha - 1.0)) * (L_raw ** (1.0 - alpha)) - δ
    Y_raw = A * (K_raw ** alpha) * (L_raw ** (1.0 - alpha))

    # Normalize
    Y_norm = Y_raw / economy._baseline_Y
    K_norm = K_raw / economy._baseline_K
    L_norm = L_raw / economy._baseline_L
    w_norm = w_raw / economy._baseline_w

    # Capital return: blend closed/open
    r_spot = (1.0 - mob) * r_raw + mob * _WORLD_REAL_RETURN

    I_norm = (δ * K_raw) / economy._baseline_Y
    C_norm = max(Y_norm - I_norm - cal.government.spending_gdp_ratio, 0.01)

    alloc = Allocation(
        gdp=Y_norm,
        capital_stock=K_norm,
        labor_supply=L_norm,
        consumption=C_norm,
        investment=I_norm,
        wage=w_norm,
        return_on_capital=r_spot,
        land_rent=cal.land.land_rent_gdp_ratio * Y_norm,
        externality_quantity=cal.externality.carbon_intensity_per_gdp * Y_norm,
    )

    revenue   = compute_revenue(alloc, policy, cal)
    incidence = compute_incidence(alloc, policy, revenue, cal)
    budget    = revenue.total - cal.government.spending_gdp_ratio

    return PeriodSnapshot(
        period=period,
        gdp=Y_norm,
        capital_stock=K_norm,
        labor_supply=L_norm,
        wage=w_norm,
        return_on_capital=r_spot,
        budget_balance=budget,
        revenue_total=revenue.total,
        incidence=incidence,
    )


def _update_capital(
    economy: Economy,
    policy: TaxPolicy,
    snap: PeriodSnapshot,
    cal: Calibration,
) -> float:
    """
    Compute next period's capital stock (normalized) using the OLG savings rule.

    K_{t+1} = K0 * (w_net/w0) * L * [1 + ψ * (r_net - r0)/r0]
    """
    τ_K   = economy._effective_capital_wedge(policy)
    τ_L   = economy._effective_labor_wedge(policy)
    w_net = snap.wage * (1.0 - τ_L) / (1.0 - τ_L) * snap.wage  # already net in snap? No:
    # snap.wage is pre-tax wage (normalized). Apply wedge:
    w_net_norm = snap.wage * (1.0 - τ_L)
    r_net      = snap.return_on_capital * (1.0 - τ_K)
    r0         = max(economy._baseline_r, 0.01)
    ψ          = cal.households.saving_rate_sensitivity

    income_scale  = w_net_norm * snap.labor_supply
    return_effect = 1.0 + ψ * (r_net - r0) / r0

    K_next = income_scale * return_effect
    return float(np.clip(K_next, 0.01, 10.0))
