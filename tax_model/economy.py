"""
Economy — the core equilibrium solver.

Architecture
------------
Economy.solve(policy) finds the steady-state equilibrium of a two-period
Overlapping Generations (OLG) model given a tax policy configuration.

The model
---------
  Production:   Y = A * K^α * L^(1-α)   (Cobb-Douglas)
  Households:   two periods — work (save, pay labor taxes) then retire (consume)
  Government:   budget constraint (revenue vs. spending); surplus/deficit reported
  Markets:      labor, capital, goods all clear in steady state

Normalization
-------------
  The baseline (no-tax steady state) has L₀ = 1 and K₀ determined by the
  Euler equation r₀ = ρ (discount rate). Y₀ = A·K₀^α is used to normalize
  all outputs so that baseline GDP = 1.0.

Capital market clearing
-----------------------
  Two-period OLG: the young save their after-tax labor income; this becomes
  the next period's capital stock.

  In steady state: K = S where S is aggregate savings.
  We parameterize savings as:
    K_supply(w_net, r_net, L) = K₀ · (w_net/w₀) · L · [1 + ψ·(r_net − r₀)/r₀]

  This is calibrated so that K_supply = K₀ at baseline (L=1, w_net=w₀, r_net=r₀).
  ψ = saving_rate_sensitivity controls how strongly capital accumulation responds
  to the after-tax return on capital.

  This avoids conflating annual savings rates (from household survey data) with
  the "lifetime" savings fraction implicit in the two-period OLG framework.

Solver
------
  scipy.optimize.root with Powell's hybrid method (hybr). We solve for
  (K, L, r) in log-space to enforce positivity.

Open economy / capital mobility
---------------------------------
  When cal.macro.capital_mobility = 1 (small open economy), the world interest
  rate pins r; K adjusts to clear the capital market. When capital_mobility = 0
  (closed), r is determined domestically by MPK. Intermediate values interpolate.
"""

from __future__ import annotations
from typing import Tuple

import numpy as np
from scipy.optimize import root

from .calibration import Calibration
from .policy import TaxPolicy
from .results import ModelResult
from .government import (
    Allocation,
    compute_revenue,
    compute_incidence,
)


# World interest rate assumption for small open economy (annual, real)
_WORLD_REAL_RETURN = 0.04


class Economy:
    """
    Macroeconomic model.  One Economy per Calibration; call .solve() with
    different TaxPolicy objects to compare scenarios.

    Parameters
    ----------
    calibration : Calibration
        Loaded from us_2024.yaml (or equivalent).
    """

    def __init__(self, calibration: Calibration) -> None:
        self.cal = calibration
        # No-tax steady-state baseline: L₀ = 1, K₀ from Euler equation
        self._baseline_K, self._baseline_L = self._no_tax_steady_state()
        self._baseline_Y  = self._production(self._baseline_K, self._baseline_L)
        self._baseline_w  = self._marginal_product_labor(self._baseline_K, self._baseline_L)
        self._baseline_r  = self._marginal_product_capital(self._baseline_K, self._baseline_L)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self, policy: TaxPolicy) -> ModelResult:
        """
        Find the steady-state equilibrium under `policy` and return a
        ModelResult.  All quantities normalized so no-tax baseline GDP = 1.0.
        """
        K, L, r = self._solve_equilibrium(policy)
        w = self._marginal_product_labor(K, L)
        Y = self._production(K, L)

        # Normalize by no-tax baseline
        K_norm = K / self._baseline_K
        L_norm = L / self._baseline_L
        Y_norm = Y / self._baseline_Y
        w_norm = w / self._baseline_w
        r_net  = r  # rate, not scaled

        δ = self.cal.production.depreciation_rate
        I_norm = (δ * K) / self._baseline_Y
        C_norm = Y_norm - I_norm - self.cal.government.spending_gdp_ratio

        # Externality quantity scales with economic activity
        ext_qty = self.cal.externality.carbon_intensity_per_gdp * Y_norm
        ext_dmg = ext_qty * self.cal.externality.social_cost_carbon

        alloc = Allocation(
            gdp=Y_norm,
            capital_stock=K_norm,
            labor_supply=L_norm,
            consumption=max(C_norm, 0.01),
            investment=I_norm,
            wage=w_norm,
            return_on_capital=r_net,
            land_rent=self.cal.land.land_rent_gdp_ratio * Y_norm,
            externality_quantity=ext_qty,
        )

        revenue   = compute_revenue(alloc, policy, self.cal)
        incidence = compute_incidence(alloc, policy, revenue, self.cal)

        gov_spending   = self.cal.government.spending_gdp_ratio
        budget_balance = revenue.total - gov_spending

        return ModelResult(
            policy_label=policy.label,
            gdp=Y_norm,
            capital_stock=K_norm,
            labor_supply=L_norm,
            consumption=max(C_norm, 0.01),
            investment=I_norm,
            wage=w_norm,
            return_on_capital=r_net,
            revenue=revenue,
            government_spending=gov_spending,
            budget_balance=budget_balance,
            incidence=incidence,
            externality_quantity=ext_qty,
            externality_damage=ext_dmg,
        )

    # ------------------------------------------------------------------
    # Equilibrium solver
    # ------------------------------------------------------------------

    def _solve_equilibrium(self, policy: TaxPolicy) -> Tuple[float, float, float]:
        """
        Solve for steady-state (K, L, r).  We work in log-space to ensure
        positivity and improve numerical conditioning.
        """
        # Initial guess: near the no-tax baseline
        x0 = np.array([
            np.log(self._baseline_K),
            np.log(self._baseline_L),
            np.log(max(self._baseline_r, 0.01)),
        ])

        result = root(
            self._residuals,
            x0,
            args=(policy,),
            method="hybr",
            options={"maxfev": 3000, "xtol": 1e-10},
        )

        if not result.success:
            # Retry with slightly perturbed initial guess
            result = root(
                self._residuals,
                x0 * 0.98,
                args=(policy,),
                method="hybr",
                options={"maxfev": 5000, "xtol": 1e-8},
            )

        K, L, r = np.exp(result.x)
        return float(K), float(L), float(r)

    def _residuals(self, log_x: np.ndarray, policy: TaxPolicy) -> np.ndarray:
        """
        Residual vector [R1, R2, R3] = 0 at equilibrium.

          R1: labor market — supply equals demand
          R2: capital market — savings equal capital stock
          R3: return on capital — production FOC consistent with r
        """
        K, L, r = np.exp(log_x)

        # Factor prices from production FOCs
        w      = self._marginal_product_labor(K, L)
        r_prod = self._marginal_product_capital(K, L)

        # After-tax factor prices
        τ_L   = self._effective_labor_wedge(policy)
        τ_K   = self._effective_capital_wedge(policy)
        w_net = w * (1.0 - τ_L)
        r_net = r * (1.0 - τ_K)

        # Household labor supply (Frisch): L = (w_net / w₀_net)^η
        # At baseline (no tax), w₀_net = w₀; labor supply = L₀ = 1.
        τ_L_base   = 0.0  # no-tax baseline
        w0_net     = self._baseline_w * (1.0 - τ_L_base)
        L_supply   = self._baseline_L * (w_net / w0_net) ** self.cal.households.frisch_elasticity

        # Household savings → capital supply
        K_supply = self._capital_supply(w_net, r_net, L)

        # Open/closed economy blend for capital market
        mob      = self.cal.macro.capital_mobility
        r_target = (1.0 - mob) * r_prod + mob * _WORLD_REAL_RETURN

        R1 = (L_supply - L)      / max(L, 1e-8)         # labor market
        R2 = (K_supply - K)      / max(K, 1e-8)         # capital market
        R3 = (r       - r_target) / max(r_target, 1e-8)  # return consistency

        return np.array([R1, R2, R3])

    # ------------------------------------------------------------------
    # Production
    # ------------------------------------------------------------------

    def _production(self, K: float, L: float) -> float:
        """Y = A · K^α · L^(1-α)"""
        α = self.cal.production.capital_share
        A = self.cal.production.tfp_level
        return A * (K ** α) * (L ** (1.0 - α))

    def _marginal_product_labor(self, K: float, L: float) -> float:
        """w = (1-α) · Y / L"""
        α = self.cal.production.capital_share
        return (1.0 - α) * self._production(K, L) / L

    def _marginal_product_capital(self, K: float, L: float) -> float:
        """r = α · Y / K − δ  (net of depreciation)"""
        α = self.cal.production.capital_share
        δ = self.cal.production.depreciation_rate
        return α * self._production(K, L) / K - δ

    # ------------------------------------------------------------------
    # Household behavior
    # ------------------------------------------------------------------

    def _capital_supply(self, w_net: float, r_net: float, L: float) -> float:
        """
        Aggregate capital supply (savings of the young generation).

        Calibrated so K_supply = K₀ at the no-tax baseline.
        Responds to:
          - After-tax wage (proportional: higher w_net → more savings)
          - After-tax return (saving_rate_sensitivity captures IES response)
          - Labor supply (more workers → more aggregate savings)

        K_supply = K₀ · (w_net/w₀) · L/L₀ · [1 + ψ · (r_net − r₀)/r₀]
        """
        τ_L_base = 0.0
        w0_net   = self._baseline_w * (1.0 - τ_L_base)
        r0       = max(self._baseline_r, 0.01)
        ψ        = self.cal.households.saving_rate_sensitivity

        income_scale  = (w_net / w0_net) * (L / self._baseline_L)
        return_effect = 1.0 + ψ * (r_net - r0) / r0

        K_supply = self._baseline_K * income_scale * return_effect
        return float(np.clip(K_supply, 1e-6, 1e6))

    # ------------------------------------------------------------------
    # Tax wedge calculations
    # ------------------------------------------------------------------

    def _effective_labor_wedge(self, policy: TaxPolicy) -> float:
        """
        Effective marginal tax wedge on labor income.
        Combines income tax + payroll tax + consumption tax (which reduces the
        real purchasing power of the after-tax wage).
        """
        from .government import _avg_effective_labor_rate, _avg_effective_payroll_rate
        τ_L      = _avg_effective_labor_rate(policy, self.cal)
        τ_payroll = _avg_effective_payroll_rate(policy, self.cal)
        τ_C      = policy.consumption.rate
        # Combined wedge: 1 − (1−τ_L−τ_payroll)(1−τ_C_eff)
        τ_C_eff = τ_C / (1.0 + τ_C)
        return float(np.clip(
            1.0 - (1.0 - τ_L - τ_payroll) * (1.0 - τ_C_eff), 0.0, 0.95
        ))

    def _effective_capital_wedge(self, policy: TaxPolicy) -> float:
        """
        Effective marginal tax wedge on capital income.
        Corporate tax (discounted for immediate expensing) + capital gains tax.
        Land tax is excluded (inelastic supply; doesn't affect capital allocation).
        """
        corp = policy.corporate.rate
        if policy.corporate.immediate_expensing:
            # EMTR on new investment → ~0; residual applies to quasi-rents only
            corp_emtr = corp * 0.15
        else:
            corp_emtr = corp

        cg_wedge = policy.capital_gains.effective_rate * 0.30  # CG ≈ 30% of total return
        return float(np.clip(corp_emtr + cg_wedge, 0.0, 0.95))

    # ------------------------------------------------------------------
    # No-tax steady state: baseline (K₀, L₀)
    # ------------------------------------------------------------------

    def _no_tax_steady_state(self) -> Tuple[float, float]:
        """
        Compute the no-tax steady-state K₀ and L₀.

        L₀ = 1 (normalization).
        K₀ is pinned by the household Euler equation at zero tax:
            MPK(K₀, L₀) − δ = ρ  →  α·A·K₀^(α−1) − δ = ρ
            K₀ = ((ρ + δ) / (α·A))^(1/(α−1))

        This gives K/Y = α/(ρ+δ). With α=0.35, ρ=0.04, δ=0.10 → K/Y=2.5 ✓
        """
        ρ = self.cal.households.discount_rate
        δ = self.cal.production.depreciation_rate
        α = self.cal.production.capital_share
        A = self.cal.production.tfp_level

        K0 = ((ρ + δ) / (α * A)) ** (1.0 / (α - 1.0))
        L0 = 1.0
        return float(K0), float(L0)
