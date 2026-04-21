"""
GovernmentBudget
----------------
Revenue accounting, incidence calculation, and budget balance.

This module sits on top of the equilibrium allocation produced by Economy.
It does pure accounting — no optimization or root-finding.

Key design: incidence is computed in two passes.
  Pass 1: Statutory burden — who writes the check.
  Pass 2: Economic burden — shift incidence using the capital mobility
          assumption and behavioral elasticities. This is where open vs.
          closed economy matters.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

import numpy as np

from .calibration import Calibration, GROUPS
from .policy import TaxPolicy
from .results import DistributionalIncidence, RevenueBreakdown


# ---------------------------------------------------------------------------
# Internal allocation bundle (passed from economy.py to government.py)
# ---------------------------------------------------------------------------

@dataclass
class Allocation:
    """
    Equilibrium allocation produced by the solver. All quantities normalized
    so baseline GDP = 1.0. Factor prices are real and pre-tax.
    """
    gdp: float
    capital_stock: float
    labor_supply: float
    consumption: float
    investment: float
    wage: float              # pre-tax real wage per unit labor
    return_on_capital: float # pre-tax real return on capital (net of depreciation)
    land_rent: float         # pre-tax real land rent
    externality_quantity: float  # tons CO2 (or equivalent) per unit GDP


# ---------------------------------------------------------------------------
# Revenue calculation
# ---------------------------------------------------------------------------

def compute_revenue(
    alloc: Allocation,
    policy: TaxPolicy,
    cal: Calibration,
) -> RevenueBreakdown:
    """
    Compute tax revenue by instrument as a fraction of baseline GDP.

    All quantities are expressed relative to GDP so they are scale-invariant.
    """
    gdp = alloc.gdp

    # --- Labor income tax ---
    # ETI feedback: higher rates shrink the base
    labor_income_total = alloc.wage * alloc.labor_supply
    eti = cal.macro.elasticity_of_taxable_income

    # Compute weighted average effective rate across groups
    avg_eff_rate = _avg_effective_labor_rate(policy, cal)
    # ETI adjustment: base shrinks by eti * rate
    labor_base = labor_income_total * (1.0 - eti * avg_eff_rate)
    labor_revenue = avg_eff_rate * labor_base / gdp

    # --- Payroll tax (FICA) ---
    # Payroll taxes apply to wages (not capital income). The ceiling means
    # effective rate declines above the SS wage base.
    avg_payroll_rate = _avg_effective_payroll_rate(policy, cal)
    # Payroll tax base is labor income; smaller ETI applies (less avoidance margin)
    payroll_base = labor_income_total * (1.0 - eti * 0.5 * avg_payroll_rate)
    payroll_revenue = avg_payroll_rate * payroll_base / gdp

    # --- Consumption tax ---
    # Prebate cost = prebate_fraction * poverty_line_consumption * population
    # poverty_line_consumption = poverty_line_consumption_share * (C/N) * N = poverty_line_share * C
    cons_base = alloc.consumption
    cons_revenue = policy.consumption.rate * cons_base / gdp
    prebate_cost = -(
        policy.consumption.prebate_fraction
        * cal.government.poverty_line_consumption_share
        * cons_base
        / gdp
        * policy.consumption.rate
    )

    # --- Land value tax ---
    land_value = cal.land.land_value_gdp_ratio  # fraction of GDP
    land_revenue = policy.land_value.rate * land_value

    # --- Corporate tax ---
    # Capital income = return_on_capital * capital_stock
    # Immediate expensing drives EMTR on new investment toward zero
    # but statutory rate still applies to inframarginal returns
    capital_income = alloc.return_on_capital * alloc.capital_stock
    cit = cal.macro.capital_income_eti
    if policy.corporate.immediate_expensing:
        # Economic depreciation is immediately deducted → EMTR ~ 0 on new investment
        # Revenue comes from returns above cost of capital (quasi-rents)
        effective_corp_rate = policy.corporate.rate * 0.40
    else:
        effective_corp_rate = policy.corporate.rate
    corp_base = capital_income * (1.0 - cit * effective_corp_rate)
    corp_revenue = effective_corp_rate * corp_base / gdp

    # --- Pigouvian tax ---
    # Externality quantity scales with GDP activity
    ext_quantity = alloc.externality_quantity * gdp
    pig_gross = policy.pigouvian.rate_per_unit * ext_quantity / gdp
    pig_dividend = -(pig_gross * policy.pigouvian.dividend_recycling_fraction)
    pig_labor_offset = pig_gross * policy.pigouvian.labor_tax_offset_fraction
    # Net Pigouvian revenue = gross - dividend - labor_offset (labor offset reduces labor tax burden)
    pig_net = pig_gross + pig_dividend  # pig_labor_offset reduces labor revenue instead

    # Adjust labor revenue down by Pigouvian labor offset
    labor_revenue -= pig_labor_offset

    # --- Capital gains tax ---
    # Capital gains are a share of capital income; effective rate accounts for lock-in
    cg_income = sum(
        cal.income_distribution.capital_gains_shares[g]
        for g in GROUPS
    ) * capital_income  # effectively = total capital gains = capital income (simplified)
    cg_income = capital_income * 0.30  # approx: 30% of capital income realized as gains annually
    cg_effective_rate = policy.capital_gains.effective_rate
    if policy.capital_gains.inflation_indexed:
        # Only real gains taxed; reduce base by inflation/nominal ratio
        nom_return = alloc.return_on_capital + cal.macro.target_labor_share  # rough nominal
        inflation_share = policy.capital_gains.inflation_rate / max(nom_return, 0.01)
        cg_income *= (1.0 - inflation_share)
    cg_revenue = cg_effective_rate * cg_income / gdp

    return RevenueBreakdown(
        labor_income_tax=max(0.0, labor_revenue),
        payroll_tax=max(0.0, payroll_revenue),
        consumption_tax=max(0.0, cons_revenue),
        land_value_tax=land_revenue,
        corporate_tax=max(0.0, corp_revenue),
        pigouvian_tax=max(0.0, pig_net),
        capital_gains_tax=max(0.0, cg_revenue),
        prebate_cost=prebate_cost,
        pigouvian_dividend_cost=pig_dividend,
    )


# ---------------------------------------------------------------------------
# Incidence calculation
# ---------------------------------------------------------------------------

def compute_incidence(
    alloc: Allocation,
    policy: TaxPolicy,
    revenue: RevenueBreakdown,
    cal: Calibration,
) -> DistributionalIncidence:
    """
    Compute net tax burden as a fraction of pre-tax income for each group.

    Approach:
    1. Assign statutory burden by factor income source.
    2. Shift corporate burden using capital_mobility parameter.
    3. Subtract transfers (prebates, Pigouvian dividends).
    4. Normalize by group income share.

    A positive value = net burden; negative = net transfer to that group.
    """
    dist = cal.income_distribution
    mob  = cal.macro.capital_mobility
    gdp  = alloc.gdp

    burden: Dict[str, float] = {g: 0.0 for g in GROUPS}

    # Representative income multiples per group (multiples of median household income)
    group_income_multiples = {
        "Q1": 0.20, "Q2": 0.55, "Q3": 0.90,
        "Q4": 1.40, "Q5_bottom": 2.80, "Q5_top": 8.00,
    }

    # --- Labor income tax burden ---
    # Use GROUP-SPECIFIC effective rates (not the GDP-average), so that
    # progressive brackets produce progressive incidence.
    for g in GROUPS:
        labor_share_g = dist.labor_income_shares[g]
        eff_rate_g = policy.labor_income.effective_rate_for_income_multiple(
            group_income_multiples[g]
        )
        eti_adj = 1.0 - cal.macro.elasticity_of_taxable_income * eff_rate_g * 0.5
        burden[g] += eff_rate_g * labor_share_g * eti_adj

    # --- Payroll tax burden ---
    # FICA is regressive above the ceiling (flat below, declining above).
    # Both employee and employer portions fall on workers in the long run.
    if policy.payroll.combined_rate_below_ceiling > 0:
        for g in GROUPS:
            labor_share_g = dist.labor_income_shares[g]
            payroll_rate_g = policy.payroll.effective_rate_for_income_multiple(
                group_income_multiples[g]
            )
            burden[g] += payroll_rate_g * labor_share_g

    # --- Consumption tax burden ---
    if policy.consumption.rate > 0:
        for g in GROUPS:
            cons_share_g = dist.consumption_shares[g]
            gross_cons_burden = policy.consumption.rate * cons_share_g
            burden[g] += gross_cons_burden

        # Prebate: equal per-capita transfer, so each group gets 1/N * total prebate
        # In fractional-of-income terms, prebate is larger relative to income for low-income groups
        if policy.consumption.prebate_fraction > 0:
            total_prebate_rev = abs(revenue.prebate_cost)
            # Equal per-capita → each group gets share proportional to population (assume equal pop shares)
            pop_share = 1.0 / len(GROUPS)
            for g in GROUPS:
                # Prebate receipt as fraction of group's income share
                prebate_per_group = total_prebate_rev * pop_share
                burden[g] -= prebate_per_group

    # --- Land value tax burden ---
    if policy.land_value.rate > 0:
        for g in GROUPS:
            land_share_g = cal.land.land_ownership_shares[g]
            burden[g] += policy.land_value.rate * cal.land.land_value_gdp_ratio * land_share_g

    # --- Corporate tax burden ---
    # Split between capital and labor using capital_mobility:
    #   mobility=0 (closed): full burden on capital owners
    #   mobility=1 (open):   full burden on workers via lower wages
    if policy.corporate.rate > 0:
        total_corp = revenue.corporate_tax
        capital_share_of_burden = 1.0 - mob
        labor_share_of_burden = mob

        for g in GROUPS:
            cap_burden_g = total_corp * capital_share_of_burden * dist.capital_income_shares[g]
            lab_burden_g = total_corp * labor_share_of_burden * dist.labor_income_shares[g]
            burden[g] += cap_burden_g + lab_burden_g

    # --- Pigouvian tax burden ---
    if policy.pigouvian.rate_per_unit > 0:
        # Gross burden: proportional to carbon-intensive consumption
        total_pig = revenue.pigouvian_tax
        for g in GROUPS:
            carbon_share_g = cal.externality.carbon_consumption_shares[g]
            gross_pig_burden = total_pig * carbon_share_g
            burden[g] += gross_pig_burden

        # Dividend: equal per-capita
        if policy.pigouvian.dividend_recycling_fraction > 0:
            total_dividend = abs(revenue.pigouvian_dividend_cost)
            pop_share = 1.0 / len(GROUPS)
            for g in GROUPS:
                burden[g] -= total_dividend * pop_share

        # Labor tax offset: reduces labor burden proportionally
        if policy.pigouvian.labor_tax_offset_fraction > 0:
            offset = revenue.pigouvian_tax * policy.pigouvian.labor_tax_offset_fraction
            for g in GROUPS:
                burden[g] -= offset * dist.labor_income_shares[g]

    # --- Capital gains tax burden ---
    if policy.capital_gains.rate > 0:
        total_cg = revenue.capital_gains_tax
        for g in GROUPS:
            cg_share_g = dist.capital_gains_shares[g]
            burden[g] += total_cg * cg_share_g

    # --- Normalize: express burden as fraction of group's income share ---
    # Burden[g] is currently in units of (fraction of total GDP).
    # Normalize by income_share[g] to get burden as fraction of group pre-tax income.
    # Income share = labor + capital income shares (weighted)
    total_incidence: Dict[str, float] = {}
    for g in GROUPS:
        income_share_g = (
            0.65 * dist.labor_income_shares[g]   # labor ≈ 65% of income
            + 0.35 * dist.capital_income_shares[g] # capital ≈ 35%
        )
        if income_share_g > 1e-9:
            total_incidence[g] = burden[g] / income_share_g
        else:
            total_incidence[g] = 0.0

    return DistributionalIncidence.from_dict(total_incidence)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _avg_effective_labor_rate(policy: TaxPolicy, cal: Calibration) -> float:
    """
    Compute a GDP-weighted average effective labor income tax rate.
    Uses quintile effective rates if provided; otherwise integrates brackets
    over the income distribution.
    """
    dist = cal.income_distribution

    if policy.labor_income.quintile_effective_rates is not None:
        rates = policy.labor_income.quintile_effective_rates
        shares = list(dist.labor_income_shares.values())
        # weighted average: rate[i] * labor_share[i]
        total = sum(r * s for r, s in zip(rates, shares))
        return total

    # Integrate brackets over income distribution using representative income multiples
    # Each group is represented by its approximate median income multiple
    group_income_multiples = {
        "Q1": 0.20,
        "Q2": 0.55,
        "Q3": 0.90,
        "Q4": 1.40,
        "Q5_bottom": 2.80,
        "Q5_top": 8.00,
    }
    total_rate = 0.0
    for g in GROUPS:
        eff_rate = policy.labor_income.effective_rate_for_income_multiple(
            group_income_multiples[g]
        )
        total_rate += eff_rate * dist.labor_income_shares[g]
    return total_rate


def _avg_effective_payroll_rate(policy: TaxPolicy, cal: Calibration) -> float:
    """
    GDP-weighted average effective payroll (FICA) tax rate across income groups.
    Accounts for the wage ceiling — regressive above the SS wage base.
    """
    dist = cal.income_distribution
    group_income_multiples = {
        "Q1": 0.20, "Q2": 0.55, "Q3": 0.90,
        "Q4": 1.40, "Q5_bottom": 2.80, "Q5_top": 8.00,
    }
    total_rate = 0.0
    for g in GROUPS:
        eff_rate = policy.payroll.effective_rate_for_income_multiple(
            group_income_multiples[g]
        )
        total_rate += eff_rate * dist.labor_income_shares[g]
    return total_rate
