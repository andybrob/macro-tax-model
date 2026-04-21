"""
TaxPolicy and sub-policy dataclasses.

Design principles
-----------------
- Every field has a sensible "no tax" default so TaxPolicy() represents the
  zero-tax baseline and partial policies compose cleanly.
- No computation happens here — this is pure declarative data.
- Add a new tax instrument by adding a new dataclass + field on TaxPolicy.
  The core solver only needs to know about taxes that shift equilibrium
  conditions (labor supply, saving rate, cost of capital). A new instrument
  that only affects revenue accounting can be added to government.py without
  touching economy.py.

Bracket notation
----------------
Labor income tax brackets are specified as (threshold, marginal_rate) pairs
where threshold is expressed as a *multiple of median household income*
(calibration-independent). Example:
    [(0.0, 0.10), (0.5, 0.22), (1.5, 0.32), (4.0, 0.37)]
means 10% on income up to 0.5× median, 22% from 0.5–1.5×, etc.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class LaborIncomeTax:
    """
    Progressive labor/wage income tax.

    Specify either brackets OR quintile_effective_rates. If both are given,
    brackets take precedence and effective rates are derived.

    brackets: list of (income_threshold_as_multiple_of_median, marginal_rate)
    quintile_effective_rates: [Q1_rate, Q2_rate, Q3_rate, Q4_rate, Q5_bottom_rate, Q5_top_rate]
    standard_deduction_median_multiple: standard deduction as multiple of median income
    """
    brackets: List[Tuple[float, float]] = field(
        default_factory=lambda: [(0.0, 0.0)]
    )
    quintile_effective_rates: Optional[List[float]] = None
    standard_deduction_median_multiple: float = 0.20

    def effective_rate_for_income_multiple(self, income_multiple: float) -> float:
        """Average effective rate at a given multiple of median income."""
        if not self.brackets:
            return 0.0
        total_tax = 0.0
        prev_threshold = 0.0
        prev_rate = 0.0
        sorted_brackets = sorted(self.brackets, key=lambda x: x[0])
        for threshold, rate in sorted_brackets:
            if income_multiple <= threshold:
                taxable = max(0.0, income_multiple - prev_threshold
                              - self.standard_deduction_median_multiple)
                total_tax += taxable * prev_rate
                break
            taxable = max(0.0, threshold - prev_threshold
                          - self.standard_deduction_median_multiple)
            total_tax += max(0.0, taxable) * prev_rate
            prev_threshold = threshold
            prev_rate = rate
        else:
            taxable = max(0.0, income_multiple - prev_threshold
                          - self.standard_deduction_median_multiple)
            total_tax += taxable * prev_rate
        return total_tax / income_multiple if income_multiple > 0 else 0.0

    def marginal_rate_at(self, income_multiple: float) -> float:
        """Marginal rate at a given income multiple."""
        sorted_brackets = sorted(self.brackets, key=lambda x: x[0])
        rate = 0.0
        for threshold, br_rate in sorted_brackets:
            if income_multiple >= threshold:
                rate = br_rate
        return rate


@dataclass
class ConsumptionTax:
    """
    Flat consumption tax (VAT or retail sales tax equivalent) with optional prebate.

    rate: fraction of consumption expenditure
    prebate_fraction: fraction of poverty-line consumption returned as equal
                      per-capita lump-sum transfer. 0 = no prebate, 1 = full
                      poverty-line prebate (makes effective rate negative for Q1/Q2).
    """
    rate: float = 0.0
    prebate_fraction: float = 0.0


@dataclass
class LandValueTax:
    """
    Annual tax on the unimproved value of land (pure economic rent).
    Because land supply is perfectly inelastic, this has zero deadweight loss.

    rate: fraction of assessed land value per year
    """
    rate: float = 0.0


@dataclass
class CorporateTax:
    """
    Corporate / business cash flow tax.

    rate: statutory corporate tax rate
    immediate_expensing: True = 100% immediate deduction of capital investment
                         (Hall-Jorgenson; drives effective marginal rate on new
                         investment toward zero). False = depreciation schedules.
    interest_deductibility: fraction of interest payments that are deductible.
                            1.0 = current US law; 0.0 = DBCFT / equity-neutral.
    border_adjustment: True = destination-based (DBCFT); taxes consumption,
                       exempts exports.
    """
    rate: float = 0.0
    immediate_expensing: bool = False
    interest_deductibility: float = 1.0
    border_adjustment: bool = False


@dataclass
class PigouvianTax:
    """
    Tax on a negative externality (e.g. carbon emissions) with revenue recycling.

    rate_per_unit: tax per unit of externality (e.g. $/ton CO2)
    dividend_recycling_fraction: share of revenue returned as equal per-capita
                                 cash dividend (fee-and-dividend).
    labor_tax_offset_fraction: share of revenue used to reduce labor taxes.
    The remainder (1 - dividend - labor_offset) goes to general revenue.
    """
    rate_per_unit: float = 0.0
    dividend_recycling_fraction: float = 0.0
    labor_tax_offset_fraction: float = 0.0

    def __post_init__(self) -> None:
        total = self.dividend_recycling_fraction + self.labor_tax_offset_fraction
        if total > 1.0 + 1e-9:
            raise ValueError(
                f"Recycling fractions sum to {total:.3f}, must be ≤ 1.0"
            )


@dataclass
class CapitalGainsTax:
    """
    Tax on capital gains.

    rate: statutory rate applied to nominal (or real, if indexed) gains
    inflation_indexed: if True, only real (inflation-adjusted) gains are taxed,
                       eliminating the taxation of phantom gains.
    inflation_rate: assumed annual inflation rate for indexing (default ~2%)
    lock_in_discount: fraction by which effective rate is reduced relative to
                      statutory rate due to realization-based deferral (lock-in
                      effect). 0.3 is a reasonable estimate for US law.
    """
    rate: float = 0.0
    inflation_indexed: bool = False
    inflation_rate: float = 0.025
    lock_in_discount: float = 0.30

    @property
    def effective_rate(self) -> float:
        """Effective rate after lock-in discount."""
        return self.rate * (1.0 - self.lock_in_discount)


@dataclass
class PayrollTax:
    """
    Payroll / social insurance tax (US: FICA).

    employee_rate: employee-side rate (withheld from wages)
    employer_rate: employer-side rate (implicit labor cost; treated as a wage
                   tax in incidence since it reduces the employer's willingness
                   to pay — economic burden falls on workers in the long run)
    wage_ceiling_median_multiple: cap on taxable wages as a multiple of median
                                  income.  Above this ceiling, rate drops to
                                  medicare_rate_above_ceiling.
                                  US 2024: SS ceiling ≈ $168k ≈ 2.8× median.
    medicare_rate_above_ceiling: Medicare portion continues above the SS ceiling.
    """
    employee_rate: float = 0.0
    employer_rate: float = 0.0
    wage_ceiling_median_multiple: float = 2.8
    medicare_rate_above_ceiling: float = 0.0

    @property
    def combined_rate_below_ceiling(self) -> float:
        return self.employee_rate + self.employer_rate

    def effective_rate_for_income_multiple(self, income_multiple: float) -> float:
        """Average effective payroll rate at a given income multiple."""
        if income_multiple <= 0:
            return 0.0
        ceiling = self.wage_ceiling_median_multiple
        if income_multiple <= ceiling:
            return self.combined_rate_below_ceiling
        # Above ceiling: SS portion capped, Medicare continues on all wages
        ss_rate = self.combined_rate_below_ceiling - self.medicare_rate_above_ceiling
        tax = ss_rate * ceiling + self.medicare_rate_above_ceiling * income_multiple
        return tax / income_multiple


@dataclass
class TaxPolicy:
    """
    A complete tax policy configuration.

    All sub-policies default to their zero-tax state, so TaxPolicy() represents
    the zero-tax counterfactual. Current US law can be approximated by supplying
    the appropriate effective rates to each sub-policy.

    revenue_neutral: if True, the solver will find the lump-sum transfer (or
                     labor tax offset) that holds total government revenue equal
                     to the baseline. Set to False when modeling deficit-financed
                     changes.
    label: human-readable name for this policy (used in charts and tables).
    """
    label: str = "unnamed"
    labor_income: LaborIncomeTax = field(default_factory=LaborIncomeTax)
    payroll: PayrollTax = field(default_factory=PayrollTax)
    consumption: ConsumptionTax = field(default_factory=ConsumptionTax)
    land_value: LandValueTax = field(default_factory=LandValueTax)
    corporate: CorporateTax = field(default_factory=CorporateTax)
    pigouvian: PigouvianTax = field(default_factory=PigouvianTax)
    capital_gains: CapitalGainsTax = field(default_factory=CapitalGainsTax)
    revenue_neutral: bool = False

    def to_dict(self) -> dict:
        """Serialize to a plain dict (for logging, YAML export, etc.)."""
        import dataclasses
        return dataclasses.asdict(self)
