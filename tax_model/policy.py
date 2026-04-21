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
    Consumption tax (VAT or retail sales tax) with optional prebate and tiered rates.

    rate: standard rate (fraction of consumption expenditure)
    prebate_fraction: fraction of poverty-line consumption returned as equal
                      per-capita lump-sum transfer.
    essentials_rate: reduced rate for food, medicine, basic housing. None = use rate.
    luxury_rate: elevated rate for luxury goods. None = use rate.
    """
    rate: float = 0.0
    prebate_fraction: float = 0.0
    essentials_rate: Optional[float] = None
    luxury_rate: Optional[float] = None

    def effective_rate_for_group(self, group: str) -> float:
        """
        Income-group-specific effective consumption tax rate accounting for tiered rates.
        Spending shares from CEX data (essentials / standard / luxury by income quintile).
        """
        # CEX-based spending shares: (essentials_share, standard_share, luxury_share)
        _SPENDING_SHARES = {
            "Q1":        (0.55, 0.42, 0.03),
            "Q2":        (0.48, 0.46, 0.06),
            "Q3":        (0.38, 0.52, 0.10),
            "Q4":        (0.28, 0.55, 0.17),
            "Q5_bottom": (0.22, 0.53, 0.25),
            "Q5_top":    (0.15, 0.50, 0.35),
        }
        ess_r = self.essentials_rate if self.essentials_rate is not None else self.rate
        lux_r = self.luxury_rate if self.luxury_rate is not None else self.rate
        std_r = self.rate
        s_ess, s_std, s_lux = _SPENDING_SHARES.get(group, (0.30, 0.55, 0.15))
        return s_ess * ess_r + s_std * std_r + s_lux * lux_r


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
    stepped_up_basis_removal_fraction: fraction of stepped-up basis benefit removed.
                      0 = current law (heirs inherit cost basis = FMV at death).
                      1 = full removal (heirs inherit decedent's original cost basis).
                      Removing step-up reduces lock-in: gains can no longer be
                      permanently avoided by holding until death.
    """
    rate: float = 0.0
    inflation_indexed: bool = False
    inflation_rate: float = 0.025
    lock_in_discount: float = 0.30
    stepped_up_basis_removal_fraction: float = 0.0

    @property
    def effective_rate(self) -> float:
        """Effective rate after lock-in discount, adjusted for stepped-up basis removal."""
        # Removing stepped-up basis reduces lock-in (gains taxed at death, not permanently deferred)
        # Full removal reduces lock-in by ~60% (empirical estimate of step-up's share of lock-in)
        adjusted_lock_in = self.lock_in_discount * (1.0 - 0.60 * self.stepped_up_basis_removal_fraction)
        return self.rate * (1.0 - adjusted_lock_in)


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
    ss_donut_top_multiple: if > wage_ceiling, SS tax resumes above this multiple
                           of median income, creating a "donut hole" (no SS between
                           ceiling and donut_top). Models proposals to extend SS
                           solvency by taxing very high earners (e.g. $400k+).
                           0.0 = disabled (standard current law structure).
    benefit_cap_median_multiple: if > 0, SS retirement benefits are capped for
                                 earners above this multiple of median income
                                 (means-testing). Above the cap, SS contributions
                                 are a net tax (no insurance value). This increases
                                 the effective labor wedge for high earners who
                                 receive no marginal benefit. 0.0 = disabled.
    """
    employee_rate: float = 0.0
    employer_rate: float = 0.0
    wage_ceiling_median_multiple: float = 2.8
    medicare_rate_above_ceiling: float = 0.0
    ss_donut_top_multiple: float = 0.0       # 0 = disabled; >ceiling = donut hole active
    benefit_cap_median_multiple: float = 0.0  # 0 = disabled; >0 = benefit cap active

    @property
    def combined_rate_below_ceiling(self) -> float:
        return self.employee_rate + self.employer_rate

    def effective_rate_for_income_multiple(self, income_multiple: float) -> float:
        """
        Average effective payroll rate at a given income multiple of median income.

        Handles three regimes:
          Standard (donut disabled): below ceiling = full rate; above = Medicare only.
          Donut hole (donut_top > ceiling): SS pauses between ceiling and donut_top,
            then resumes above donut_top (taxing very high earners on the excess).
          Benefit cap: if income is above benefit_cap, the SS contribution yields
            no marginal retirement benefit — modeled as an additional 50% wedge on
            the SS portion above the cap (conservative; empirical range 30–70%).
        """
        if income_multiple <= 0:
            return 0.0

        ceiling = self.wage_ceiling_median_multiple
        ss_rate = self.combined_rate_below_ceiling - self.medicare_rate_above_ceiling
        medicare = self.medicare_rate_above_ceiling
        donut_top = self.ss_donut_top_multiple
        donut_active = donut_top > ceiling + 1e-9

        if income_multiple <= ceiling:
            # Below SS wage ceiling: full combined rate
            effective = self.combined_rate_below_ceiling
        elif donut_active and income_multiple <= donut_top:
            # In the donut hole: SS capped at ceiling, only Medicare continues
            tax = ss_rate * ceiling + medicare * income_multiple
            effective = tax / income_multiple
        elif donut_active and income_multiple > donut_top:
            # Above donut top: SS resumes on income above donut_top + Medicare on all
            # (SS cap applies only up to ceiling; the high-end SS is on the *excess*)
            tax = ss_rate * ceiling + medicare * income_multiple + ss_rate * (income_multiple - donut_top)
            effective = tax / income_multiple
        else:
            # Standard law above ceiling: SS capped, Medicare continues
            tax = ss_rate * ceiling + medicare * income_multiple
            effective = tax / income_multiple

        # Benefit cap adjustment: SS above the cap is a net tax (no insurance value).
        # We add 50% of the SS rate on the capped portion as an additional wedge.
        if self.benefit_cap_median_multiple > 0 and income_multiple > self.benefit_cap_median_multiple:
            cap_m = self.benefit_cap_median_multiple
            # SS paid above the cap (on the excess, up to ceiling) with 50% benefit-tax wedge
            ss_above_cap = ss_rate * max(0.0, min(ceiling, income_multiple) - cap_m)
            effective += 0.50 * ss_above_cap / income_multiple

        return float(effective)


@dataclass
class EstateTax:
    """
    Estate / wealth transfer tax.

    Models federal estate tax on large inheritances.
    Revenue is approximated as a levy on Q5_top capital holdings above the exemption.

    rate: top marginal rate (e.g. 0.40 = 40%)
    exemption_median_multiple: exemption threshold as a multiple of median income.
                               US 2024 exemption ~$13.6M ≈ 170× median.
    enforcement_fraction: fraction of theoretical revenue actually collected
                          after legal avoidance (trusts, GRATs, valuation discounts).
                          Current law ≈ 0.30; better enforcement could reach 0.60+.
    """
    rate: float = 0.0
    exemption_median_multiple: float = 170.0
    enforcement_fraction: float = 0.30


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
    estate: EstateTax = field(default_factory=EstateTax)
    revenue_neutral: bool = False

    def to_dict(self) -> dict:
        """Serialize to a plain dict (for logging, YAML export, etc.)."""
        import dataclasses
        return dataclasses.asdict(self)
