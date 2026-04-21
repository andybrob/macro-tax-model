"""
Named policy presets.

Each preset is a TaxPolicy that approximates a well-known tax reform proposal
or current law. They are the starting points for analysis — adjust sliders
from here, or compare directly against each other.

Usage
-----
    from tax_model.presets import PRESETS, current_law

    economy.solve(PRESETS["Current Law"])
    economy.solve(PRESETS["X-Tax (Bradford)"])

Revenue calibration note
------------------------
All-in effective rates for "Current Law" are calibrated to match US 2023
federal + state/local aggregate revenue (~30% of GDP):
  - Federal income tax:  ~8.5% GDP
  - Payroll (FICA):      ~6.0% GDP
  - Corporate:           ~1.7% GDP
  - State/local income:  ~3.5% GDP (captured via higher bracket rates)
  - Other (excise, estate, property): ~5% GDP (approximate via consumption rate)

The bracket rates include a state/local surcharge (~5pp) on federal rates to
capture the combined all-in effective rates that drive actual behavioral wedges.
"""

from .policy import (
    TaxPolicy,
    LaborIncomeTax,
    PayrollTax,
    ConsumptionTax,
    LandValueTax,
    CorporateTax,
    PigouvianTax,
    CapitalGainsTax,
)

# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

# Federal income tax brackets (2024) + ~5pp state/local surcharge on each rate
# Expressed as multiples of ~$80k median household income
# Thresholds: 0.5× ($40k), 1.5× ($120k), 3× ($240k), 7× ($560k)
_CURRENT_LAW_INCOME_BRACKETS = [
    (0.0, 0.15),   # 10% federal + 5% state/local — most Q1, Q2
    (0.5, 0.27),   # 22% + 5%
    (1.5, 0.29),   # 24% + 5%
    (3.0, 0.37),   # 32% + 5%
    (7.0, 0.42),   # 37% + 5%
]

# FICA calibration note:
# Statutory rate = 15.3% (7.65% each side), but effective rate on *total labor income*
# is lower (~9%) because:
#   1. SS wage ceiling caps high earners (only Medicare 2.9% above $168k)
#   2. ~35% of labor income is excluded: S-corp distributions, partnership income,
#      non-covered employment, self-employment partial exclusions
# Target: ~6% GDP payroll revenue  →  effective avg rate ≈ 9% × 65% labor share ≈ 5.9%
# We calibrate by setting rates to the effective level after coverage adjustment.
_CURRENT_LAW_PAYROLL = PayrollTax(
    employee_rate=0.045,    # ~4.5% effective employee rate (7.65% × coverage)
    employer_rate=0.045,    # ~4.5% effective employer rate
    wage_ceiling_median_multiple=2.8,
    medicare_rate_above_ceiling=0.015,  # effective Medicare rate above ceiling
)

_CURRENT_LAW_CORPORATE = CorporateTax(rate=0.21, immediate_expensing=False)
_CURRENT_LAW_CG = CapitalGainsTax(rate=0.238, lock_in_discount=0.30)  # 20% + 3.8% NIIT


# ---------------------------------------------------------------------------
# Preset definitions
# ---------------------------------------------------------------------------

def current_law() -> TaxPolicy:
    """
    Approximate current US law (federal + state/local, 2024).
    All-in effective rates calibrated to ~30% GDP total revenue.
    """
    return TaxPolicy(
        label="Current Law",
        labor_income=LaborIncomeTax(
            brackets=_CURRENT_LAW_INCOME_BRACKETS,
            standard_deduction_median_multiple=0.20,
        ),
        payroll=_CURRENT_LAW_PAYROLL,
        corporate=_CURRENT_LAW_CORPORATE,
        capital_gains=_CURRENT_LAW_CG,
        consumption=ConsumptionTax(rate=0.02),  # excise + misc consumption taxes
    )


def x_tax() -> TaxPolicy:
    """
    Bradford X-Tax (consumed-income tax).

    Businesses taxed on cash flow (sales − wages − purchases); individuals
    taxed only on wages with a progressive schedule. Capital income exempt
    (immediate expensing eliminates the tax on normal returns).

    The wage tax schedule here is more progressive than current law at the bottom
    (lower rates for Q1/Q2 because capital income exclusion benefits high earners
    less, and the standard deduction is raised significantly).
    """
    return TaxPolicy(
        label="X-Tax (Bradford)",
        labor_income=LaborIncomeTax(
            brackets=[
                (0.0, 0.00),   # exempt — large standard deduction covers Q1
                (0.5, 0.15),
                (1.5, 0.25),
                (3.0, 0.35),
                (7.0, 0.40),
            ],
            standard_deduction_median_multiple=0.50,  # doubled vs current law
        ),
        payroll=PayrollTax(),  # payroll tax eliminated (absorbed into wage tax)
        corporate=CorporateTax(rate=0.25, immediate_expensing=True),
        capital_gains=CapitalGainsTax(rate=0.0),  # no capital income tax at individual level
    )


def vat_plus_prebate() -> TaxPolicy:
    """
    VAT + universal prebate (Roberts / FairTax style).

    A broad-based value-added tax replaces federal income and payroll taxes.
    A monthly per-capita prebate equal to the VAT on poverty-line consumption
    makes the effective rate zero at the poverty line and negative below it.
    """
    return TaxPolicy(
        label="VAT + Prebate",
        labor_income=LaborIncomeTax(brackets=[(0.0, 0.0)]),  # income tax eliminated
        payroll=PayrollTax(),  # payroll tax eliminated
        consumption=ConsumptionTax(rate=0.23, prebate_fraction=1.0),
        corporate=CorporateTax(rate=0.0),   # embedded in VAT
        capital_gains=CapitalGainsTax(rate=0.0),
    )


def lvt_swap() -> TaxPolicy:
    """
    Land Value Tax swap: replace corporate income tax with an LVT.

    LVT has zero deadweight loss (land supply is perfectly inelastic).
    Burden falls on land owners (concentrated at Q5_top).
    Corporate tax eliminated → improves capital formation.
    """
    return TaxPolicy(
        label="LVT Swap (replaces corp tax)",
        labor_income=LaborIncomeTax(brackets=_CURRENT_LAW_INCOME_BRACKETS),
        payroll=_CURRENT_LAW_PAYROLL,
        corporate=CorporateTax(rate=0.0),
        land_value=LandValueTax(rate=0.035),  # ~revenue-neutral at land_value/GDP = 0.80
        capital_gains=_CURRENT_LAW_CG,
        consumption=ConsumptionTax(rate=0.02),
    )


def carbon_fee_dividend() -> TaxPolicy:
    """
    Carbon fee + 100% dividend recycling (Citizens' Climate Lobby style).

    $100/ton CO₂ price, all revenue returned as equal per-capita monthly dividend.
    Strongly progressive: dividend exceeds carbon cost for Q1/Q2.
    Existing tax structure unchanged.
    """
    return TaxPolicy(
        label="Carbon Fee + Dividend ($100/ton)",
        labor_income=LaborIncomeTax(brackets=_CURRENT_LAW_INCOME_BRACKETS),
        payroll=_CURRENT_LAW_PAYROLL,
        corporate=_CURRENT_LAW_CORPORATE,
        capital_gains=_CURRENT_LAW_CG,
        consumption=ConsumptionTax(rate=0.02),
        pigouvian=PigouvianTax(rate_per_unit=100.0, dividend_recycling_fraction=1.0),
    )


def dbcft() -> TaxPolicy:
    """
    Destination-Based Cash Flow Tax (DBCFT / House Blueprint 2016 style).

    Immediate expensing of all capital investment + border adjustment.
    Effectively a consumption tax on domestic sales.
    EMTR on new investment → 0; revenue from inframarginal returns.
    """
    return TaxPolicy(
        label="DBCFT",
        labor_income=LaborIncomeTax(brackets=_CURRENT_LAW_INCOME_BRACKETS),
        payroll=_CURRENT_LAW_PAYROLL,
        corporate=CorporateTax(
            rate=0.20,
            immediate_expensing=True,
            interest_deductibility=0.0,
            border_adjustment=True,
        ),
        capital_gains=CapitalGainsTax(rate=0.0),  # business income taxed at entity level
        consumption=ConsumptionTax(rate=0.02),
    )


def capital_gains_reform() -> TaxPolicy:
    """
    Capital gains taxed at ordinary rates, inflation-indexed.

    Eliminates the preferential rate for capital gains (currently 20% vs 37%
    top ordinary rate), but indexes for inflation so only real gains are taxed.
    Reduces the taxation of phantom gains; keeps incentive for long-term investment.
    """
    return TaxPolicy(
        label="CG Reform (ordinary rates, indexed)",
        labor_income=LaborIncomeTax(brackets=_CURRENT_LAW_INCOME_BRACKETS),
        payroll=_CURRENT_LAW_PAYROLL,
        corporate=_CURRENT_LAW_CORPORATE,
        capital_gains=CapitalGainsTax(
            rate=0.42,           # matches top ordinary rate (37% federal + 5% state)
            inflation_indexed=True,
            lock_in_discount=0.35,  # higher lock-in at higher rate
        ),
        consumption=ConsumptionTax(rate=0.02),
    )


def progressive_hybrid() -> TaxPolicy:
    """
    Progressive hybrid: LVT + carbon dividend + reduced income tax.

    Combines three efficiency-improving reforms:
    1. LVT partially replaces corporate tax
    2. Carbon fee with 50% dividend / 50% labor tax offset
    3. Reduced income tax rates (fiscal space from LVT + carbon)

    This is a "kitchen sink" growth-and-equity scenario.
    """
    return TaxPolicy(
        label="Progressive Hybrid",
        labor_income=LaborIncomeTax(
            brackets=[
                (0.0, 0.00),
                (0.5, 0.18),
                (1.5, 0.24),
                (3.0, 0.30),
                (7.0, 0.35),
            ],
            standard_deduction_median_multiple=0.35,
        ),
        payroll=_CURRENT_LAW_PAYROLL,
        corporate=CorporateTax(rate=0.15, immediate_expensing=False),
        land_value=LandValueTax(rate=0.02),
        capital_gains=_CURRENT_LAW_CG,
        consumption=ConsumptionTax(rate=0.02),
        pigouvian=PigouvianTax(
            rate_per_unit=75.0,
            dividend_recycling_fraction=0.50,
            labor_tax_offset_fraction=0.50,
        ),
    )


# ---------------------------------------------------------------------------
# Registry — used by the Streamlit GUI and notebook
# ---------------------------------------------------------------------------

PRESETS: dict = {
    "Current Law": current_law,
    "X-Tax (Bradford)": x_tax,
    "VAT + Prebate": vat_plus_prebate,
    "LVT Swap": lvt_swap,
    "Carbon Fee + Dividend": carbon_fee_dividend,
    "DBCFT": dbcft,
    "CG Reform (ordinary rates, indexed)": capital_gains_reform,
    "Progressive Hybrid": progressive_hybrid,
}
