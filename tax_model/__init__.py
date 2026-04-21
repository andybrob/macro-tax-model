"""
macro-tax-model
===============
Simplified two-period OLG macroeconomic model for US tax policy evaluation.

Quick start
-----------
    from tax_model import Calibration, TaxPolicy, Economy, PolicyComparison

    cal      = Calibration.from_yaml("calibration/us_2024.yaml")
    economy  = Economy(cal)
    baseline = economy.solve(TaxPolicy(label="current_law"))
    reform   = economy.solve(TaxPolicy(
        label="consumption_tax",
        consumption=ConsumptionTax(rate=0.15, prebate_fraction=0.8),
    ))
    comp = PolicyComparison(baseline=baseline, reform=reform)
    comp.summary()
"""

from .calibration import Calibration
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
from .results import ModelResult, PolicyComparison, DistributionalIncidence, RevenueBreakdown
from .economy import Economy

__all__ = [
    "Calibration",
    "TaxPolicy",
    "LaborIncomeTax",
    "PayrollTax",
    "ConsumptionTax",
    "LandValueTax",
    "CorporateTax",
    "PigouvianTax",
    "CapitalGainsTax",
    "ModelResult",
    "PolicyComparison",
    "DistributionalIncidence",
    "RevenueBreakdown",
    "Economy",
]
