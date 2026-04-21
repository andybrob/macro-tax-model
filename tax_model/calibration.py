"""
Calibration
-----------
Holds all parameters that describe the real economy — independent of any policy.
Loaded from a YAML file (see calibration/us_2024.yaml for documented sources).

The Calibration object is the single source of truth for numerical values.
Never hardcode numbers elsewhere; reference cal.<field> instead.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict
import yaml


GROUPS = ("Q1", "Q2", "Q3", "Q4", "Q5_bottom", "Q5_top")


def _load_group_dict(raw: dict) -> Dict[str, float]:
    """Validate that a per-group dict contains all expected groups."""
    missing = [g for g in GROUPS if g not in raw]
    if missing:
        raise ValueError(f"Calibration missing groups: {missing}")
    return {g: float(raw[g]) for g in GROUPS}


@dataclass
class ProductionParams:
    capital_share: float       # alpha in Y = A * K^alpha * L^(1-alpha)
    depreciation_rate: float   # delta
    tfp_level: float           # A


@dataclass
class HouseholdParams:
    discount_rate: float           # rho (annual)
    ies: float                     # intertemporal elasticity of substitution
    frisch_elasticity: float       # Frisch elasticity of labor supply
    saving_rate_sensitivity: float # d(savings_rate)/d(after_tax_return)


@dataclass
class IncomeDistribution:
    labor_income_shares: Dict[str, float]
    capital_income_shares: Dict[str, float]
    capital_gains_shares: Dict[str, float]
    consumption_shares: Dict[str, float]
    savings_rates: Dict[str, float]


@dataclass
class GovernmentParams:
    spending_gdp_ratio: float
    baseline_transfers_gdp_ratio: float
    poverty_line_consumption_share: float


@dataclass
class ExternalityParams:
    carbon_intensity_per_gdp: float
    social_cost_carbon: float
    carbon_consumption_shares: Dict[str, float]


@dataclass
class LandParams:
    land_value_gdp_ratio: float
    land_rent_gdp_ratio: float
    land_share_of_capital: float
    land_ownership_shares: Dict[str, float]


@dataclass
class MacroParams:
    capital_mobility: float            # 0=closed, 1=small open economy
    elasticity_of_taxable_income: float
    capital_income_eti: float
    target_capital_output_ratio: float
    target_labor_share: float


@dataclass
class Calibration:
    """
    All parameters that characterize the US economy (not any specific policy).
    Loaded from YAML; never modified at runtime.
    """
    production: ProductionParams
    households: HouseholdParams
    income_distribution: IncomeDistribution
    government: GovernmentParams
    externality: ExternalityParams
    land: LandParams
    macro: MacroParams

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Calibration":
        with open(path) as f:
            raw = yaml.safe_load(f)

        prod = raw["production"]
        hh   = raw["households"]
        inc  = raw["income_distribution"]
        gov  = raw["government"]
        ext  = raw["externality"]
        lnd  = raw["land"]
        mac  = raw["macroeconomic"]

        return cls(
            production=ProductionParams(
                capital_share=float(prod["capital_share"]),
                depreciation_rate=float(prod["depreciation_rate"]),
                tfp_level=float(prod["tfp_level"]),
            ),
            households=HouseholdParams(
                discount_rate=float(hh["discount_rate"]),
                ies=float(hh["ies"]),
                frisch_elasticity=float(hh["frisch_elasticity"]),
                saving_rate_sensitivity=float(hh["saving_rate_sensitivity"]),
            ),
            income_distribution=IncomeDistribution(
                labor_income_shares=_load_group_dict(inc["labor_income_shares"]),
                capital_income_shares=_load_group_dict(inc["capital_income_shares"]),
                capital_gains_shares=_load_group_dict(inc["capital_gains_shares"]),
                consumption_shares=_load_group_dict(inc["consumption_shares"]),
                savings_rates=_load_group_dict(inc["savings_rates"]),
            ),
            government=GovernmentParams(
                spending_gdp_ratio=float(gov["spending_gdp_ratio"]),
                baseline_transfers_gdp_ratio=float(gov["baseline_transfers_gdp_ratio"]),
                poverty_line_consumption_share=float(gov["poverty_line_consumption_share"]),
            ),
            externality=ExternalityParams(
                carbon_intensity_per_gdp=float(ext["carbon_intensity_per_gdp"]),
                social_cost_carbon=float(ext["social_cost_carbon"]),
                carbon_consumption_shares=_load_group_dict(ext["carbon_consumption_shares"]),
            ),
            land=LandParams(
                land_value_gdp_ratio=float(lnd["land_value_gdp_ratio"]),
                land_rent_gdp_ratio=float(lnd["land_rent_gdp_ratio"]),
                land_share_of_capital=float(lnd["land_share_of_capital"]),
                land_ownership_shares=_load_group_dict(lnd["land_ownership_shares"]),
            ),
            macro=MacroParams(
                capital_mobility=float(mac["capital_mobility"]),
                elasticity_of_taxable_income=float(mac["elasticity_of_taxable_income"]),
                capital_income_eti=float(mac["capital_income_eti"]),
                target_capital_output_ratio=float(mac["target_capital_output_ratio"]),
                target_labor_share=float(mac["target_labor_share"]),
            ),
        )
