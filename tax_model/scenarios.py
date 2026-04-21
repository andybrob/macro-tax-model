"""
Scenario cards — structural reforms that modify calibration parameters.

These layer on top of a TaxPolicy to capture supply-side and spending effects
that can't be expressed purely as tax wedges:
  - TFPCard:         Human capital investment (Pre-K, MOOCs) → TFP boost + spending cost
  - LaborSupplyCard: High-skill immigration reform (H1B) → effective TFP boost + spillovers

Usage
-----
    from tax_model.scenarios import ScenarioBundle, TFPCard, LaborSupplyCard

    reform_scenario = ScenarioBundle(
        tfp_card=TFPCard(name="Pre-K", spend_gdp_fraction=0.005, long_run_tfp_multiplier=1.01),
        labor_card=LaborSupplyCard(name="H1B+", high_skill_labor_multiplier=1.10),
    )
    reform_result = economy.solve(reform_policy, scenario=reform_scenario)

Design note
-----------
ScenarioBundle.apply_to_calibration() returns a modified *copy* of the calibration.
The original economy baseline (K0, Y0, w0) is unchanged, so GDP normalization is
relative to the no-scenario baseline — reform GDP gains include both the tax reform
effect AND any structural reform effects. This is intentional.
"""

from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import Optional


@dataclass
class TFPCard:
    """
    Human capital / productivity investment card.

    Models programs like universal pre-K or federal college MOOCs.
    These increase TFP (long-run) via higher human capital, while the
    fiscal cost adds to effective government spending.

    name:                    display label
    spend_gdp_fraction:      program cost as fraction of GDP
    long_run_tfp_multiplier: multiplicative boost to TFP level A (e.g. 1.01 = +1%)
    targeting_bottom_half:   fraction of benefits going to Q1+Q2 (for future incidence work)
    """
    name: str = "TFP Investment"
    spend_gdp_fraction: float = 0.0
    long_run_tfp_multiplier: float = 1.0
    targeting_bottom_half: float = 0.5


@dataclass
class LaborSupplyCard:
    """
    High-skill immigration / labor supply card.

    Models H1B expansion and targeted skilled immigration reform.
    In Cobb-Douglas production, higher-skilled labor acts like a TFP boost
    (higher quality-adjusted L). We model this as an effective TFP multiplier:
      effective_boost = (high_skill_labor_multiplier - 1) × 0.5 + tfp_spillover
    where the 0.5 factor captures that only the high-skill increment contributes.

    name:                        display label
    high_skill_labor_multiplier: % increase in high-skill labor supply (e.g. 1.10 = +10%)
    tfp_spillover:               additive TFP boost from knowledge spillovers (e.g. 0.005 = +0.5%)
    wage_compression_fraction:   placeholder for future wage distribution modeling
    """
    name: str = "High-Skill Immigration"
    high_skill_labor_multiplier: float = 1.0
    tfp_spillover: float = 0.0
    wage_compression_fraction: float = 0.0


@dataclass
class ScenarioBundle:
    """
    Bundle of structural scenario cards applied on top of a TaxPolicy.
    All cards default to None (inactive / no effect on calibration).
    """
    tfp_card: Optional[TFPCard] = None
    labor_card: Optional[LaborSupplyCard] = None

    def is_empty(self) -> bool:
        return self.tfp_card is None and self.labor_card is None

    def apply_to_calibration(self, cal):
        """
        Return a modified deep copy of `cal` with all active scenario cards applied.
        Does NOT mutate the original calibration object.
        """
        if self.is_empty():
            return cal

        new_cal = copy.deepcopy(cal)

        if self.tfp_card is not None:
            t = self.tfp_card
            new_cal.production.tfp_level *= t.long_run_tfp_multiplier
            new_cal.government.spending_gdp_ratio += t.spend_gdp_fraction

        if self.labor_card is not None:
            lc = self.labor_card
            # High-skill labor increase: in Cobb-Douglas, +X% skilled labor ≈ +(X*0.5)% effective TFP
            # (labor's production share × fraction that's high-skill premium)
            effective_tfp_boost = (lc.high_skill_labor_multiplier - 1.0) * 0.5 + lc.tfp_spillover
            new_cal.production.tfp_level *= (1.0 + effective_tfp_boost)

        return new_cal
