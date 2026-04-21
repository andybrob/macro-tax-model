"""
ModelResult and PolicyComparison output interfaces.

These are pure data containers — no computation happens here.
The Economy produces ModelResults; PolicyComparison wraps two of them
and computes the differences.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from .calibration import GROUPS


@dataclass
class DistributionalIncidence:
    """
    Tax burden (or burden change) by income group, expressed as a fraction of
    that group's pre-tax income. Positive = net burden; negative = net transfer.

    Groups: Q1 (bottom 20%), Q2, Q3, Q4, Q5_bottom (80-99%), Q5_top (top 1%).
    """
    values: Dict[str, float] = field(default_factory=lambda: {g: 0.0 for g in GROUPS})

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "DistributionalIncidence":
        return cls(values={g: d.get(g, 0.0) for g in GROUPS})

    def __getitem__(self, group: str) -> float:
        return self.values[group]

    def __sub__(self, other: "DistributionalIncidence") -> "DistributionalIncidence":
        return DistributionalIncidence(
            values={g: self.values[g] - other.values[g] for g in GROUPS}
        )

    def as_list(self) -> List[float]:
        return [self.values[g] for g in GROUPS]

    def as_pct_list(self) -> List[float]:
        return [v * 100 for v in self.as_list()]


@dataclass
class RevenueBreakdown:
    """
    Tax revenue by instrument, as a fraction of baseline GDP.
    Positive = inflow to government; negative = net transfer/rebate outflow.
    """
    labor_income_tax: float = 0.0
    consumption_tax: float = 0.0
    land_value_tax: float = 0.0
    corporate_tax: float = 0.0
    pigouvian_tax: float = 0.0
    capital_gains_tax: float = 0.0
    prebate_cost: float = 0.0          # negative (outflow)
    pigouvian_dividend_cost: float = 0.0   # negative (outflow)

    @property
    def total(self) -> float:
        return (
            self.labor_income_tax
            + self.consumption_tax
            + self.land_value_tax
            + self.corporate_tax
            + self.pigouvian_tax
            + self.capital_gains_tax
            + self.prebate_cost
            + self.pigouvian_dividend_cost
        )

    def to_dict(self) -> Dict[str, float]:
        import dataclasses
        return dataclasses.asdict(self)


@dataclass
class ModelResult:
    """
    Full output of Economy.solve() for a given TaxPolicy.

    Macro aggregates are in units consistent with the calibration (GDP = 1.0
    at baseline). Factor prices are real (pre-tax). The budget_balance is
    positive when the government runs a surplus.
    """
    # --- Policy that produced this result ---
    policy_label: str = "unnamed"

    # --- Macro aggregates (GDP normalized to 1.0 at baseline) ---
    gdp: float = 0.0
    capital_stock: float = 0.0
    labor_supply: float = 0.0         # aggregate effective labor units
    consumption: float = 0.0
    investment: float = 0.0

    # --- Factor prices (real, pre-tax) ---
    wage: float = 0.0                 # real wage per unit of labor
    return_on_capital: float = 0.0   # real return on capital (net of depreciation)

    # --- Government ---
    revenue: RevenueBreakdown = field(default_factory=RevenueBreakdown)
    government_spending: float = 0.0
    budget_balance: float = 0.0       # revenue.total - spending (positive = surplus)

    # --- Distributional (burden as fraction of pre-tax group income) ---
    incidence: DistributionalIncidence = field(default_factory=DistributionalIncidence)

    # --- Welfare (consumption-equivalent, optional) ---
    welfare_by_group: Optional[Dict[str, float]] = None

    # --- Externality ---
    externality_quantity: Optional[float] = None  # e.g. tons CO2 / GDP
    externality_damage: Optional[float] = None    # monetized $/GDP

    def summary(self) -> str:
        lines = [
            f"Policy: {self.policy_label}",
            f"  GDP:                {self.gdp:.4f}  (baseline = 1.0)",
            f"  Capital stock:      {self.capital_stock:.4f}",
            f"  Labor supply:       {self.labor_supply:.4f}",
            f"  Wage (pre-tax):     {self.wage:.4f}",
            f"  Return on K:        {self.return_on_capital:.4f}",
            f"  Budget balance:     {self.budget_balance:+.4f} (as % GDP)",
            f"  Total revenue:      {self.revenue.total:.4f}",
            "",
            "  Tax burden by group (% of pre-tax income):",
        ]
        for g in GROUPS:
            lines.append(f"    {g:<12} {self.incidence[g]*100:+.1f}%")
        return "\n".join(lines)


@dataclass
class PolicyComparison:
    """
    Wraps a baseline and a reform ModelResult and provides difference metrics.
    This is the primary object for analysis and visualization.
    """
    baseline: ModelResult
    reform: ModelResult

    # --- Macro deltas ---
    @property
    def gdp_change_pct(self) -> float:
        """% change in GDP relative to baseline."""
        return (self.reform.gdp / self.baseline.gdp - 1.0) * 100.0

    @property
    def capital_change_pct(self) -> float:
        return (self.reform.capital_stock / self.baseline.capital_stock - 1.0) * 100.0

    @property
    def labor_change_pct(self) -> float:
        return (self.reform.labor_supply / self.baseline.labor_supply - 1.0) * 100.0

    @property
    def wage_change_pct(self) -> float:
        return (self.reform.wage / self.baseline.wage - 1.0) * 100.0

    @property
    def budget_balance_change(self) -> float:
        """Change in budget balance as % of baseline GDP."""
        return (
            (self.reform.budget_balance - self.baseline.budget_balance)
            / self.baseline.gdp
            * 100.0
        )

    @property
    def incidence_change(self) -> DistributionalIncidence:
        """
        Change in net tax burden by group (percentage points of pre-tax income).
        Positive = reform adds burden; negative = reform reduces burden.
        """
        return self.reform.incidence - self.baseline.incidence

    def summary(self) -> str:
        ic = self.incidence_change
        lines = [
            f"Policy comparison: [{self.baseline.policy_label}]  →  [{self.reform.policy_label}]",
            "",
            "  Macro effects (relative to baseline):",
            f"    GDP:              {self.gdp_change_pct:+.2f}%",
            f"    Capital stock:    {self.capital_change_pct:+.2f}%",
            f"    Labor supply:     {self.labor_change_pct:+.2f}%",
            f"    Wage:             {self.wage_change_pct:+.2f}%",
            f"    Budget balance:   {self.budget_balance_change:+.2f}pp of GDP",
            "",
            "  Distributional incidence (change in burden, pp of pre-tax income):",
            "  Negative = reform reduces burden on that group.",
        ]
        for g in GROUPS:
            lines.append(f"    {g:<12} {ic[g]*100:+.1f}pp")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "baseline_label": self.baseline.policy_label,
            "reform_label": self.reform.policy_label,
            "gdp_change_pct": self.gdp_change_pct,
            "capital_change_pct": self.capital_change_pct,
            "labor_change_pct": self.labor_change_pct,
            "wage_change_pct": self.wage_change_pct,
            "budget_balance_change_pp": self.budget_balance_change,
            "incidence_change": {g: ic * 100 for g, ic in self.incidence_change.values.items()},
            "baseline_revenue": self.baseline.revenue.to_dict(),
            "reform_revenue": self.reform.revenue.to_dict(),
        }
