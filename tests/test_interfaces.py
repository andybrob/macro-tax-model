"""
test_interfaces.py
==================
Tests for the pure data structures in calibration.py, policy.py, and results.py.
These tests do not require economy.py; they exercise only the interface layer.

Run from the project root:
    python -m pytest tests/test_interfaces.py
"""

import pytest
from pathlib import Path

from tax_model.calibration import Calibration, GROUPS
from tax_model.policy import (
    TaxPolicy,
    LaborIncomeTax,
    ConsumptionTax,
    LandValueTax,
    CorporateTax,
    PigouvianTax,
    CapitalGainsTax,
)
from tax_model.results import (
    DistributionalIncidence,
    RevenueBreakdown,
    ModelResult,
    PolicyComparison,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

YAML_PATH = Path(__file__).parent.parent / "calibration" / "us_2024.yaml"

# A representative progressive bracket schedule (income multiples of median).
PROGRESSIVE_BRACKETS = [
    (0.0, 0.10),
    (0.5, 0.22),
    (1.5, 0.24),
    (3.0, 0.32),
    (7.0, 0.37),
]


@pytest.fixture(scope="module")
def calibration():
    return Calibration.from_yaml(YAML_PATH)


@pytest.fixture
def progressive_labor_tax():
    return LaborIncomeTax(brackets=PROGRESSIVE_BRACKETS)


def _make_model_result(label, gdp, capital, labor, wage, rok, revenue_total=0.1,
                       budget_balance=0.0, incidence_dict=None):
    """Helper: build a minimal ModelResult with controlled values."""
    incidence = DistributionalIncidence.from_dict(incidence_dict or {g: 0.01 for g in GROUPS})
    revenue = RevenueBreakdown(labor_income_tax=revenue_total)
    return ModelResult(
        policy_label=label,
        gdp=gdp,
        capital_stock=capital,
        labor_supply=labor,
        wage=wage,
        return_on_capital=rok,
        revenue=revenue,
        government_spending=0.35,
        budget_balance=budget_balance,
        incidence=incidence,
    )


# ===========================================================================
# Calibration tests
# ===========================================================================

class TestCalibrationLoading:

    def test_loads_without_error(self, calibration):
        """from_yaml must parse us_2024.yaml and return a Calibration instance."""
        assert isinstance(calibration, Calibration)

    def test_all_groups_present_in_labor_income_shares(self, calibration):
        """Every income group (Q1 through Q5_top) must appear in labor_income_shares."""
        dist = calibration.income_distribution
        for g in GROUPS:
            assert g in dist.labor_income_shares, f"Missing group {g} in labor_income_shares"

    def test_all_groups_present_in_capital_income_shares(self, calibration):
        for g in GROUPS:
            assert g in calibration.income_distribution.capital_income_shares

    def test_all_groups_present_in_consumption_shares(self, calibration):
        for g in GROUPS:
            assert g in calibration.income_distribution.consumption_shares

    def test_all_groups_present_in_land_ownership_shares(self, calibration):
        for g in GROUPS:
            assert g in calibration.land.land_ownership_shares

    def test_all_groups_present_in_carbon_consumption_shares(self, calibration):
        for g in GROUPS:
            assert g in calibration.externality.carbon_consumption_shares

    def test_production_params_are_positive(self, calibration):
        """Basic sanity: all production parameters must be strictly positive."""
        p = calibration.production
        assert p.capital_share > 0
        assert p.depreciation_rate > 0
        assert p.tfp_level > 0

    def test_capital_share_in_plausible_range(self, calibration):
        """Capital share alpha should be between 0.25 and 0.45 for a realistic economy."""
        assert 0.25 <= calibration.production.capital_share <= 0.45

    def test_frisch_elasticity_positive(self, calibration):
        assert calibration.households.frisch_elasticity > 0

    def test_capital_mobility_bounded(self, calibration):
        """capital_mobility must lie in [0, 1]: 0 = closed, 1 = small open."""
        assert 0.0 <= calibration.macro.capital_mobility <= 1.0

    def test_labor_income_shares_sum_to_one(self, calibration):
        total = sum(calibration.income_distribution.labor_income_shares.values())
        assert abs(total - 1.0) < 1e-6, f"Labor income shares sum to {total}, expected 1.0"

    def test_capital_income_shares_sum_to_one(self, calibration):
        total = sum(calibration.income_distribution.capital_income_shares.values())
        assert abs(total - 1.0) < 1e-6

    def test_missing_group_raises_value_error(self):
        """from_yaml must raise ValueError if a required group is absent from a dict field."""
        import yaml, tempfile, os
        # Load the real YAML, strip one group from labor_income_shares, write a bad file.
        with open(YAML_PATH) as f:
            raw = yaml.safe_load(f)
        del raw["income_distribution"]["labor_income_shares"]["Q5_top"]
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as tmp:
            yaml.dump(raw, tmp)
            tmp_path = tmp.name
        try:
            with pytest.raises(ValueError, match="Q5_top"):
                Calibration.from_yaml(tmp_path)
        finally:
            os.unlink(tmp_path)


# ===========================================================================
# TaxPolicy / sub-policy construction tests
# ===========================================================================

class TestTaxPolicyDefaults:

    def test_default_policy_is_zero_tax(self):
        """TaxPolicy() with no arguments should represent the zero-tax counterfactual.
        Every sub-policy rate must be zero so the model can use it as the no-tax baseline."""
        p = TaxPolicy()
        assert p.labor_income.brackets == [(0.0, 0.0)]
        assert p.consumption.rate == 0.0
        assert p.land_value.rate == 0.0
        assert p.corporate.rate == 0.0
        assert p.pigouvian.rate_per_unit == 0.0
        assert p.capital_gains.rate == 0.0

    def test_full_policy_construction(self):
        """All six sub-policies can be specified together without error."""
        p = TaxPolicy(
            label="full_test",
            labor_income=LaborIncomeTax(brackets=PROGRESSIVE_BRACKETS),
            consumption=ConsumptionTax(rate=0.10, prebate_fraction=0.5),
            land_value=LandValueTax(rate=0.02),
            corporate=CorporateTax(rate=0.21, immediate_expensing=True),
            pigouvian=PigouvianTax(rate_per_unit=50.0, dividend_recycling_fraction=0.6),
            capital_gains=CapitalGainsTax(rate=0.20, lock_in_discount=0.30),
            revenue_neutral=True,
        )
        assert p.label == "full_test"
        assert p.corporate.rate == 0.21
        assert p.revenue_neutral is True

    def test_to_dict_is_serializable(self):
        """TaxPolicy.to_dict() must return a plain dict with no dataclass instances."""
        p = TaxPolicy(corporate=CorporateTax(rate=0.21))
        d = p.to_dict()
        assert isinstance(d, dict)
        # Check a nested key
        assert d["corporate"]["rate"] == 0.21


# ===========================================================================
# LaborIncomeTax computation tests
# ===========================================================================

class TestLaborIncomeTaxComputation:

    def test_zero_bracket_returns_zero(self):
        """A single (0.0, 0.0) bracket means no tax for any income level."""
        tax = LaborIncomeTax(brackets=[(0.0, 0.0)])
        for multiple in [0.5, 1.0, 5.0]:
            assert tax.effective_rate_for_income_multiple(multiple) == 0.0

    def test_effective_rate_monotonically_increasing(self, progressive_labor_tax):
        """For a progressive bracket structure the average effective rate must rise
        strictly with income — that is the definition of progression.  Failure here
        would mean the bracket logic is inverting burden from high- to low-income."""
        multiples = [0.3, 0.6, 1.0, 2.0, 4.0, 8.0, 12.0]
        rates = [progressive_labor_tax.effective_rate_for_income_multiple(m) for m in multiples]
        for i in range(len(rates) - 1):
            assert rates[i] <= rates[i + 1], (
                f"Effective rate not monotone: {multiples[i]}x → {rates[i]:.4f}, "
                f"{multiples[i+1]}x → {rates[i+1]:.4f}"
            )

    def test_marginal_rate_at_returns_correct_bracket(self, progressive_labor_tax):
        """marginal_rate_at must return the rate for the bracket that income falls into.
        This is used by the solver when computing after-tax labor incentives."""
        # Below first step-up threshold (0.5x): should be 10%
        assert progressive_labor_tax.marginal_rate_at(0.4) == pytest.approx(0.10)
        # Between 0.5x and 1.5x: should be 22%
        assert progressive_labor_tax.marginal_rate_at(1.0) == pytest.approx(0.22)
        # Between 1.5x and 3.0x: should be 24%
        assert progressive_labor_tax.marginal_rate_at(2.0) == pytest.approx(0.24)
        # Above 7.0x: should be 37%
        assert progressive_labor_tax.marginal_rate_at(10.0) == pytest.approx(0.37)

    def test_marginal_rate_exactly_at_threshold(self, progressive_labor_tax):
        """At a threshold boundary the higher bracket rate should apply (>= comparison)."""
        # At 0.5x the threshold for 22% is crossed
        assert progressive_labor_tax.marginal_rate_at(0.5) == pytest.approx(0.22)

    def test_effective_rate_below_standard_deduction_is_zero(self):
        """Income below the standard deduction should produce zero effective tax,
        which is the purpose of the deduction floor."""
        tax = LaborIncomeTax(
            brackets=[(0.0, 0.20)],
            standard_deduction_median_multiple=0.20,
        )
        # At exactly the deduction level: taxable base = 0
        assert tax.effective_rate_for_income_multiple(0.20) == pytest.approx(0.0)

    def test_zero_income_multiple_returns_zero(self, progressive_labor_tax):
        """effective_rate_for_income_multiple(0) must not divide by zero."""
        assert progressive_labor_tax.effective_rate_for_income_multiple(0.0) == 0.0


# ===========================================================================
# PigouvianTax validation
# ===========================================================================

class TestPigouvianTaxValidation:

    def test_valid_fractions_do_not_raise(self):
        """Fractions summing to exactly 1.0 must be accepted."""
        tax = PigouvianTax(rate_per_unit=50.0,
                           dividend_recycling_fraction=0.6,
                           labor_tax_offset_fraction=0.4)
        assert tax.dividend_recycling_fraction + tax.labor_tax_offset_fraction == pytest.approx(1.0)

    def test_fractions_exceeding_one_raise_value_error(self):
        """The constructor must reject recycling fractions that sum to > 1.0
        because that would require more revenue than was collected — physically impossible."""
        with pytest.raises(ValueError, match="Recycling fractions"):
            PigouvianTax(rate_per_unit=50.0,
                         dividend_recycling_fraction=0.7,
                         labor_tax_offset_fraction=0.5)

    def test_both_fractions_zero_is_valid(self):
        """All revenue going to general fund (both fractions zero) is valid."""
        tax = PigouvianTax(rate_per_unit=100.0,
                           dividend_recycling_fraction=0.0,
                           labor_tax_offset_fraction=0.0)
        assert tax.rate_per_unit == 100.0

    def test_slightly_above_one_raises(self):
        """1.0 + epsilon (outside the 1e-9 tolerance) must still raise."""
        with pytest.raises(ValueError):
            PigouvianTax(dividend_recycling_fraction=0.5, labor_tax_offset_fraction=0.6)


# ===========================================================================
# CapitalGainsTax effective rate
# ===========================================================================

class TestCapitalGainsTax:

    def test_effective_rate_applies_lock_in_discount(self):
        """The lock-in discount reduces the effective rate below the statutory rate.
        A 30% discount on a 20% rate should yield an effective rate of 14%."""
        cgt = CapitalGainsTax(rate=0.20, lock_in_discount=0.30)
        assert cgt.effective_rate == pytest.approx(0.20 * 0.70)  # 0.14

    def test_zero_lock_in_gives_statutory_rate(self):
        """With no lock-in (liquid market), effective rate = statutory rate."""
        cgt = CapitalGainsTax(rate=0.20, lock_in_discount=0.0)
        assert cgt.effective_rate == pytest.approx(0.20)

    def test_full_lock_in_gives_zero_effective_rate(self):
        """If assets are never realised (lock_in_discount=1.0) the effective rate is zero."""
        cgt = CapitalGainsTax(rate=0.20, lock_in_discount=1.0)
        assert cgt.effective_rate == pytest.approx(0.0)

    def test_effective_rate_scales_linearly_with_statutory_rate(self):
        """Doubling the statutory rate must double the effective rate
        (holding the discount constant)."""
        cgt_low = CapitalGainsTax(rate=0.10, lock_in_discount=0.30)
        cgt_high = CapitalGainsTax(rate=0.20, lock_in_discount=0.30)
        assert cgt_high.effective_rate == pytest.approx(2 * cgt_low.effective_rate)


# ===========================================================================
# DistributionalIncidence
# ===========================================================================

class TestDistributionalIncidence:

    def test_subtraction_works_element_wise(self):
        """Incidence subtraction is needed for PolicyComparison.incidence_change.
        It must operate element-wise across all groups."""
        a = DistributionalIncidence.from_dict({g: 0.10 for g in GROUPS})
        b = DistributionalIncidence.from_dict({g: 0.04 for g in GROUPS})
        diff = a - b
        for g in GROUPS:
            assert diff[g] == pytest.approx(0.06), f"Group {g}: expected 0.06, got {diff[g]}"

    def test_subtraction_with_mixed_signs(self):
        """Subtraction of a larger value from a smaller gives a negative result,
        representing a net transfer (reform reduces burden relative to baseline)."""
        a = DistributionalIncidence.from_dict({"Q1": 0.02, "Q2": 0.03, "Q3": 0.05,
                                               "Q4": 0.07, "Q5_bottom": 0.09, "Q5_top": 0.12})
        b = DistributionalIncidence.from_dict({"Q1": 0.10, "Q2": 0.08, "Q3": 0.06,
                                               "Q4": 0.05, "Q5_bottom": 0.04, "Q5_top": 0.03})
        diff = a - b
        assert diff["Q1"] == pytest.approx(-0.08)
        assert diff["Q5_top"] == pytest.approx(0.09)

    def test_default_values_are_zero(self):
        inc = DistributionalIncidence()
        for g in GROUPS:
            assert inc[g] == 0.0

    def test_as_list_preserves_group_order(self):
        """as_list() must follow GROUPS order so charts render correctly."""
        vals = {g: float(i) for i, g in enumerate(GROUPS)}
        inc = DistributionalIncidence.from_dict(vals)
        lst = inc.as_list()
        assert lst == [float(i) for i in range(len(GROUPS))]

    def test_as_pct_list_scales_by_100(self):
        inc = DistributionalIncidence.from_dict({g: 0.05 for g in GROUPS})
        assert all(abs(v - 5.0) < 1e-9 for v in inc.as_pct_list())


# ===========================================================================
# RevenueBreakdown
# ===========================================================================

class TestRevenueBreakdown:

    def test_total_sums_all_fields(self):
        """RevenueBreakdown.total must sum every revenue and outflow field.
        This is the figure used for budget-balance accounting."""
        rb = RevenueBreakdown(
            labor_income_tax=0.10,
            consumption_tax=0.05,
            land_value_tax=0.02,
            corporate_tax=0.03,
            pigouvian_tax=0.01,
            capital_gains_tax=0.02,
            prebate_cost=-0.01,
            pigouvian_dividend_cost=-0.005,
        )
        expected = 0.10 + 0.05 + 0.02 + 0.03 + 0.01 + 0.02 - 0.01 - 0.005
        assert rb.total == pytest.approx(expected)

    def test_all_zero_total_is_zero(self):
        assert RevenueBreakdown().total == pytest.approx(0.0)

    def test_negative_outflows_reduce_total(self):
        """Prebate and dividend costs are outflows (negative); they must reduce the total."""
        rb = RevenueBreakdown(labor_income_tax=0.10, prebate_cost=-0.10)
        assert rb.total == pytest.approx(0.0)

    def test_to_dict_contains_all_fields(self):
        rb = RevenueBreakdown(labor_income_tax=0.08)
        d = rb.to_dict()
        assert "labor_income_tax" in d
        assert "prebate_cost" in d
        assert d["labor_income_tax"] == 0.08


# ===========================================================================
# ModelResult
# ===========================================================================

class TestModelResult:

    def test_summary_returns_non_empty_string(self):
        """summary() must return a non-empty string containing key macro labels.
        It is used for logging and quick visual inspection."""
        result = _make_model_result("test_policy", gdp=1.0, capital=2.5,
                                    labor=1.0, wage=0.65, rok=0.05)
        s = result.summary()
        assert isinstance(s, str)
        assert len(s) > 0
        assert "test_policy" in s

    def test_summary_contains_all_groups(self):
        """The distributional table inside summary() must list every income group."""
        result = _make_model_result("p", gdp=1.0, capital=2.5, labor=1.0, wage=0.65, rok=0.05)
        s = result.summary()
        for g in GROUPS:
            assert g in s, f"Group {g} missing from ModelResult.summary()"

    def test_revenue_total_accessible(self):
        """revenue.total should be computable from a constructed ModelResult."""
        rb = RevenueBreakdown(labor_income_tax=0.15, corporate_tax=0.03)
        result = ModelResult(policy_label="test", gdp=1.0, capital_stock=2.5,
                             labor_supply=1.0, wage=0.65, return_on_capital=0.05,
                             revenue=rb, government_spending=0.35, budget_balance=-0.17)
        assert result.revenue.total == pytest.approx(0.18)


# ===========================================================================
# PolicyComparison
# ===========================================================================

class TestPolicyComparison:

    @pytest.fixture
    def comparison(self):
        baseline = _make_model_result("baseline", gdp=1.0, capital=2.5,
                                      labor=1.0, wage=0.65, rok=0.05,
                                      revenue_total=0.15, budget_balance=-0.20)
        reform = _make_model_result("reform", gdp=1.05, capital=2.7,
                                    labor=1.02, wage=0.67, rok=0.048,
                                    revenue_total=0.17, budget_balance=-0.18)
        return PolicyComparison(baseline=baseline, reform=reform)

    def test_gdp_change_pct(self, comparison):
        """gdp_change_pct must be (reform/baseline - 1) * 100."""
        expected = (1.05 / 1.0 - 1.0) * 100
        assert comparison.gdp_change_pct == pytest.approx(expected)

    def test_capital_change_pct(self, comparison):
        expected = (2.7 / 2.5 - 1.0) * 100
        assert comparison.capital_change_pct == pytest.approx(expected)

    def test_labor_change_pct(self, comparison):
        expected = (1.02 / 1.0 - 1.0) * 100
        assert comparison.labor_change_pct == pytest.approx(expected)

    def test_wage_change_pct(self, comparison):
        expected = (0.67 / 0.65 - 1.0) * 100
        assert comparison.wage_change_pct == pytest.approx(expected)

    def test_budget_balance_change(self, comparison):
        """budget_balance_change = (reform_balance - baseline_balance) / baseline_gdp * 100."""
        expected = (-0.18 - (-0.20)) / 1.0 * 100
        assert comparison.budget_balance_change == pytest.approx(expected)

    def test_incidence_change_is_distributional_incidence(self, comparison):
        """incidence_change must return a DistributionalIncidence object."""
        ic = comparison.incidence_change
        assert isinstance(ic, DistributionalIncidence)

    def test_incidence_change_computes_difference(self, comparison):
        """incidence_change must equal reform.incidence - baseline.incidence element-wise."""
        ic = comparison.incidence_change
        for g in GROUPS:
            expected = comparison.reform.incidence[g] - comparison.baseline.incidence[g]
            assert ic[g] == pytest.approx(expected)

    def test_summary_returns_non_empty_string(self, comparison):
        """PolicyComparison.summary() must be a non-empty string that names both policies."""
        s = comparison.summary()
        assert isinstance(s, str)
        assert len(s) > 0
        assert "baseline" in s
        assert "reform" in s

    def test_summary_contains_all_groups(self, comparison):
        s = comparison.summary()
        for g in GROUPS:
            assert g in s, f"Group {g} missing from PolicyComparison.summary()"

    def test_to_dict_structure(self, comparison):
        """to_dict() must include the key macro delta fields and incidence_change."""
        d = comparison.to_dict()
        assert "gdp_change_pct" in d
        assert "capital_change_pct" in d
        assert "incidence_change" in d
        assert "baseline_label" in d
        assert d["reform_label"] == "reform"
