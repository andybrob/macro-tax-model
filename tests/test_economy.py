"""
test_economy.py
===============
Economic sanity tests for Economy.solve().

These are *behavioural contracts* — they encode what any coherent general-
equilibrium tax model must satisfy.  A model that passes all of them is not
necessarily correct, but one that fails any of them is definitely wrong.

Each test is structured as:
    1. Arrange  — build a policy (or two) that differs in one dimension.
    2. Act      — call economy.solve() to get ModelResult(s).
    3. Assert   — check the directional / sign / magnitude prediction.

Tests are deliberately NOT asserting exact numbers, only qualitative direction
and loose bounds.  That keeps them valid even as model parameters are refined.

Run from the project root:
    python -m pytest tests/test_economy.py
"""

import pytest
from pathlib import Path

# NOTE: economy.py does not yet exist.  These imports will fail until it is
# implemented.  That is intentional — tests are written first (TDD).
from tax_model import (
    Calibration,
    Economy,
    TaxPolicy,
    LaborIncomeTax,
    ConsumptionTax,
    LandValueTax,
    CorporateTax,
    PigouvianTax,
    CapitalGainsTax,
    ModelResult,
    PolicyComparison,
)
from tax_model.calibration import GROUPS


# ---------------------------------------------------------------------------
# Constants and shared data
# ---------------------------------------------------------------------------

YAML_PATH = Path(__file__).parent.parent / "calibration" / "us_2024.yaml"

# Approximate current US law labour-income tax brackets.
# Thresholds are expressed as multiples of median household income.
CURRENT_LAW_BRACKETS = [
    (0.0, 0.10),
    (0.5, 0.22),
    (1.5, 0.24),
    (3.0, 0.32),
    (7.0, 0.37),
]

# A flat labor-tax schedule with a higher rate used for comparison tests.
HIGH_LABOR_BRACKETS = [
    (0.0, 0.20),
    (0.5, 0.32),
    (1.5, 0.36),
    (3.0, 0.44),
    (7.0, 0.50),
]


# ---------------------------------------------------------------------------
# Module-scoped fixtures (heavy: calibration + economy constructed once)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def calibration():
    """Load the 2024 US calibration from YAML once per test session."""
    return Calibration.from_yaml(YAML_PATH)


@pytest.fixture(scope="module")
def economy(calibration):
    """Construct the Economy model once.  Economy is stateless after construction."""
    return Economy(calibration)


@pytest.fixture(scope="module")
def current_law_policy():
    """
    Approximate current US law:
      - Progressive labour income tax (2024 brackets)
      - Corporate tax at 21%
      - Capital gains tax at 20% with 30% lock-in discount
    """
    return TaxPolicy(
        label="current_law",
        labor_income=LaborIncomeTax(brackets=CURRENT_LAW_BRACKETS),
        corporate=CorporateTax(rate=0.21),
        capital_gains=CapitalGainsTax(rate=0.20, lock_in_discount=0.30),
    )


@pytest.fixture(scope="module")
def baseline_result(economy, current_law_policy):
    """Solve the model under current law.  Used as the reference for comparisons."""
    return economy.solve(current_law_policy)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _solve_and_compare(economy, base_policy, reform_policy) -> PolicyComparison:
    """Convenience: solve both and return a PolicyComparison."""
    base = economy.solve(base_policy)
    reform = economy.solve(reform_policy)
    return PolicyComparison(baseline=base, reform=reform)


# ===========================================================================
# Test 1 — Baseline sanity
# ===========================================================================

class TestBaselineSanity:
    """
    The model must produce a valid, internally consistent steady state when
    solved under current law.  If any of these assertions fail the model is
    fundamentally broken (negative prices, etc.).
    """

    def test_gdp_is_positive(self, baseline_result):
        """GDP (normalised to ~1.0 at baseline) must be strictly positive."""
        assert baseline_result.gdp > 0, f"GDP = {baseline_result.gdp}"

    def test_capital_stock_is_positive(self, baseline_result):
        """A real economy must have a positive capital stock."""
        assert baseline_result.capital_stock > 0

    def test_labor_supply_is_positive(self, baseline_result):
        """Aggregate effective labour units must be positive."""
        assert baseline_result.labor_supply > 0

    def test_wage_is_positive(self, baseline_result):
        """The real pre-tax wage must be positive — labour has positive marginal product."""
        assert baseline_result.wage > 0

    def test_return_on_capital_is_positive(self, baseline_result):
        """The net real return on capital must be positive in a productive economy."""
        assert baseline_result.return_on_capital > 0

    def test_incidence_values_bounded(self, baseline_result):
        """
        Incidence (burden as fraction of pre-tax income) must lie in [-1, 1].
        A value outside this range implies the model is computing burdens larger
        than the group's entire income, which is implausible in a steady state.
        """
        for g in GROUPS:
            v = baseline_result.incidence[g]
            assert -1.0 <= v <= 1.0, (
                f"Incidence for {g} = {v:.4f}, outside [-1, 1]"
            )

    def test_summary_runs_without_error(self, baseline_result):
        """summary() must be callable and return a non-empty string."""
        s = baseline_result.summary()
        assert isinstance(s, str) and len(s) > 0


# ===========================================================================
# Test 2 — Higher corporate tax lowers capital stock
# ===========================================================================

class TestCorporateTaxReducesCapital:
    """
    Neoclassical result: a higher corporate tax raises the cost of capital,
    so the equilibrium capital-output ratio falls.  This is the central
    prediction of Jorgenson-Hall cost-of-capital theory.

    We test the direction, not the magnitude.
    """

    def test_high_corporate_tax_lowers_capital(self, economy):
        """
        CorporateTax(rate=0.35) must produce lower capital_stock than
        CorporateTax(rate=0.10), holding all other taxes at zero.
        """
        low_tax = TaxPolicy(label="low_corp", corporate=CorporateTax(rate=0.10))
        high_tax = TaxPolicy(label="high_corp", corporate=CorporateTax(rate=0.35))
        result_low = economy.solve(low_tax)
        result_high = economy.solve(high_tax)
        assert result_high.capital_stock < result_low.capital_stock, (
            f"Expected higher corporate tax to reduce capital.  "
            f"Low-rate K={result_low.capital_stock:.4f}, "
            f"High-rate K={result_high.capital_stock:.4f}"
        )


# ===========================================================================
# Test 3 — Immediate expensing increases capital
# ===========================================================================

class TestImmediateExpensingIncreasesCapital:
    """
    Immediate (100%) expensing of investment eliminates the effective marginal
    tax on new capital under a cash-flow tax (Hall-Jorgenson, 1967).  Even
    at the same statutory rate, capital formation should be higher with
    expensing than with depreciation schedules.
    """

    def test_expensing_raises_capital_stock(self, economy):
        """
        CorporateTax(rate=0.21, immediate_expensing=True) must yield higher
        capital_stock than CorporateTax(rate=0.21, immediate_expensing=False).
        """
        with_expensing = TaxPolicy(
            label="expensing",
            corporate=CorporateTax(rate=0.21, immediate_expensing=True),
        )
        without_expensing = TaxPolicy(
            label="depreciation",
            corporate=CorporateTax(rate=0.21, immediate_expensing=False),
        )
        result_exp = economy.solve(with_expensing)
        result_dep = economy.solve(without_expensing)
        assert result_exp.capital_stock > result_dep.capital_stock, (
            f"Expected expensing to increase capital.  "
            f"With={result_exp.capital_stock:.4f}, Without={result_dep.capital_stock:.4f}"
        )


# ===========================================================================
# Test 4 — Higher labour income tax reduces labour supply
# ===========================================================================

class TestLaborTaxReducesLaborSupply:
    """
    A higher marginal labour income tax reduces the after-tax return to work,
    and with a positive Frisch elasticity labour supply falls.  This is a
    standard result in any model with elastic labour supply.
    """

    def test_high_labor_tax_lowers_labor_supply(self, economy):
        """
        A flat 40% labour tax across all brackets must produce lower labor_supply
        than the current-law progressive schedule.
        """
        low_labor = TaxPolicy(
            label="low_labor",
            labor_income=LaborIncomeTax(brackets=CURRENT_LAW_BRACKETS),
        )
        high_labor = TaxPolicy(
            label="high_labor",
            labor_income=LaborIncomeTax(brackets=HIGH_LABOR_BRACKETS),
        )
        result_low = economy.solve(low_labor)
        result_high = economy.solve(high_labor)
        assert result_high.labor_supply < result_low.labor_supply, (
            f"Expected higher labour tax to reduce supply.  "
            f"Low={result_low.labor_supply:.4f}, High={result_high.labor_supply:.4f}"
        )


# ===========================================================================
# Test 5 — Land Value Tax incidence is concentrated at the top
# ===========================================================================

class TestLVTIncidenceIsCapitalConcentrated:
    """
    Land is disproportionately owned by high-income households (Q5_top and Q5_bottom).
    A Land Value Tax (LVT) must therefore fall primarily on those groups.

    Economic reasoning: because land supply is perfectly inelastic, the tax cannot
    be shifted.  Incidence follows ownership shares directly.  If the model disperses
    LVT burden uniformly it is ignoring the ownership distribution.
    """

    def test_lvt_burden_higher_for_top_quintile_than_bottom(self, economy, calibration):
        """
        Under an LVT, incidence[Q5_top] > incidence[Q1] and incidence[Q5_bottom] > incidence[Q2].
        """
        lvt_policy = TaxPolicy(label="lvt", land_value=LandValueTax(rate=0.02))
        result = economy.solve(lvt_policy)
        assert result.incidence["Q5_top"] > result.incidence["Q1"], (
            f"LVT should burden Q5_top more than Q1.  "
            f"Q5_top={result.incidence['Q5_top']:.4f}, Q1={result.incidence['Q1']:.4f}"
        )
        assert result.incidence["Q5_bottom"] > result.incidence["Q2"], (
            f"LVT should burden Q5_bottom more than Q2.  "
            f"Q5_bottom={result.incidence['Q5_bottom']:.4f}, Q2={result.incidence['Q2']:.4f}"
        )

    def test_lvt_q5top_has_largest_burden(self, economy, calibration):
        """
        Q5_top should bear the largest absolute incidence under an LVT,
        consistent with their ~32% land ownership share in the calibration.
        """
        lvt_policy = TaxPolicy(label="lvt_top", land_value=LandValueTax(rate=0.02))
        result = economy.solve(lvt_policy)
        incidences = {g: result.incidence[g] for g in GROUPS}
        max_group = max(incidences, key=incidences.get)
        # Either Q5_top or Q5_bottom (both are top quintile sub-groups)
        assert max_group in ("Q5_top", "Q5_bottom"), (
            f"Expected LVT burden to peak in top quintile, got {max_group}"
        )


# ===========================================================================
# Test 6 — Consumption tax without prebate is regressive
# ===========================================================================

class TestConsumptionTaxIsRegressive:
    """
    A flat consumption tax (like a VAT) is regressive before any prebate because
    lower-income households spend a higher fraction of their income on consumption.
    The effective burden as a share of income is therefore higher for Q1 than Q5_top.

    This is a well-documented empirical regularity (Congressional Budget Office,
    Tax Policy Center distributional analyses).
    """

    def test_consumption_tax_burden_higher_for_q1_than_q5top(self, economy):
        """
        ConsumptionTax(rate=0.15, prebate_fraction=0.0):
        incidence[Q1] > incidence[Q5_top]  (more regressive burden for low income)
        """
        policy = TaxPolicy(
            label="flat_vat",
            consumption=ConsumptionTax(rate=0.15, prebate_fraction=0.0),
        )
        result = economy.solve(policy)
        assert result.incidence["Q1"] > result.incidence["Q5_top"], (
            f"Expected flat VAT to burden Q1 more than Q5_top.  "
            f"Q1={result.incidence['Q1']:.4f}, Q5_top={result.incidence['Q5_top']:.4f}"
        )


# ===========================================================================
# Test 7 — Full prebate makes consumption tax progressive
# ===========================================================================

class TestPrebateMakesConsumptionTaxProgressive:
    """
    A prebate returns to every household an equal lump sum equal to the tax
    owed on poverty-line consumption.  For Q1 (whose consumption is near or
    below the poverty line) this can eliminate or reverse the net burden.
    """

    def test_full_prebate_reduces_q1_burden_more_than_q5top(self, economy):
        """
        ConsumptionTax(rate=0.15, prebate_fraction=1.0) should reduce Q1's burden
        MORE (in pp) than Q5_top's burden, because the equal per-capita prebate is
        larger relative to Q1's income.

        Note: the prebate may not flip Q1 to negative if Q1 consumes more than the
        poverty-line threshold (their consumption share > poverty_line_share * pop_share).
        The correct economic property is that the prebate is MORE progressive than
        the raw consumption tax — it compresses the incidence distribution.
        """
        no_prebate = TaxPolicy(
            label="vat_no_prebate",
            consumption=ConsumptionTax(rate=0.15, prebate_fraction=0.0),
        )
        with_prebate = TaxPolicy(
            label="vat_full_prebate",
            consumption=ConsumptionTax(rate=0.15, prebate_fraction=1.0),
        )
        r_no  = economy.solve(no_prebate)
        r_yes = economy.solve(with_prebate)

        q1_reduction    = r_no.incidence["Q1"]     - r_yes.incidence["Q1"]
        q5top_reduction = r_no.incidence["Q5_top"] - r_yes.incidence["Q5_top"]

        assert q1_reduction > q5top_reduction, (
            f"Prebate should reduce Q1's burden more than Q5_top's.  "
            f"Q1 reduction={q1_reduction:.4f}, Q5_top reduction={q5top_reduction:.4f}"
        )


# ===========================================================================
# Test 8 — Pigouvian dividend is progressive
# ===========================================================================

class TestPigouvianDividendIsProgressive:
    """
    A carbon tax (fee-and-dividend) with 100% of revenue returned as an equal
    per-capita cash dividend is strongly progressive.  Lower-income groups emit
    less carbon per dollar of income, so their carbon tax burden is small, but
    they receive the same dollar dividend.  The net effect is a transfer from
    high emitters to low emitters.

    This is the policy rationale behind Citizens' Climate Lobby's fee-and-dividend
    proposal and the Canada carbon rebate programme.
    """

    def test_dividend_burden_lower_for_q1_than_q5top(self, economy):
        """
        PigouvianTax(rate_per_unit=50.0, dividend_recycling_fraction=1.0):
        incidence[Q1] < incidence[Q5_top]  (Q1 net better-off or less burdened)
        """
        policy = TaxPolicy(
            label="fee_and_dividend",
            pigouvian=PigouvianTax(
                rate_per_unit=50.0,
                dividend_recycling_fraction=1.0,
                labor_tax_offset_fraction=0.0,
            ),
        )
        result = economy.solve(policy)
        assert result.incidence["Q1"] < result.incidence["Q5_top"], (
            f"Expected Q1 to bear lower net burden under fee-and-dividend.  "
            f"Q1={result.incidence['Q1']:.4f}, Q5_top={result.incidence['Q5_top']:.4f}"
        )

    def test_q1_net_transfer_under_high_dividend(self, economy):
        """
        At a high enough carbon price with full dividend recycling, Q1 should
        receive a net transfer (negative incidence).
        """
        policy = TaxPolicy(
            label="high_fee_and_dividend",
            pigouvian=PigouvianTax(
                rate_per_unit=150.0,
                dividend_recycling_fraction=1.0,
            ),
        )
        result = economy.solve(policy)
        assert result.incidence["Q1"] < 0.0, (
            f"Expected Q1 net transfer at high carbon price + full dividend.  "
            f"Q1 incidence = {result.incidence['Q1']:.4f}"
        )


# ===========================================================================
# Test 9 — Capital mobility shifts corporate tax burden to labour
# ===========================================================================

class TestCapitalMobilityShiftsCorporateBurden:
    """
    The Harberger-Gravelle debate in one test.

    In a CLOSED economy: capital cannot flee, so the corporate tax is borne by
    capital owners — incidence is concentrated in Q5 (high capital ownership).

    In an OPEN economy (small open economy / perfect capital mobility):
    the domestic return on capital is pinned by world rates; the tax raises the
    cost of capital, which reduces investment and lowers wages.  Burden shifts
    to workers — concentrated in labour income groups (Q1–Q4).

    We test the directional shift, not the exact split.

    Implementation note: the calibration's macro.capital_mobility parameter
    controls this.  To test cleanly we build two Calibration objects or accept
    that the Economy fixture wraps the default mobility.  Here we use two
    separate Economy instances with patched calibrations.
    """

    def _make_economy_with_mobility(self, calibration, mobility: float) -> Economy:
        """Return a new Economy whose calibration has the given capital_mobility."""
        import copy
        from tax_model.calibration import MacroParams
        cal2 = copy.deepcopy(calibration)
        cal2.macro = MacroParams(
            capital_mobility=mobility,
            elasticity_of_taxable_income=cal2.macro.elasticity_of_taxable_income,
            capital_income_eti=cal2.macro.capital_income_eti,
            target_capital_output_ratio=cal2.macro.target_capital_output_ratio,
            target_labor_share=cal2.macro.target_labor_share,
        )
        return Economy(cal2)

    def test_open_economy_corporate_burden_more_on_labor(self, calibration):
        """
        With high capital_mobility, corporate tax incidence on Q1+Q2 (labour-heavy)
        should be larger than in a closed economy with the same statutory rate.
        """
        corp_policy = TaxPolicy(
            label="corp_21",
            corporate=CorporateTax(rate=0.21),
        )
        closed_economy = self._make_economy_with_mobility(calibration, mobility=0.0)
        open_economy = self._make_economy_with_mobility(calibration, mobility=1.0)

        result_closed = closed_economy.solve(corp_policy)
        result_open = open_economy.solve(corp_policy)

        # In the open economy, more burden falls on labour groups (Q1, Q2)
        labor_burden_closed = result_closed.incidence["Q1"] + result_closed.incidence["Q2"]
        labor_burden_open = result_open.incidence["Q1"] + result_open.incidence["Q2"]

        # In the closed economy, more burden falls on capital owners (Q5_top, Q5_bottom)
        capital_burden_closed = (result_closed.incidence["Q5_top"]
                                 + result_closed.incidence["Q5_bottom"])
        capital_burden_open = (result_open.incidence["Q5_top"]
                               + result_open.incidence["Q5_bottom"])

        assert labor_burden_open > labor_burden_closed, (
            f"Open economy should shift more burden to labour.  "
            f"Open Q1+Q2={labor_burden_open:.4f}, Closed Q1+Q2={labor_burden_closed:.4f}"
        )
        assert capital_burden_closed > capital_burden_open, (
            f"Closed economy should concentrate more burden on capital owners.  "
            f"Closed Q5={capital_burden_closed:.4f}, Open Q5={capital_burden_open:.4f}"
        )


# ===========================================================================
# Test 10 — Budget balance
# ===========================================================================

class TestBudgetBalance:
    """
    Accounting identity: budget_balance = revenue.total - government_spending.

    For a zero-tax policy:
        revenue = 0  →  budget_balance ≈ -government_spending
    This is a pure accounting check — if it fails, the model has an error in
    its revenue or balance calculation.

    For any non-zero positive tax rate:
        revenue.total > 0
    """

    def test_zero_tax_budget_balance_equals_negative_spending(self, economy):
        """
        Under a zero-tax policy, the government collects no revenue, so the
        budget deficit equals the spending level.
        """
        zero_policy = TaxPolicy(label="zero_tax")
        result = economy.solve(zero_policy)

        expected_balance = -result.government_spending
        assert result.budget_balance == pytest.approx(expected_balance, rel=1e-3), (
            f"Zero-tax budget balance should equal -spending.  "
            f"Balance={result.budget_balance:.4f}, -Spending={expected_balance:.4f}"
        )

    def test_positive_tax_rates_generate_positive_revenue(self, economy, current_law_policy):
        """
        Any policy with at least one positive tax rate must generate positive gross revenue.
        If revenue.total <= 0 under current law, the model is not accounting for tax receipts.
        """
        result = economy.solve(current_law_policy)
        assert result.revenue.total > 0, (
            f"Expected positive total revenue under current law.  "
            f"revenue.total = {result.revenue.total:.4f}"
        )

    def test_budget_balance_identity_holds(self, economy, current_law_policy):
        """
        The budget balance must equal revenue.total - government_spending.
        This is a pure accounting identity: if violated, there is a coding bug
        in the government module.
        """
        result = economy.solve(current_law_policy)
        expected = result.revenue.total - result.government_spending
        assert result.budget_balance == pytest.approx(expected, rel=1e-4), (
            f"Budget balance identity violated.  "
            f"Stored={result.budget_balance:.6f}, Computed={expected:.6f}"
        )

    def test_corporate_tax_raises_revenue(self, economy):
        """
        Adding a 21% corporate tax to an otherwise zero-tax baseline must
        increase corporate_tax revenue from zero.
        """
        zero_policy = TaxPolicy(label="zero_tax")
        corp_policy = TaxPolicy(label="corp_21", corporate=CorporateTax(rate=0.21))
        result_zero = economy.solve(zero_policy)
        result_corp = economy.solve(corp_policy)
        assert result_corp.revenue.corporate_tax > result_zero.revenue.corporate_tax, (
            f"Corporate tax should raise corporate revenue.  "
            f"Zero={result_zero.revenue.corporate_tax:.4f}, "
            f"21%={result_corp.revenue.corporate_tax:.4f}"
        )
