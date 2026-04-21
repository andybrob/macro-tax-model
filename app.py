"""
Macro Tax Model — Streamlit GUI
================================
Interactive US tax policy explorer with named preset scenarios.

Run:
    streamlit run app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import matplotlib.pyplot as plt

from tax_model import (
    Calibration, Economy, TaxPolicy,
    LaborIncomeTax, PayrollTax, ConsumptionTax, LandValueTax,
    CorporateTax, PigouvianTax, CapitalGainsTax,
)
from tax_model.results import PolicyComparison
from tax_model.presets import PRESETS
from tax_model.visualization import (
    plot_dashboard, plot_macro_bars, plot_revenue,
    plot_incidence, plot_transition,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="US Tax Policy Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("US Tax Policy Explorer")
st.caption(
    "Two-period OLG macroeconomic model · Calibrated to US 2024 data · "
    "Long-run steady-state + transition path analysis"
)


# ---------------------------------------------------------------------------
# Load economy (cached)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_economy() -> Economy:
    cal = Calibration.from_yaml(Path(__file__).parent / "calibration" / "us_2024.yaml")
    return Economy(cal)

economy = load_economy()


# ---------------------------------------------------------------------------
# Preset loader helpers
# ---------------------------------------------------------------------------

def _policy_to_slider_state(policy: TaxPolicy) -> dict:
    """Extract slider values from a TaxPolicy object."""
    b = policy.labor_income.brackets
    # Helper: find rate at a given threshold, defaulting to last bracket
    def _rate(thresh):
        rates = [r for t, r in b if t <= thresh]
        return rates[-1] if rates else 0.0

    return {
        "l1": _rate(0.0), "l2": _rate(0.5), "l3": _rate(1.5),
        "l4": _rate(3.0), "l5": _rate(7.0),
        "std": policy.labor_income.standard_deduction_median_multiple,
        "pr_emp": policy.payroll.employee_rate,
        "pr_er":  policy.payroll.employer_rate,
        "pr_ceil": policy.payroll.wage_ceiling_median_multiple,
        "cons_rate": policy.consumption.rate,
        "prebate":   policy.consumption.prebate_fraction,
        "lvt":       policy.land_value.rate,
        "corp":      policy.corporate.rate,
        "expensing": policy.corporate.immediate_expensing,
        "int_ded":   policy.corporate.interest_deductibility,
        "border":    policy.corporate.border_adjustment,
        "pig_rate":  policy.pigouvian.rate_per_unit,
        "div_frac":  policy.pigouvian.dividend_recycling_fraction,
        "lab_off":   policy.pigouvian.labor_tax_offset_fraction,
        "cg_rate":   policy.capital_gains.rate,
        "cg_idx":    policy.capital_gains.inflation_indexed,
        "lock_in":   policy.capital_gains.lock_in_discount,
    }


def _load_preset_to_state(key: str, preset_name: str):
    """Load a named preset into session_state keys for sliders."""
    policy = PRESETS[preset_name]()
    vals = _policy_to_slider_state(policy)
    for field, val in vals.items():
        st.session_state[f"{key}_{field}"] = val


# ---------------------------------------------------------------------------
# Sidebar — global settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Model Settings")

    capital_mobility_pct = st.slider(
        "Capital mobility (0=closed, 100=open)",
        0, 100, 50, 5, key="cap_mob",
        help=(
            "0=closed economy: corporate tax falls on capital owners (Q5_top).\n"
            "100=open economy: corporate tax shifts entirely to workers via lower wages.\n"
            "This is the single most contested assumption in corporate tax incidence."
        ),
    )

    run_transition = st.checkbox(
        "Compute transition path",
        value=False,
        help="Simulate 30-period adjustment from baseline to reform. Slower.",
    )
    transition_periods = st.slider("Transition periods", 10, 50, 30, 5,
                                   disabled=not run_transition)

    st.divider()
    st.caption(
        "Model: Two-period OLG · Cobb-Douglas production · "
        "6 income groups · All-in rates (federal + state/local)"
    )


# ---------------------------------------------------------------------------
# Build active economy with capital mobility setting
# ---------------------------------------------------------------------------

@st.cache_resource
def _get_economy_with_mobility(mobility: float) -> Economy:
    from tax_model.sensitivity import _set_nested
    base_cal = Calibration.from_yaml(
        Path(__file__).parent / "calibration" / "us_2024.yaml"
    )
    if abs(mobility - base_cal.macro.capital_mobility) > 1e-4:
        new_cal = _set_nested(base_cal, "macro.capital_mobility", mobility)
        return Economy(new_cal)
    return Economy(base_cal)

active_economy = _get_economy_with_mobility(capital_mobility_pct / 100.0)


# ---------------------------------------------------------------------------
# Two columns: Baseline | Reform
# ---------------------------------------------------------------------------

col_base, col_reform = st.columns(2)


def _policy_controls(key: str, default_preset: str) -> TaxPolicy:
    """
    Render all policy controls for one column (baseline or reform).
    Returns the constructed TaxPolicy.
    `key` is a prefix for all session_state keys (e.g. "b" or "r").
    """
    # Preset selector
    preset_names = ["(Custom)"] + list(PRESETS.keys())
    selected = st.selectbox(
        "Load preset", preset_names, key=f"{key}_preset",
        index=preset_names.index(default_preset) if default_preset in preset_names else 0,
    )
    if selected != "(Custom)" and st.button(f"Apply preset", key=f"{key}_apply"):
        _load_preset_to_state(key, selected)
        st.rerun()

    def sv(field, default):
        return st.session_state.get(f"{key}_{field}", default)

    tab_labor, tab_payroll, tab_cons, tab_corp, tab_pig, tab_cg = st.tabs([
        "Income Tax", "Payroll", "Consumption + LVT", "Corporate", "Pigouvian", "Cap. Gains"
    ])

    with tab_labor:
        l1 = st.slider("Bracket 1 rate (0–0.5× median)",  0.0, 0.5, sv("l1", 0.15), 0.01, key=f"{key}_l1")
        l2 = st.slider("Bracket 2 rate (0.5–1.5× median)", 0.0, 0.5, sv("l2", 0.27), 0.01, key=f"{key}_l2")
        l3 = st.slider("Bracket 3 rate (1.5–3× median)",   0.0, 0.5, sv("l3", 0.29), 0.01, key=f"{key}_l3")
        l4 = st.slider("Bracket 4 rate (3–7× median)",     0.0, 0.6, sv("l4", 0.37), 0.01, key=f"{key}_l4")
        l5 = st.slider("Top bracket rate (7×+ median)",    0.0, 0.6, sv("l5", 0.42), 0.01, key=f"{key}_l5")
        std = st.slider("Standard deduction (× median)",   0.0, 0.6, sv("std", 0.20), 0.01, key=f"{key}_std")

    with tab_payroll:
        st.caption("FICA (Social Security + Medicare). Both employee and employer portions fall on workers in the long run.")
        pr_emp  = st.slider("Employee rate",  0.0, 0.15, sv("pr_emp",  0.0765), 0.005, format="%.3f", key=f"{key}_pr_emp")
        pr_er   = st.slider("Employer rate",  0.0, 0.15, sv("pr_er",   0.0765), 0.005, format="%.3f", key=f"{key}_pr_er")
        pr_ceil = st.slider("SS wage ceiling (× median)", 0.5, 5.0, sv("pr_ceil", 2.8), 0.1, key=f"{key}_pr_ceil")
        st.caption(f"Combined rate: {(pr_emp+pr_er)*100:.1f}%  |  Medicare (2.9%) continues above ceiling")

    with tab_cons:
        cons_rate = st.slider("Consumption tax rate", 0.0, 0.35, sv("cons_rate", 0.02), 0.01, key=f"{key}_cons_rate")
        prebate   = st.slider("Prebate fraction",     0.0, 1.0,  sv("prebate",   0.0),  0.05, key=f"{key}_prebate",
                              help="Fraction of poverty-line consumption returned as equal per-capita cash transfer.")
        st.divider()
        lvt = st.slider("Land Value Tax rate (fraction of land value/yr)",
                        0.0, 0.10, sv("lvt", 0.0), 0.005, format="%.3f", key=f"{key}_lvt",
                        help="Zero deadweight loss — land supply is perfectly inelastic.")

    with tab_corp:
        corp      = st.slider("Corporate rate",             0.0, 0.50, sv("corp",     0.21), 0.01, key=f"{key}_corp")
        expensing = st.checkbox("Immediate expensing",                  sv("expensing", False),      key=f"{key}_expensing")
        int_ded   = st.slider("Interest deductibility",    0.0, 1.0,  sv("int_ded",  1.0),  0.05, key=f"{key}_int_ded",
                              help="1.0 = current law; 0.0 = DBCFT (equity-neutral)")
        border    = st.checkbox("Destination-based border adjustment",  sv("border",    False),      key=f"{key}_border")

    with tab_pig:
        pig_rate = st.slider("Rate ($/ton CO₂-equivalent)", 0.0, 300.0, sv("pig_rate", 0.0), 5.0, key=f"{key}_pig_rate")
        div_frac = st.slider("Dividend recycling fraction", 0.0, 1.0,   sv("div_frac", 0.0), 0.05, key=f"{key}_div_frac")
        max_lab  = max(0.0, 1.0 - div_frac - 0.001)
        lab_off  = st.slider("Labor tax offset fraction",  0.0, max_lab, min(sv("lab_off", 0.0), max_lab), 0.05, key=f"{key}_lab_off")

    with tab_cg:
        cg_rate = st.slider("Capital gains rate",     0.0, 0.50, sv("cg_rate", 0.238), 0.01, key=f"{key}_cg_rate")
        cg_idx  = st.checkbox("Inflation-indexed",              sv("cg_idx",  False),        key=f"{key}_cg_idx")
        lock_in = st.slider("Lock-in discount",      0.0, 0.60, sv("lock_in", 0.30),  0.05, key=f"{key}_lock_in",
                            help="Effective rate = statutory × (1 − lock_in).")

    label = st.text_input("Policy label", value=selected if selected != "(Custom)" else f"{key.upper()} Policy",
                          key=f"{key}_label")

    return TaxPolicy(
        label=label,
        labor_income=LaborIncomeTax(
            brackets=[(0.0, l1), (0.5, l2), (1.5, l3), (3.0, l4), (7.0, l5)],
            standard_deduction_median_multiple=std,
        ),
        payroll=PayrollTax(
            employee_rate=pr_emp,
            employer_rate=pr_er,
            wage_ceiling_median_multiple=pr_ceil,
            medicare_rate_above_ceiling=0.029,
        ),
        consumption=ConsumptionTax(rate=cons_rate, prebate_fraction=prebate),
        land_value=LandValueTax(rate=lvt),
        corporate=CorporateTax(rate=corp, immediate_expensing=expensing,
                               interest_deductibility=int_ded, border_adjustment=border),
        pigouvian=PigouvianTax(
            rate_per_unit=pig_rate,
            dividend_recycling_fraction=div_frac,
            labor_tax_offset_fraction=min(lab_off, 1.0 - div_frac),
        ),
        capital_gains=CapitalGainsTax(rate=cg_rate, inflation_indexed=cg_idx, lock_in_discount=lock_in),
    )


with col_base:
    st.header("Baseline")
    baseline_policy = _policy_controls("b", "Current Law")

with col_reform:
    st.header("Reform")
    reform_policy = _policy_controls("r", "LVT Swap")


# ---------------------------------------------------------------------------
# Solve and display results
# ---------------------------------------------------------------------------

st.divider()
st.header("Results")

with st.spinner("Solving equilibrium..."):
    try:
        baseline_result = active_economy.solve(baseline_policy)
        reform_result   = active_economy.solve(reform_policy)
        comparison      = PolicyComparison(baseline=baseline_result, reform=reform_result)

        # Key metrics
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("GDP change",        f"{comparison.gdp_change_pct:+.2f}%")
        m2.metric("Capital change",    f"{comparison.capital_change_pct:+.2f}%")
        m3.metric("Labor change",      f"{comparison.labor_change_pct:+.2f}%")
        m4.metric("Wage change",       f"{comparison.wage_change_pct:+.2f}%")
        m5.metric("Budget balance Δ",  f"{comparison.budget_balance_change:+.2f}pp GDP",
                  help="Change in budget balance (pp of GDP). Positive = reform improves fiscal position.")

        # Revenue info
        rev_base  = baseline_result.revenue.total * 100
        rev_ref   = reform_result.revenue.total * 100
        col_a, col_b = st.columns(2)
        col_a.caption(f"Baseline revenue: **{rev_base:.1f}% GDP**  |  budget balance: {baseline_result.budget_balance*100:+.1f}%")
        col_b.caption(f"Reform revenue:   **{rev_ref:.1f}% GDP**  |  budget balance: {reform_result.budget_balance*100:+.1f}%")

        # Chart tabs
        tabs = ["Dashboard", "Macro", "Revenue", "Distributional"]
        if run_transition:
            tabs.append("Transition Path")
        chart_tabs = st.tabs(tabs)

        with chart_tabs[0]:
            fig = plot_dashboard(comparison)
            st.pyplot(fig); plt.close(fig)
        with chart_tabs[1]:
            fig = plot_macro_bars(comparison)
            st.pyplot(fig); plt.close(fig)
        with chart_tabs[2]:
            fig = plot_revenue(comparison)
            st.pyplot(fig); plt.close(fig)
        with chart_tabs[3]:
            fig = plot_incidence(comparison)
            st.pyplot(fig); plt.close(fig)

        if run_transition and len(tabs) == 5:
            with chart_tabs[4]:
                with st.spinner("Computing transition path..."):
                    from tax_model.transition import compute_transition
                    tr = compute_transition(
                        active_economy, baseline_policy, reform_policy,
                        periods=transition_periods,
                    )
                    st.caption(tr.summary())
                    fig = plot_transition(tr)
                    st.pyplot(fig); plt.close(fig)

        with st.expander("Full text summary"):
            st.code(comparison.summary(), language=None)

    except Exception as e:
        st.error(f"Solver error: {e}")
        st.caption("Try moving sliders closer to baseline values, or select a named preset.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
with st.expander("About this model"):
    st.markdown("""
**Model**: Two-period Overlapping Generations (OLG) steady-state model.

**Calibration**: US 2024. Capital share α=0.35, depreciation δ=0.10, K/Y target 2.5.
Frisch elasticity 0.5, IES 0.5. Income distribution from CBO + IRS SOI data.

**Revenue calibration**: All-in effective rates (federal + state/local) calibrated to
target ~30% of GDP total government revenue.

**Key assumption**: The *capital mobility* slider (left sidebar) controls who bears the
corporate tax. At 0 (closed economy), capital owners bear it entirely. At 100 (open economy),
workers bear it via lower wages. Most economists place the US somewhere in between.

**Presets** represent well-known tax reform proposals. Use them as starting points, not
precise policy simulations. This is an 80/20 model — directionally correct, not CBO-grade.

Source: [github.com/andybrob/macro-tax-model](https://github.com/andybrob/macro-tax-model)
    """)
