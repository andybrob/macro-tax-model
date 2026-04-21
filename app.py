"""
Macro Tax Model — Streamlit GUI
================================
Interactive policy explorer. Toggle tax levers on the left sidebar,
see macroeconomic and distributional effects update in real time.

Run:
    streamlit run app.py
"""

import sys
from pathlib import Path

# Ensure the package is importable from the project root
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import matplotlib.pyplot as plt

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
)
from tax_model.visualization import plot_dashboard, plot_macro_bars, plot_revenue, plot_incidence


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Macro Tax Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("US Tax Policy Explorer")
st.caption(
    "Two-period OLG macroeconomic model · Calibrated to US 2024 data · "
    "Long-run steady-state comparison"
)


# ---------------------------------------------------------------------------
# Load calibration (cached — only runs once per session)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_economy() -> Economy:
    cal = Calibration.from_yaml(Path(__file__).parent / "calibration" / "us_2024.yaml")
    return Economy(cal)


economy = load_economy()


# ---------------------------------------------------------------------------
# Sidebar — baseline policy controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Baseline Policy")
    st.caption("Approximate current US law")

    st.subheader("Labor Income Tax")
    base_labor_bracket_rates = {
        "10% bracket (0–0.5× median)":   st.slider("10% bracket rate", 0.0, 0.5, 0.10, 0.01, key="b_l1"),
        "22% bracket (0.5–1.5× median)": st.slider("22% bracket rate", 0.0, 0.5, 0.22, 0.01, key="b_l2"),
        "32% bracket (1.5–3× median)":   st.slider("32% bracket rate", 0.0, 0.5, 0.32, 0.01, key="b_l3"),
        "37% bracket (3×+ median)":      st.slider("37% bracket rate", 0.0, 0.6, 0.37, 0.01, key="b_l4"),
    }
    base_std_ded = st.slider("Standard deduction (× median income)", 0.0, 0.5, 0.20, 0.01, key="b_std")

    st.subheader("Corporate Tax")
    base_corp_rate = st.slider("Corporate rate", 0.0, 0.5, 0.21, 0.01, key="b_corp")
    base_expensing = st.checkbox("Immediate expensing", value=False, key="b_exp")

    st.subheader("Capital Gains Tax")
    base_cg_rate = st.slider("Capital gains rate", 0.0, 0.4, 0.20, 0.01, key="b_cg")

    st.subheader("Capital Mobility Assumption")
    capital_mobility_pct = st.slider(
        "Capital mobility (0=closed, 100=fully open)",
        0, 100, 50, 5, key="cap_mob",
        help="In a fully open economy, workers bear more of the corporate tax burden.",
    )


# ---------------------------------------------------------------------------
# Main panel — reform policy controls
# ---------------------------------------------------------------------------

col_left, col_right = st.columns([1, 1])

with col_left:
    st.header("Reform Policy")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Labor Tax", "Consumption + LVT", "Corporate", "Pigouvian", "Capital Gains"]
    )

    with tab1:
        st.subheader("Labor Income Tax")
        r_l1 = st.slider("10% bracket rate (reform)", 0.0, 0.5, 0.10, 0.01, key="r_l1")
        r_l2 = st.slider("22% bracket rate (reform)", 0.0, 0.5, 0.22, 0.01, key="r_l2")
        r_l3 = st.slider("32% bracket rate (reform)", 0.0, 0.5, 0.24, 0.01, key="r_l3")
        r_l4 = st.slider("37% bracket rate (reform)", 0.0, 0.6, 0.32, 0.01, key="r_l4")
        r_std = st.slider("Standard deduction (reform, × median)", 0.0, 0.5, 0.20, 0.01, key="r_std")

    with tab2:
        st.subheader("Consumption Tax")
        cons_rate = st.slider("Consumption tax rate", 0.0, 0.30, 0.0, 0.01, key="cons_rate")
        prebate = st.slider(
            "Prebate fraction",
            0.0, 1.0, 0.0, 0.05, key="prebate",
            help="Fraction of poverty-line consumption returned as equal per-capita cash transfer.",
        )
        st.caption(
            "Prebate = 1.0 makes the effective rate zero or negative for Q1/Q2. "
            "This is the VAT + Fee & Dividend structure."
        )

        st.subheader("Land Value Tax")
        lvt_rate = st.slider(
            "LVT rate (fraction of land value/year)",
            0.0, 0.10, 0.0, 0.005, format="%.3f", key="lvt",
            help="Land supply is perfectly inelastic — no deadweight loss. "
                 "Burden falls on land owners (concentrated at Q5_top).",
        )

    with tab3:
        st.subheader("Corporate Tax")
        r_corp_rate = st.slider("Corporate rate (reform)", 0.0, 0.5, 0.21, 0.01, key="r_corp")
        r_expensing = st.checkbox("Immediate expensing (reform)", value=False, key="r_exp")
        r_interest_ded = st.slider(
            "Interest deductibility fraction",
            0.0, 1.0, 1.0, 0.05, key="r_int",
            help="1.0 = current law. 0.0 = DBCFT (no interest deduction, equity-neutral).",
        )
        r_border_adj = st.checkbox(
            "Destination-based border adjustment (DBCFT)",
            value=False, key="r_border",
        )

    with tab4:
        st.subheader("Pigouvian Tax (e.g. Carbon)")
        pig_rate = st.slider(
            "Rate ($/ton CO₂-equivalent)",
            0.0, 300.0, 0.0, 5.0, key="pig_rate",
        )
        dividend_frac = st.slider(
            "Fee-and-dividend recycling fraction",
            0.0, 1.0, 0.0, 0.05, key="div_frac",
            help="Fraction of Pigouvian revenue returned as equal per-capita cash dividend.",
        )
        labor_offset_frac = st.slider(
            "Labor tax offset fraction",
            0.0, 1.0 - dividend_frac, 0.0, 0.05, key="lab_off",
            help="Fraction used to reduce labor income taxes.",
        )

    with tab5:
        st.subheader("Capital Gains Tax")
        r_cg_rate = st.slider("Capital gains rate (reform)", 0.0, 0.40, 0.20, 0.01, key="r_cg")
        cg_indexed = st.checkbox(
            "Inflation-indexed (tax only real gains)",
            value=False, key="cg_idx",
        )
        lock_in = st.slider(
            "Lock-in discount",
            0.0, 0.6, 0.30, 0.05, key="lock_in",
            help="Effective rate = statutory × (1 − lock_in). "
                 "Captures realization-basis deferral.",
        )

    reform_label = st.text_input("Reform label", value="Reform Policy", key="r_label")


# ---------------------------------------------------------------------------
# Build policy objects
# ---------------------------------------------------------------------------

def _labor_tax(l1, l2, l3, l4, std) -> LaborIncomeTax:
    return LaborIncomeTax(
        brackets=[
            (0.0, l1),
            (0.5, l2),
            (1.5, l3),
            (3.0, l4),
        ],
        standard_deduction_median_multiple=std,
    )


baseline_policy = TaxPolicy(
    label="Current Law (approx.)",
    labor_income=_labor_tax(
        base_labor_bracket_rates["10% bracket (0–0.5× median)"],
        base_labor_bracket_rates["22% bracket (0.5–1.5× median)"],
        base_labor_bracket_rates["32% bracket (1.5–3× median)"],
        base_labor_bracket_rates["37% bracket (3×+ median)"],
        base_std_ded,
    ),
    corporate=CorporateTax(rate=base_corp_rate, immediate_expensing=base_expensing),
    capital_gains=CapitalGainsTax(rate=base_cg_rate),
)

reform_policy = TaxPolicy(
    label=reform_label,
    labor_income=_labor_tax(r_l1, r_l2, r_l3, r_l4, r_std),
    consumption=ConsumptionTax(rate=cons_rate, prebate_fraction=prebate),
    land_value=LandValueTax(rate=lvt_rate),
    corporate=CorporateTax(
        rate=r_corp_rate,
        immediate_expensing=r_expensing,
        interest_deductibility=r_interest_ded,
        border_adjustment=r_border_adj,
    ),
    pigouvian=PigouvianTax(
        rate_per_unit=pig_rate,
        dividend_recycling_fraction=dividend_frac,
        labor_tax_offset_fraction=min(labor_offset_frac, 1.0 - dividend_frac),
    ),
    capital_gains=CapitalGainsTax(
        rate=r_cg_rate,
        inflation_indexed=cg_indexed,
        lock_in_discount=lock_in,
    ),
)

# Override capital mobility in calibration via a patched economy approach
# (We rebuild economy only if capital_mobility changes, using session state)
cal = economy.cal
if cal.macro.capital_mobility != capital_mobility_pct / 100.0:
    from tax_model.sensitivity import _set_nested
    new_cal = _set_nested(cal, "macro.capital_mobility", capital_mobility_pct / 100.0)
    active_economy = Economy(new_cal)
else:
    active_economy = economy


# ---------------------------------------------------------------------------
# Solve and display
# ---------------------------------------------------------------------------

with col_right:
    st.header("Results")

    with st.spinner("Solving equilibrium..."):
        try:
            baseline_result = active_economy.solve(baseline_policy)
            reform_result   = active_economy.solve(reform_policy)

            from tax_model.results import PolicyComparison
            comparison = PolicyComparison(baseline=baseline_result, reform=reform_result)

            # --- Key metrics row ---
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("GDP change",     f"{comparison.gdp_change_pct:+.2f}%")
            m2.metric("Capital change", f"{comparison.capital_change_pct:+.2f}%")
            m3.metric("Wage change",    f"{comparison.wage_change_pct:+.2f}%")
            m4.metric(
                "Budget balance",
                f"{comparison.budget_balance_change:+.2f}pp",
                help="Change in budget balance as percentage points of GDP. "
                     "Positive = reform improves fiscal position.",
            )

            # --- Charts ---
            chart_tabs = st.tabs(["Dashboard", "Macro", "Revenue", "Distributional"])

            with chart_tabs[0]:
                fig = plot_dashboard(comparison)
                st.pyplot(fig)
                plt.close(fig)

            with chart_tabs[1]:
                fig = plot_macro_bars(comparison)
                st.pyplot(fig)
                plt.close(fig)

            with chart_tabs[2]:
                fig = plot_revenue(comparison)
                st.pyplot(fig)
                plt.close(fig)

            with chart_tabs[3]:
                fig = plot_incidence(comparison)
                st.pyplot(fig)
                plt.close(fig)

            # --- Summary text ---
            with st.expander("Full text summary"):
                st.code(comparison.summary(), language=None)

        except Exception as e:
            st.error(f"Solver error: {e}")
            st.caption(
                "This can happen at extreme parameter combinations. "
                "Try moving sliders closer to baseline values."
            )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "Model: Two-period OLG · Cobb-Douglas production · 6 income groups "
    "(Q1–Q5_bottom, Q5_top) · Calibrated to US 2024 data · "
    "All results are long-run steady-state comparisons."
)
