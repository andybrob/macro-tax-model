"""
Macro Tax Model — Streamlit GUI
================================
Interactive US tax policy explorer with named preset scenarios.

Run:
    streamlit run app.py
"""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from tax_model import (
    Calibration, Economy, TaxPolicy,
    LaborIncomeTax, PayrollTax, ConsumptionTax, LandValueTax,
    CorporateTax, PigouvianTax, CapitalGainsTax, EstateTax,
)
from tax_model.results import PolicyComparison
from tax_model.presets import PRESETS
from tax_model.scenarios import ScenarioBundle, TFPCard, LaborSupplyCard
from tax_model.visualization import (
    plot_dashboard, plot_macro_bars, plot_revenue,
    plot_incidence, plot_transition,
    plot_instrument_incidence, plot_pareto_frontier,
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
# Revenue-neutral helper
# ---------------------------------------------------------------------------

def _make_revenue_neutral(
    reform_policy: TaxPolicy,
    target_revenue: float,
    economy: Economy,
    instrument: str,
) -> TaxPolicy:
    """
    Adjust one instrument in reform_policy until total revenue matches target_revenue.
    Returns the adjusted policy (a deep copy — does not mutate the input).
    """
    from scipy.optimize import brentq

    p = copy.deepcopy(reform_policy)

    def revenue_gap(rate: float) -> float:
        p2 = copy.deepcopy(reform_policy)
        if instrument == "Consumption rate":
            p2.consumption.rate = rate
        elif instrument == "Top income bracket":
            b = list(p2.labor_income.brackets)
            b[-1] = (b[-1][0], rate)
            p2.labor_income.brackets = b
        elif instrument == "Corporate rate":
            p2.corporate.rate = rate
        return economy.solve(p2).revenue.total - target_revenue

    lo, hi = 0.001, 0.65
    try:
        gap_lo = revenue_gap(lo)
        gap_hi = revenue_gap(hi)
        if gap_lo * gap_hi > 0:
            return reform_policy  # can't bracket — return as-is
        adjusted_rate = brentq(revenue_gap, lo, hi, xtol=1e-4, maxiter=30)
    except Exception:
        return reform_policy

    p2 = copy.deepcopy(reform_policy)
    if instrument == "Consumption rate":
        p2.consumption.rate = adjusted_rate
        p2.label += f" [RN: cons={adjusted_rate*100:.1f}%]"
    elif instrument == "Top income bracket":
        b = list(p2.labor_income.brackets)
        b[-1] = (b[-1][0], adjusted_rate)
        p2.labor_income.brackets = b
        p2.label += f" [RN: top={adjusted_rate*100:.1f}%]"
    elif instrument == "Corporate rate":
        p2.corporate.rate = adjusted_rate
        p2.label += f" [RN: corp={adjusted_rate*100:.1f}%]"
    return p2


# ---------------------------------------------------------------------------
# Preset loader helpers
# ---------------------------------------------------------------------------

def _policy_to_slider_state(policy: TaxPolicy) -> dict:
    """Extract slider values from a TaxPolicy object."""
    b = policy.labor_income.brackets
    def _rate(thresh):
        rates = [r for t, r in b if t <= thresh]
        return rates[-1] if rates else 0.0

    return {
        "l1": _rate(0.0), "l2": _rate(0.5), "l3": _rate(1.5),
        "l4": _rate(3.0), "l5": _rate(7.0),
        "std": policy.labor_income.standard_deduction_median_multiple,
        "pr_emp":  policy.payroll.employee_rate,
        "pr_er":   policy.payroll.employer_rate,
        "pr_ceil": policy.payroll.wage_ceiling_median_multiple,
        "cons_rate": policy.consumption.rate,
        "cons_ess":  policy.consumption.essentials_rate if policy.consumption.essentials_rate is not None else policy.consumption.rate,
        "cons_lux":  policy.consumption.luxury_rate if policy.consumption.luxury_rate is not None else policy.consumption.rate,
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
        "cg_sub":    policy.capital_gains.stepped_up_basis_removal_fraction,
        "est_rate":  policy.estate.rate,
        "est_exem":  policy.estate.exemption_median_multiple,
        "est_enf":   policy.estate.enforcement_fraction,
    }


def _load_preset_to_state(key: str, preset_name: str):
    """Load a named preset into session_state keys for sliders."""
    policy = PRESETS[preset_name]()
    vals = _policy_to_slider_state(policy)
    for field, val in vals.items():
        st.session_state[f"{key}_{field}"] = val


# ---------------------------------------------------------------------------
# Sidebar — global settings + structural reforms
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

    # --- Revenue Neutrality ---
    st.subheader("Revenue Neutrality")
    revenue_neutral = st.checkbox(
        "Make reform revenue-neutral",
        value=False,
        help=(
            "Auto-adjusts one instrument in the reform to match baseline revenue. "
            "Ensures you're comparing apples-to-apples, not a tax cut to a funded reform."
        ),
    )
    if revenue_neutral:
        rn_instrument = st.selectbox(
            "Adjust via",
            ["Consumption rate", "Top income bracket", "Corporate rate"],
            key="rn_instrument",
            help="Which rate to auto-adjust to achieve revenue neutrality.",
        )
    else:
        rn_instrument = "Consumption rate"

    st.divider()

    # --- Structural Reforms (reform only) ---
    st.subheader("Structural Reforms")
    st.caption("Applied to the reform scenario only. Captures supply-side effects beyond tax wedges.")

    use_prk = st.checkbox("🏫 Universal Pre-K / Early Education")
    if use_prk:
        prk_spend = st.slider("Program cost (% GDP)", 0.1, 3.0, 0.5, 0.1, key="prk_spend")
        prk_tfp   = st.slider("Long-run TFP boost (%)", 0.0, 3.0, 1.0, 0.1, key="prk_tfp",
                              help="Estimated long-run productivity gain from higher human capital.")
    else:
        prk_spend, prk_tfp = 0.0, 0.0

    use_mooc = st.checkbox("🎓 Federal College MOOCs")
    if use_mooc:
        mooc_spend = st.slider("Program cost (% GDP)", 0.1, 2.0, 0.3, 0.1, key="mooc_spend")
        mooc_tfp   = st.slider("Skill upgrade TFP effect (%)", 0.0, 2.0, 0.5, 0.1, key="mooc_tfp")
    else:
        mooc_spend, mooc_tfp = 0.0, 0.0

    use_h1b = st.checkbox("✈️ H1B / High-Skill Immigration Reform")
    if use_h1b:
        h1b_mult = st.slider("High-skill labor increase (%)", 1.0, 30.0, 10.0, 1.0, key="h1b_mult",
                             help="+10% = 10% more high-skill workers. Modeled as TFP boost via knowledge spillovers.")
        h1b_spill = st.slider("Knowledge spillover TFP (%)", 0.0, 2.0, 0.5, 0.1, key="h1b_spill")
    else:
        h1b_mult, h1b_spill = 0.0, 0.0

    # Build ScenarioBundle from sidebar inputs
    tfp_card = None
    if use_prk or use_mooc:
        total_spend = prk_spend / 100.0 + mooc_spend / 100.0
        total_tfp_mult = (1.0 + prk_tfp / 100.0) * (1.0 + mooc_tfp / 100.0)
        tfp_card = TFPCard(
            name="Human Capital Investment",
            spend_gdp_fraction=total_spend,
            long_run_tfp_multiplier=total_tfp_mult,
        )

    labor_card = None
    if use_h1b:
        labor_card = LaborSupplyCard(
            name="H1B Reform",
            high_skill_labor_multiplier=1.0 + h1b_mult / 100.0,
            tfp_spillover=h1b_spill / 100.0,
        )

    reform_scenario = ScenarioBundle(tfp_card=tfp_card, labor_card=labor_card)
    if reform_scenario.is_empty():
        reform_scenario = None

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
    preset_names = ["(Custom)"] + list(PRESETS.keys())
    selected = st.selectbox(
        "Load preset", preset_names, key=f"{key}_preset",
        index=preset_names.index(default_preset) if default_preset in preset_names else 0,
    )
    if selected != "(Custom)" and st.button("Apply preset", key=f"{key}_apply"):
        _load_preset_to_state(key, selected)
        st.rerun()

    def sv(field, default):
        return st.session_state.get(f"{key}_{field}", default)

    tab_labor, tab_payroll, tab_cons, tab_corp, tab_pig, tab_cg, tab_estate = st.tabs([
        "Income Tax", "Payroll", "Consumption + LVT", "Corporate", "Pigouvian", "Cap. Gains", "Estate Tax"
    ])

    with tab_labor:
        l1  = st.slider("Bracket 1 rate (0–0.5× median)",   0.0, 0.5, sv("l1", 0.15), 0.01, key=f"{key}_l1")
        l2  = st.slider("Bracket 2 rate (0.5–1.5× median)", 0.0, 0.5, sv("l2", 0.27), 0.01, key=f"{key}_l2")
        l3  = st.slider("Bracket 3 rate (1.5–3× median)",   0.0, 0.5, sv("l3", 0.29), 0.01, key=f"{key}_l3")
        l4  = st.slider("Bracket 4 rate (3–7× median)",     0.0, 0.6, sv("l4", 0.37), 0.01, key=f"{key}_l4")
        l5  = st.slider("Top bracket rate (7×+ median)",    0.0, 0.6, sv("l5", 0.42), 0.01, key=f"{key}_l5")
        std = st.slider("Standard deduction (× median)",    0.0, 0.6, sv("std", 0.20), 0.01, key=f"{key}_std")

    with tab_payroll:
        st.caption("FICA (Social Security + Medicare). Both portions fall on workers in the long run.")
        pr_emp  = st.slider("Employee rate",  0.0, 0.15, sv("pr_emp",  0.0765), 0.005, format="%.3f", key=f"{key}_pr_emp")
        pr_er   = st.slider("Employer rate",  0.0, 0.15, sv("pr_er",   0.0765), 0.005, format="%.3f", key=f"{key}_pr_er")
        pr_ceil = st.slider("SS wage ceiling (× median)", 0.5, 5.0, sv("pr_ceil", 2.8), 0.1, key=f"{key}_pr_ceil")
        st.caption(f"Combined rate: {(pr_emp+pr_er)*100:.1f}%  |  Medicare (2.9%) continues above ceiling")

    with tab_cons:
        cons_rate = st.slider("Standard rate", 0.0, 0.35, sv("cons_rate", 0.02), 0.01, key=f"{key}_cons_rate")
        use_tiers = st.checkbox("Tiered rates (France-style VAT)", key=f"{key}_use_tiers")
        if use_tiers:
            cons_ess = st.slider("Essentials rate (food, medicine)", 0.0, cons_rate, sv("cons_ess", max(0.0, cons_rate - 0.05)), 0.01, key=f"{key}_cons_ess",
                                 help="Q1 spends ~55% of budget on essentials; Q5_top only ~15%.")
            cons_lux = st.slider("Luxury rate", cons_rate, 0.40, sv("cons_lux", min(0.40, cons_rate + 0.05)), 0.01, key=f"{key}_cons_lux",
                                 help="Q5_top spends ~35% of budget on luxury goods; Q1 only ~3%.")
        else:
            cons_ess = None
            cons_lux = None
        prebate = st.slider("Prebate fraction", 0.0, 1.0, sv("prebate", 0.0), 0.05, key=f"{key}_prebate",
                            help="Fraction of poverty-line consumption returned as equal per-capita cash transfer.")
        st.divider()
        lvt = st.slider("Land Value Tax rate (fraction of land value/yr)",
                        0.0, 0.10, sv("lvt", 0.0), 0.005, format="%.3f", key=f"{key}_lvt",
                        help="Zero deadweight loss — land supply is perfectly inelastic.")

    with tab_corp:
        corp      = st.slider("Corporate rate",           0.0, 0.50, sv("corp",     0.21), 0.01, key=f"{key}_corp")
        expensing = st.checkbox("Immediate expensing",                sv("expensing", False),      key=f"{key}_expensing")
        int_ded   = st.slider("Interest deductibility",  0.0, 1.0,  sv("int_ded",  1.0),  0.05, key=f"{key}_int_ded",
                              help="1.0 = current law; 0.0 = DBCFT (equity-neutral)")
        border    = st.checkbox("Destination-based border adjustment", sv("border",  False),       key=f"{key}_border")

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
        cg_sub  = st.slider("Stepped-up basis removal", 0.0, 1.0, sv("cg_sub", 0.0), 0.05, key=f"{key}_cg_sub",
                            help=(
                                "0 = current law (heirs inherit FMV basis; gains permanently exempt).\n"
                                "1 = full removal (heirs inherit original cost basis; reduces lock-in)."
                            ))
        eff = cg_rate * (1.0 - lock_in * (1.0 - 0.60 * cg_sub))
        st.caption(f"Effective rate after lock-in & step-up adjustment: **{eff*100:.1f}%**")

    with tab_estate:
        st.caption(
            "Estate tax on large inheritances. Falls almost entirely on Q5_top (top 1%). "
            "Revenue ~0.1% GDP at current rates; could reach 0.5%+ with higher rates + enforcement."
        )
        est_rate = st.slider("Top marginal rate",             0.0, 0.80, sv("est_rate", 0.0),   0.05, key=f"{key}_est_rate")
        est_exem = st.slider("Exemption (× median income)",  10.0, 300.0, sv("est_exem", 170.0), 10.0, key=f"{key}_est_exem",
                             help="US 2024 exemption ≈ $13.6M ≈ 170× median income.")
        est_enf  = st.slider("Enforcement fraction",          0.10, 0.90, sv("est_enf",  0.30), 0.05, key=f"{key}_est_enf",
                             help="Current law ~30% (trusts, GRATs, valuation discounts). Better enforcement → higher fraction.")
        if est_rate > 0:
            approx_rev = est_rate * est_enf * 0.70 * 0.40 * 2.5 * 0.35  # rough order-of-magnitude
            st.caption(f"Approximate revenue: ~{approx_rev*100:.2f}% GDP")

    label = st.text_input(
        "Policy label",
        value=selected if selected != "(Custom)" else f"{key.upper()} Policy",
        key=f"{key}_label",
    )

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
        consumption=ConsumptionTax(
            rate=cons_rate,
            prebate_fraction=prebate,
            essentials_rate=cons_ess,
            luxury_rate=cons_lux,
        ),
        land_value=LandValueTax(rate=lvt),
        corporate=CorporateTax(
            rate=corp,
            immediate_expensing=expensing,
            interest_deductibility=int_ded,
            border_adjustment=border,
        ),
        pigouvian=PigouvianTax(
            rate_per_unit=pig_rate,
            dividend_recycling_fraction=div_frac,
            labor_tax_offset_fraction=min(lab_off, 1.0 - div_frac),
        ),
        capital_gains=CapitalGainsTax(
            rate=cg_rate,
            inflation_indexed=cg_idx,
            lock_in_discount=lock_in,
            stepped_up_basis_removal_fraction=cg_sub,
        ),
        estate=EstateTax(
            rate=est_rate,
            exemption_median_multiple=est_exem,
            enforcement_fraction=est_enf,
        ),
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
        # Always solve baseline first (revenue-neutral needs baseline revenue)
        baseline_result = active_economy.solve(baseline_policy)

        # Apply revenue-neutral adjustment to reform if requested
        if revenue_neutral:
            reform_policy = _make_revenue_neutral(
                reform_policy,
                target_revenue=baseline_result.revenue.total,
                economy=active_economy,
                instrument=rn_instrument,
            )
            if reform_scenario:
                st.info(
                    f"Revenue-neutral: auto-adjusted **{rn_instrument}** to match baseline revenue "
                    f"({baseline_result.revenue.total*100:.1f}% GDP)."
                )

        reform_result = active_economy.solve(reform_policy, scenario=reform_scenario)
        comparison    = PolicyComparison(baseline=baseline_result, reform=reform_result)

        # --- Scenario card info banner ---
        if reform_scenario is not None:
            parts = []
            if reform_scenario.tfp_card is not None:
                t = reform_scenario.tfp_card
                parts.append(f"**{t.name}**: {t.spend_gdp_fraction*100:.1f}% GDP cost, {(t.long_run_tfp_multiplier-1)*100:.1f}% TFP boost")
            if reform_scenario.labor_card is not None:
                lc = reform_scenario.labor_card
                parts.append(f"**{lc.name}**: +{(lc.high_skill_labor_multiplier-1)*100:.0f}% high-skill labor, +{lc.tfp_spillover*100:.1f}% TFP spillover")
            st.info("Structural reforms applied to reform: " + " · ".join(parts))

        # --- Key metrics ---
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("GDP change",       f"{comparison.gdp_change_pct:+.2f}%")
        m2.metric("Capital change",   f"{comparison.capital_change_pct:+.2f}%")
        m3.metric("Labor change",     f"{comparison.labor_change_pct:+.2f}%")
        m4.metric("Wage change",      f"{comparison.wage_change_pct:+.2f}%")
        m5.metric("Budget balance Δ", f"{comparison.budget_balance_change:+.2f}pp GDP",
                  help="Change in budget balance (pp of GDP). Positive = reform improves fiscal position.")

        # --- Revenue info ---
        rev_base = baseline_result.revenue.total * 100
        rev_ref  = reform_result.revenue.total * 100
        col_a, col_b = st.columns(2)
        col_a.caption(f"Baseline revenue: **{rev_base:.1f}% GDP**  |  budget balance: {baseline_result.budget_balance*100:+.1f}%")
        col_b.caption(f"Reform revenue:   **{rev_ref:.1f}% GDP**  |  budget balance: {reform_result.budget_balance*100:+.1f}%")

        # --- CSV export ---
        from tax_model.calibration import GROUPS
        import pandas as pd
        rows = [
            {"Category": "Macro", "Item": "GDP (normalized)",        "Baseline": round(baseline_result.gdp, 4),            "Reform": round(reform_result.gdp, 4)},
            {"Category": "Macro", "Item": "Capital (normalized)",     "Baseline": round(baseline_result.capital_stock, 4),  "Reform": round(reform_result.capital_stock, 4)},
            {"Category": "Macro", "Item": "Labor (normalized)",       "Baseline": round(baseline_result.labor_supply, 4),   "Reform": round(reform_result.labor_supply, 4)},
            {"Category": "Macro", "Item": "Wage (normalized)",        "Baseline": round(baseline_result.wage, 4),           "Reform": round(reform_result.wage, 4)},
            {"Category": "Macro", "Item": "Revenue (% GDP)",          "Baseline": round(rev_base, 2),                       "Reform": round(rev_ref, 2)},
            {"Category": "Macro", "Item": "Budget balance (% GDP)",   "Baseline": round(baseline_result.budget_balance*100, 2), "Reform": round(reform_result.budget_balance*100, 2)},
        ]
        for g in GROUPS:
            rows.append({
                "Category": "Incidence (% of group income)",
                "Item": g,
                "Baseline": round(baseline_result.incidence[g] * 100, 2),
                "Reform":   round(reform_result.incidence[g] * 100, 2),
            })
        df_export = pd.DataFrame(rows)
        df_export["Change"] = (df_export["Reform"] - df_export["Baseline"]).round(4)
        csv_bytes = df_export.to_csv(index=False).encode()
        fname = (f"tax_{baseline_result.policy_label}_vs_{reform_result.policy_label}.csv"
                 .replace(" ", "_").replace("/", "-").replace("[", "").replace("]", ""))
        st.download_button("⬇ Download comparison (CSV)", data=csv_bytes, file_name=fname, mime="text/csv")

        # --- Chart tabs ---
        tabs = ["Dashboard", "Macro", "Revenue", "Distributional", "By Instrument", "Frontier"]
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

        with chart_tabs[4]:
            from tax_model.government import compute_instrument_incidence
            from tax_model.government import Allocation, compute_revenue as _cr
            # Recompute allocations for per-instrument breakdown
            # (Use the already-solved results directly — reconstruct alloc from ModelResult)
            # Since alloc isn't stored on ModelResult, call the breakdown using revenue + policy
            # We approximate alloc fields from ModelResult scalars for the breakdown
            from tax_model.government import _GROUP_INCOME_MULTIPLES
            from dataclasses import dataclass

            def _make_alloc_from_result(res, cal):
                return Allocation(
                    gdp=res.gdp,
                    capital_stock=res.capital_stock,
                    labor_supply=res.labor_supply,
                    consumption=res.consumption,
                    investment=res.investment,
                    wage=res.wage,
                    return_on_capital=res.return_on_capital,
                    land_rent=cal.land.land_rent_gdp_ratio * res.gdp,
                    externality_quantity=cal.externality.carbon_intensity_per_gdp * res.gdp,
                )

            cal = active_economy.cal
            base_alloc   = _make_alloc_from_result(baseline_result, cal)
            reform_alloc = _make_alloc_from_result(reform_result, cal)

            base_breakdown   = compute_instrument_incidence(base_alloc,   baseline_policy, baseline_result.revenue, cal)
            reform_breakdown = compute_instrument_incidence(reform_alloc,  reform_policy,   reform_result.revenue,  cal)

            fig = plot_instrument_incidence(
                base_breakdown, reform_breakdown,
                baseline_label=baseline_result.policy_label,
                reform_label=reform_result.policy_label,
            )
            st.pyplot(fig); plt.close(fig)
            st.caption(
                "Each bar shows that instrument's contribution to a group's total effective tax rate. "
                "Stacked height ≈ total burden. Prebate offsets show as negative (may be hidden in stack)."
            )

        with chart_tabs[5]:
            st.caption(
                "All 8 named presets vs. Current Law. "
                "**Green** = Q1 better off than current law. **↗ corner** = Pareto-improving (higher GDP AND better for Q1)."
            )
            with st.spinner("Computing frontier (all 8 presets)..."):
                frontier_data = []
                baseline_preset = PRESETS["Current Law"]()
                baseline_for_frontier = active_economy.solve(baseline_preset)
                for name, factory in PRESETS.items():
                    if name == "Current Law":
                        continue
                    try:
                        ref_result = active_economy.solve(factory())
                        comp_f = PolicyComparison(baseline=baseline_for_frontier, reform=ref_result)
                        q1_change = (ref_result.incidence["Q1"] - baseline_for_frontier.incidence["Q1"]) * 100
                        frontier_data.append((name, comp_f.gdp_change_pct, q1_change))
                    except Exception:
                        pass
            fig = plot_pareto_frontier(frontier_data)
            st.pyplot(fig); plt.close(fig)

        if run_transition:
            with chart_tabs[6]:
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

        # --- Monte Carlo uncertainty ---
        with st.expander("Model uncertainty (Monte Carlo sensitivity)"):
            st.caption(
                "90% credible intervals across calibration parameter uncertainty (N=50 draws). "
                "Vary: Frisch elasticity, IES, saving sensitivity, capital share, capital mobility, ETI."
            )
            if st.button("Run sensitivity analysis (~10 sec)", key="run_mc"):
                from tax_model.sensitivity import monte_carlo, credible_interval
                with st.spinner("Running Monte Carlo (N=50)..."):
                    mc_comps = monte_carlo(active_economy.cal, baseline_policy, reform_policy, n_draws=50, seed=42)
                col1, col2, col3 = st.columns(3)
                gdp_ci  = credible_interval(mc_comps, "gdp_change_pct",        lower=5.0, upper=95.0)
                cap_ci  = credible_interval(mc_comps, "capital_change_pct",     lower=5.0, upper=95.0)
                wage_ci = credible_interval(mc_comps, "wage_change_pct",        lower=5.0, upper=95.0)
                bud_ci  = credible_interval(mc_comps, "budget_balance_change",  lower=5.0, upper=95.0)
                col1.metric("GDP change 90% CI",       f"[{gdp_ci['lower']:+.2f}%, {gdp_ci['upper']:+.2f}%]",
                            delta=f"median {gdp_ci['mean']:+.2f}%")
                col2.metric("Capital change 90% CI",   f"[{cap_ci['lower']:+.2f}%, {cap_ci['upper']:+.2f}%]",
                            delta=f"median {cap_ci['mean']:+.2f}%")
                col3.metric("Wage change 90% CI",      f"[{wage_ci['lower']:+.2f}%, {wage_ci['upper']:+.2f}%]",
                            delta=f"median {wage_ci['mean']:+.2f}%")
                st.caption(
                    f"Budget balance 90% CI: [{bud_ci['lower']:+.2f}, {bud_ci['upper']:+.2f}] pp GDP  |  "
                    f"median: {bud_ci['mean']:+.2f} pp GDP"
                )

    except Exception as e:
        import traceback
        st.error(f"Solver error: {e}")
        st.caption("Try moving sliders closer to baseline values, or select a named preset.")
        with st.expander("Debug traceback"):
            st.code(traceback.format_exc())


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

**Key assumption**: The *capital mobility* slider controls who bears the corporate tax.
At 0 (closed economy), capital owners bear it entirely. At 100 (open economy), workers
bear it via lower wages. Most economists place the US somewhere in between.

**New in v2**: Estate tax, stepped-up basis removal, tiered VAT (France-style),
revenue-neutral toggle, structural reform cards (Pre-K, MOOCs, H1B), per-instrument
incidence breakdown, growth–equity frontier, transition debt path, Monte Carlo CI.

**Presets** represent well-known tax reform proposals. This is an 80/20 model —
directionally correct, not CBO-grade.

Source: [github.com/andybrob/macro-tax-model](https://github.com/andybrob/macro-tax-model)
    """)
