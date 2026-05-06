"""
Iran-Energy Impact Analyzer — Streamlit Dashboard
==================================================
Analyzes the geopolitical impact of the 2026 Iran-USA conflict on German
energy markets and models business implications for renewable forecasting
companies like Enercast.

Run: streamlit run app/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="Iran-Energy Impact Analyzer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Colour palette ────────────────────────────────────────────────────────────
COLORS = {
    "pre":       "#4A90D9",
    "acute":     "#E67E22",
    "crisis":    "#C0392B",
    "ceasefire": "#27AE60",
    "ttf":       "#8E44AD",
    "brent":     "#2C3E50",
    "scenario_A": "#27AE60",
    "scenario_B": "#C0392B",
    "scenario_C": "#3498DB",
}

PHASE_COLORS = {
    "pre_conflict":      "#4A90D9",
    "acute_shock":       "#E67E22",
    "sustained_crisis":  "#C0392B",
    "post_ceasefire":    "#27AE60",
}

CONFLICT_START = pd.Timestamp("2026-02-28")
CEASEFIRE      = pd.Timestamp("2026-04-08")

KEY_EVENTS = {
    "2026-02-28": ("Conflict starts", "#E74C3C"),
    "2026-03-04": ("Qatar LNG attack", "#E67E22"),
    "2026-03-15": ("TTF doubles", "#9B59B6"),
    "2026-04-08": ("Ceasefire", "#27AE60"),
}


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_all_data():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    master     = pd.read_csv(f"{base}/data/raw/master_dataset.csv",       parse_dates=["date"])
    ev_ts      = pd.read_csv(f"{base}/data/processed/event_study_timeseries.csv", parse_dates=["date"])
    ev_stats   = pd.read_csv(f"{base}/data/processed/event_study_stats.csv")
    phase_comp = pd.read_csv(f"{base}/data/processed/phase_comparison.csv")
    scenarios  = pd.read_csv(f"{base}/data/processed/scenario_forecasts.csv",   parse_dates=["date"])
    impact     = pd.read_csv(f"{base}/data/processed/enercast_business_impact.csv", parse_dates=["date"])
    timeline   = pd.read_csv(f"{base}/data/raw/conflict_timeline.csv",      parse_dates=["date"])
    premium    = pd.read_csv(f"{base}/data/processed/renewable_premium.csv",     parse_dates=["date"])
    return master, ev_ts, ev_stats, phase_comp, scenarios, impact, timeline, premium

master, ev_ts, ev_stats, phase_comp, scenarios, impact, timeline, premium = load_all_data()


def add_event_lines(fig, row=1, col=1):
    for date_str, (label, color) in KEY_EVENTS.items():
        fig.add_shape(
            type="line",
            x0=date_str, x1=date_str,
            y0=0, y1=1,
            xref=f"x{'' if (row==1 and col==1) else row}",
            yref=f"y{'' if (row==1 and col==1) else row} domain",
            line=dict(dash="dash", color=color, width=1.5),
        )
        fig.add_annotation(
            x=date_str,
            y=1,
            xref=f"x{'' if (row==1 and col==1) else row}",
            yref=f"y{'' if (row==1 and col==1) else row} domain",
            text=label,
            showarrow=False,
            font=dict(size=9, color=color),
            textangle=-90,
            xanchor="left",
            yanchor="top",
        )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/Above_Gotland.jpg/640px-Above_Gotland.jpg",
             use_container_width=True)
    st.title("⚡ Iran–Energy\nImpact Analyzer")
    st.markdown("**Geopolitical DS Portfolio Project**")
    st.markdown("---")
    st.markdown("""
**Author:** Gaurav Bhatia  
**Tools:** Python · XGBoost · Streamlit  
**Data:** SMARD · ENTSO-E · EIA · Open-Meteo  
**Methods:** Event study · Scenario forecasting
    """)
    st.markdown("---")
    page = st.radio("Navigate", [
        "📊 Overview & Timeline",
        "📉 Event Study",
        "🔮 Scenario Forecasts",
        "🏭 Enercast Business Impact",
        "📖 Methodology"
    ])
    st.markdown("---")
    st.caption("2026 Iran-USA conflict · German energy markets")


# ═══════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW & TIMELINE
# ═══════════════════════════════════════════════════════════════════════
if page == "📊 Overview & Timeline":

    st.title("🌍 Geopolitical Shock — German Energy Markets")
    st.markdown(
        "Analyzing the impact of the **2026 Iran-USA conflict** on German electricity, "
        "gas, and oil prices — and what it means for renewable energy forecasting companies like **Enercast**."
    )

    # ── KPI cards
    pre  = phase_comp[phase_comp["phase"] == "pre_conflict"].iloc[0]
    cris = phase_comp[phase_comp["phase"] == "sustained_crisis"].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Electricity price spike",
                f"€{cris['de_price_eur_mwh_mean']:.0f}/MWh",
                f"+{cris['de_price_eur_mwh_pct_change']:.0f}% vs baseline")
    col2.metric("TTF gas benchmark",
                f"€{cris['ttf_eur_mwh_mean']:.0f}/MWh",
                f"+{phase_comp[phase_comp['phase']=='sustained_crisis']['ttf_eur_mwh_pct_change'].iloc[0]:.0f}%")
    col3.metric("Brent crude peak",
                f"${master['brent_usd_bbl'].max():.0f}/bbl",
                f"+{(master['brent_usd_bbl'].max() / master[master['conflict_phase']=='pre_conflict']['brent_usd_bbl'].mean() - 1)*100:.0f}%")
    col4.metric("Enercast value uplift",
                "€8.2M/day",
                "per 1% forecast improvement")

    st.markdown("---")

    # ── Main price chart
    st.subheader("German Energy Price Dashboard")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4],
                        subplot_titles=["German day-ahead electricity price (€/MWh)",
                                        "Supporting commodities"])

    # Color by phase
    for phase, color in PHASE_COLORS.items():
        subset = master[master["conflict_phase"] == phase]
        fig.add_trace(go.Scatter(
            x=subset["date"], y=subset["de_price_eur_mwh"],
            mode="lines", name=phase.replace("_", " ").title(),
            line=dict(color=color, width=2),
            fill="tozeroy", fillcolor="rgba(74,144,217,0.08)",
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=master["date"], y=master["ttf_eur_mwh"],
        name="TTF gas (€/MWh)", line=dict(color=COLORS["ttf"], width=1.5, dash="dot")
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=master["date"], y=master["brent_usd_bbl"],
        name="Brent crude ($/bbl)", line=dict(color=COLORS["brent"], width=1.5)
    ), row=2, col=1)

    add_event_lines(fig, row=1, col=1)
    fig.update_layout(height=550, showlegend=True,
                      legend=dict(orientation="h", y=-0.15),
                      hovermode="x unified")
    fig.update_yaxes(title_text="€/MWh", row=1, col=1)
    fig.update_yaxes(title_text="Price", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # ── Phase comparison table
    st.subheader("Phase Comparison")
    display = phase_comp[["phase", "n_days",
                           "de_price_eur_mwh_mean", "de_price_eur_mwh_pct_change",
                           "ttf_eur_mwh_mean", "brent_usd_bbl_mean"]].copy()
    display.columns = ["Phase", "Days", "Electricity (€/MWh)", "Elec. Change %",
                        "TTF Gas (€/MWh)", "Brent ($/bbl)"]
    display["Phase"] = display["Phase"].str.replace("_", " ").str.title()
    st.dataframe(display.style.background_gradient(subset=["Elec. Change %"], cmap="RdYlGn_r"),
                 use_container_width=True, hide_index=True)

    # ── Timeline
    st.subheader("Conflict Event Timeline")
    for _, row in timeline.iterrows():
        sev_color = {"critical": "🔴", "high": "🟠", "medium": "🟡"}.get(row["severity"], "⚪")
        st.markdown(f"{sev_color} **{row['date'].strftime('%d %b %Y')}** — {row['event']}")


# ═══════════════════════════════════════════════════════════════════════
# PAGE 2 — EVENT STUDY
# ═══════════════════════════════════════════════════════════════════════
elif page == "📉 Event Study":

    st.title("📉 Event Study Analysis")
    st.markdown("""
    **Methodology:** Standard financial event study — measures *abnormal* price movements
    attributable to the conflict, controlling for pre-existing trends.
    - **Estimation window:** 90 days pre-conflict (baseline trend)
    - **Event window:** −10 to +60 days around Feb 28 conflict start
    - **Abnormal Return (AR):** Actual price − predicted "normal" price
    - **CAR:** Cumulative Abnormal Return
    """)

    # ── Stats summary
    st.subheader("Statistical Significance")
    col1, col2, col3 = st.columns(3)
    for col, (_, row) in zip([col1, col2, col3], ev_stats.iterrows()):
        with col:
            sig_badge = "✅ Significant" if row["significant"] else "❌ Not significant"
            st.markdown(f"**{row['label']}**")
            st.metric("Mean Abnormal Return", f"{row['mean_AR']:+.2f}")
            st.metric("Total CAR", f"{row['total_CAR']:+.2f}")
            st.metric("p-value", f"{row['p_value']:.4f}")
            st.markdown(f"Cohen's d: **{row['cohens_d']:.2f}** | {sig_badge}")
            st.markdown(f"Effect: {'Large' if abs(row['cohens_d']) > 0.8 else 'Medium' if abs(row['cohens_d']) > 0.5 else 'Small'}")

    st.markdown("---")

    # ── CAR chart
    metric_choice = st.selectbox("Select metric", ev_stats["metric"].tolist(),
                                  format_func=lambda x: ev_stats[ev_stats["metric"]==x]["label"].values[0])

    ev_subset = ev_ts[ev_ts["metric"] == metric_choice]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=["Actual vs Normal price",
                                        "Cumulative Abnormal Return (CAR)"])

    fig.add_trace(go.Scatter(x=ev_subset["date"], y=ev_subset[metric_choice],
                             name="Actual", line=dict(color="#E74C3C", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=ev_subset["date"], y=ev_subset["normal_price"],
                             name="Normal (expected)", line=dict(color="#3498DB", width=2, dash="dash")), row=1, col=1)
    fig.add_trace(go.Bar(x=ev_subset["date"], y=ev_subset["abnormal_return"],
                         name="Abnormal Return",
                         marker_color=ev_subset["abnormal_return"].apply(
                             lambda v: "#C0392B" if v > 0 else "#27AE60")), row=1, col=1)
    fig.add_trace(go.Scatter(x=ev_subset["date"], y=ev_subset["CAR"],
                             name="CAR", fill="tozeroy",
                             line=dict(color="#E67E22", width=2),
                             fillcolor="rgba(230,126,34,0.15)"), row=2, col=1)

    fig.add_shape(type="line", x0="2026-02-28", x1="2026-02-28", y0=0, y1=1,
                  xref="x", yref="y domain", line=dict(dash="dash", color="#E74C3C", width=1.5))
    fig.add_annotation(x="2026-02-28", y=1, xref="x", yref="y domain",
                       text="Conflict Start", showarrow=False,
                       font=dict(size=9, color="#E74C3C"), textangle=-90, xanchor="left")
    fig.update_layout(height=550, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# PAGE 3 — SCENARIO FORECASTS
# ═══════════════════════════════════════════════════════════════════════
elif page == "🔮 Scenario Forecasts":

    st.title("🔮 Geopolitical Scenario Forecasts")
    st.markdown("**90-day electricity price forecast (May–Jul 2026)** under 3 geopolitical scenarios")

    sc_colors = {
        "A_rapid_resolution":        ("#27AE60", "Scenario A: Rapid resolution"),
        "B_prolonged_crisis":         ("#C0392B", "Scenario B: Prolonged crisis"),
        "C_renewables_acceleration":  ("#3498DB", "Scenario C: Renewables surge"),
    }

    # ── Scenario metrics
    col1, col2, col3 = st.columns(3)
    for col, (sc_key, (color, label)) in zip([col1, col2, col3], sc_colors.items()):
        s = scenarios[scenarios["scenario"] == sc_key]
        with col:
            avg = s["predicted_price_eur_mwh"].mean()
            end = s["predicted_price_eur_mwh"].iloc[-1]
            st.markdown(f"<div style='border-left: 4px solid {color}; padding-left: 10px;'>", unsafe_allow_html=True)
            st.markdown(f"**{label}**")
            st.metric("90-day avg", f"€{avg:.0f}/MWh")
            st.metric("End price", f"€{end:.0f}/MWh")
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    fig = go.Figure()

    # Historical context
    hist = master[master["date"] >= pd.Timestamp("2026-01-01")]
    fig.add_trace(go.Scatter(
        x=hist["date"], y=hist["de_price_eur_mwh"],
        name="Historical (actual)", line=dict(color="#2C3E50", width=2.5),
        mode="lines"
    ))

    for sc_key, (color, label) in sc_colors.items():
        s = scenarios[scenarios["scenario"] == sc_key]
        fig.add_trace(go.Scatter(
            x=s["date"], y=s["predicted_price_eur_mwh"],
            name=label, line=dict(color=color, width=2, dash="dash"),
            mode="lines"
        ))
        fig.add_trace(go.Scatter(
            x=pd.concat([s["date"], s["date"].iloc[::-1]]),
            y=pd.concat([
                s["predicted_price_eur_mwh"] + 12,
                (s["predicted_price_eur_mwh"] - 12).iloc[::-1]
            ]),
            fill="toself", fillcolor="rgba(100,100,100,0.1)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False, name=f"{label} CI"
        ))

    fig.add_shape(type="line", x0="2026-05-01", x1="2026-05-01", y0=0, y1=1,
                  xref="x", yref="y domain", line=dict(dash="dot", color="#888", width=1.5))
    fig.add_annotation(x="2026-05-01", y=1, xref="x", yref="y domain",
                       text="Forecast start", showarrow=False,
                       font=dict(size=9, color="#888"), textangle=-90, xanchor="left")
    fig.add_shape(type="line", x0="2026-02-28", x1="2026-02-28", y0=0, y1=1,
                  xref="x", yref="y domain", line=dict(dash="dash", color="#E74C3C", width=1.5))
    fig.add_annotation(x="2026-02-28", y=1, xref="x", yref="y domain",
                       text="Conflict", showarrow=False,
                       font=dict(size=9, color="#E74C3C"), textangle=-90, xanchor="left")

    fig.update_layout(
        title="German electricity price — historical + 3 scenarios",
        yaxis_title="€/MWh", height=480,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.2)
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Assumption table
    st.subheader("Scenario assumptions")
    assump = pd.DataFrame({
        "Scenario": ["A: Rapid resolution", "B: Prolonged crisis", "C: Renewables surge"],
        "Hormuz status": ["Fully reopens by Jun", "Remains restricted", "Partially open"],
        "Qatar LNG": ["Production recovers", "Damage persists 3-5 yrs", "Partial recovery"],
        "TTF target (€/MWh)": ["€45 by Jul", "€60-65 plateau", "€48 (renewables offset)"],
        "Brent target ($/bbl)": ["$75", "$90+", "$72"],
        "Enercast tailwind": ["Moderate", "Strong", "Very Strong"],
    })
    st.dataframe(assump, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════
# PAGE 4 — ENERCAST BUSINESS IMPACT
# ═══════════════════════════════════════════════════════════════════════
elif page == "🏭 Enercast Business Impact":

    st.title("🏭 Enercast Business Impact Analysis")
    st.markdown("""
    **Enercast** (Kassel, Germany) delivers AI-powered wind & solar forecasting to 240 GW of
    capacity across 30 countries. Higher electricity price volatility = higher value of accurate forecasting.

    This section models how the conflict amplifies the business value of Enercast's platform.
    """)

    # ── Renewable premium during crisis
    cris_prem = premium[premium["conflict_phase"].isin(["acute_shock", "sustained_crisis"])]
    pre_prem  = premium[premium["conflict_phase"] == "pre_conflict"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg avoided cost/day (crisis)", f"€{cris_prem['avoided_cost_eur_day'].mean()/1e6:.1f}M")
    col2.metric("Pre-crisis baseline", f"€{pre_prem['avoided_cost_eur_day'].mean()/1e6:.1f}M/day")
    col3.metric("Enercast value uplift (crisis)", f"€{cris_prem['enercast_value_improvement_eur_day'].mean():.1f}M/day")
    col4.metric("Total 39-day crisis value", f"€{cris_prem['enercast_value_improvement_eur_day'].sum():.0f}M")

    st.markdown("---")

    # ── Value uplift over time
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=[
                            "Daily value of 1% MAPE improvement — Enercast 240 GW portfolio (€M)",
                            "Electricity price driving the value"
                        ])

    fig.add_trace(go.Scatter(
        x=premium["date"], y=premium["enercast_value_improvement_eur_day"],
        name="Enercast daily value (€M)", fill="tozeroy",
        line=dict(color="#E67E22", width=2),
        fillcolor="rgba(230,126,34,0.2)"
    ), row=1, col=1)

    for phase, color in PHASE_COLORS.items():
        s = premium[premium["conflict_phase"] == phase]
        fig.add_trace(go.Scatter(
            x=s["date"], y=s["de_price_eur_mwh"],
            name=phase.replace("_", " ").title(),
            line=dict(color=color, width=2),
            mode="lines"
        ), row=2, col=1)

    add_event_lines(fig, row=1, col=1)
    fig.update_layout(height=500, hovermode="x unified",
                      legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, use_container_width=True)

    # ── Scenario impact
    st.subheader("Forward-looking: value under 3 scenarios (May–Jul 2026)")

    sc_colors_map = {
        "A_rapid_resolution":       ("#27AE60", "Scenario A: Rapid resolution"),
        "B_prolonged_crisis":        ("#C0392B", "Scenario B: Prolonged crisis"),
        "C_renewables_acceleration": ("#3498DB", "Scenario C: Renewables surge"),
    }

    fig2 = go.Figure()
    for sc_key, (color, label) in sc_colors_map.items():
        s = impact[impact["scenario"] == sc_key]
        fig2.add_trace(go.Scatter(
            x=s["date"], y=s["enercast_value_M_eur_per_day"],
            name=label, line=dict(color=color, width=2)
        ))

    fig2.update_layout(
        title="Enercast daily value uplift per scenario (€M/day per 1% MAPE improvement)",
        yaxis_title="€M / day", height=380,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.2)
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Key insight
    st.info("""
    **Key insight for Enercast:**
    During the sustained crisis phase, the value of a 1% improvement in forecast accuracy
    across their 240 GW portfolio exceeds **€8M/day** — versus ~€5.7M/day pre-conflict.
    This creates a strong case for customers to upgrade to premium forecasting tiers.
    Even under Scenario A (rapid resolution), elevated electricity prices keep the value
    well above pre-conflict levels through Q3 2026.
    """)


# ═══════════════════════════════════════════════════════════════════════
# PAGE 5 — METHODOLOGY
# ═══════════════════════════════════════════════════════════════════════
elif page == "📖 Methodology":

    st.title("📖 Methodology & Data Sources")

    st.subheader("Data Sources")
    sources = pd.DataFrame([
        ["German electricity prices", "SMARD (Bundesnetzagentur)", "Day-ahead €/MWh, DE-LU zone", "Free, no auth"],
        ["TTF natural gas", "ICE / Bruegel / Wikipedia benchmarks", "Dutch TTF €/MWh daily", "Published reports"],
        ["Brent crude oil", "EIA API", "Daily $/bbl", "Free, API key"],
        ["Wind generation", "Open-Meteo historical API", "Hourly wind speed, 3 German regions", "Free, no auth"],
        ["Conflict timeline", "Wikipedia, Bruegel, CSIS, Atlantic Council", "Key event dates", "Public"],
        ["Energy benchmarks", "Fraunhofer ISE, IRENA, Goldman Sachs", "Cost/price validation", "Published"],
    ], columns=["Dataset", "Source", "Content", "Access"])
    st.dataframe(sources, use_container_width=True, hide_index=True)

    st.subheader("Methods Used")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Event Study**
- Estimation window: 90 days pre-conflict
- Linear trend model as "normal" price baseline
- Abnormal Return = Actual − Expected
- CAR = Cumulative sum of AR
- Significance: independent t-test (pre vs post)
- Effect size: Cohen's d
        """)
    with col2:
        st.markdown("""
**Scenario Forecasting**
- Model: Gradient Boosting Regressor
- Features: 26 (lags, rolling stats, seasonality, conflict dummies)
- Validation: last 15% holdout → MAPE ~11.8%
- Horizon: 90 days (May–Jul 2026)
- 3 scenarios: geopolitical assumption variations
- Autoregressive iteration (day-by-day)
        """)

    st.subheader("Enercast Impact Model")
    st.markdown("""
**Value of forecast accuracy:**
```
Daily value = electricity_price × imbalance_multiplier (1.25)
            × MAPE_improvement (1%)
            × portfolio_capacity (240 GW × 1000 MW/GW × 24 hrs)
```
Higher electricity prices directly amplify the value of accurate renewable forecasting,
because imbalance penalties and trading margins scale with the spot price.
    """)

    st.subheader("Project Structure")
    st.code("""
iran-energy-impact/
├── data/
│   ├── raw/           # master_dataset, conflict_timeline, prices
│   └── processed/     # event_study, scenarios, enercast_impact
├── src/
│   ├── 01_data_collection.py    # SMARD, EIA, Open-Meteo
│   ├── 02_event_study.py        # Statistical event analysis
│   └── 03_scenario_forecasting.py  # GradientBoosting + scenarios
├── app/
│   └── dashboard.py             # This Streamlit app
└── requirements.txt
    """, language="text")

    st.subheader("How to run this project locally")
    st.code("""
git clone https://github.com/gauravbhatia-bit/iran-energy-impact
cd iran-energy-impact
pip install -r requirements.txt

# Step 1: Collect data (SMARD + EIA + Open-Meteo)
python src/01_data_collection.py

# Step 2: Run event study
python src/02_event_study.py

# Step 3: Scenario forecasting
python src/03_scenario_forecasting.py

# Step 4: Launch dashboard
streamlit run app/dashboard.py
    """, language="bash")
