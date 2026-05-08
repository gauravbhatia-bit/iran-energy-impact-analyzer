"""
Event Study Analysis
====================
Measures the statistical impact of the Iran-USA conflict on German energy markets.

Methodology (standard finance/economics event study):
  1. Define estimation window (pre-event baseline)
  2. Define event window (around the shock)
  3. Calculate "normal" (expected) prices using the baseline trend
  4. Calculate Abnormal Returns (AR) = Actual - Expected
  5. Cumulative Abnormal Returns (CAR) = sum of AR over event window
  6. Statistical significance test (t-test)

Output:
  - data/processed/event_study_results.csv
  - data/processed/phase_comparison.csv

Run:  python src/02_event_study.py
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

PROCESSED = "data/processed"
import os; os.makedirs(PROCESSED, exist_ok=True)

CONFLICT_START = pd.Timestamp("2026-02-28")
CEASEFIRE      = pd.Timestamp("2026-04-08")
KEY_EVENTS = {
    "2026-02-28": "Conflict starts",
    "2026-03-02": "Oil/gas spike overnight",
    "2026-03-04": "Qatar LNG attack",
    "2026-03-15": "TTF doubles",
    "2026-04-08": "Ceasefire announced",
}


def load_master():
    df = pd.read_csv("data/raw/master_dataset.csv", parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def run_event_study(df, metric="de_price_eur_mwh", label="German electricity price"):
    """
    Run event study for a given price metric.
    Estimation window: 90 days before conflict (Oct–Jan baseline)
    Event window: -10 to +60 days around conflict start
    """
    estimation = df[df["date"] < CONFLICT_START].tail(90).copy()
    event_window = df[
        (df["date"] >= CONFLICT_START - pd.Timedelta(days=10)) &
        (df["date"] <= CONFLICT_START + pd.Timedelta(days=60))
    ].copy()

    # Fit linear trend on estimation window to model "normal" price
    X_est = np.arange(len(estimation))
    y_est = estimation[metric].values
    slope, intercept, r, p, se = stats.linregress(X_est, y_est)

    # Predict "normal" prices over event window
    normal_prices = []
    for i, row in event_window.iterrows():
        days_from_est_end = (row["date"] - estimation["date"].iloc[-1]).days
        predicted = intercept + slope * (len(estimation) + days_from_est_end)
        normal_prices.append(predicted)

    event_window = event_window.copy()
    event_window["normal_price"]    = normal_prices
    event_window["abnormal_return"] = event_window[metric] - event_window["normal_price"]
    event_window["days_from_event"] = (event_window["date"] - CONFLICT_START).dt.days
    event_window["CAR"]             = event_window["abnormal_return"].cumsum()
    event_window["metric"]          = metric
    event_window["label"]           = label

    # Statistical significance: t-test of AR in post-event vs pre-event window
    pre  = event_window[event_window["days_from_event"] < 0]["abnormal_return"]
    post = event_window[event_window["days_from_event"] >= 0]["abnormal_return"]
    t_stat, p_val = stats.ttest_ind(post, pre)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((pre.std()**2 + post.std()**2) / 2)
    cohens_d   = (post.mean() - pre.mean()) / pooled_std if pooled_std > 0 else 0

    stats_summary = {
        "metric":         metric,
        "label":          label,
        "pre_mean":       round(pre.mean(), 2),
        "post_mean":      round(post.mean(), 2),
        "mean_AR":        round(post.mean(), 2),
        "total_CAR":      round(event_window[event_window["days_from_event"] >= 0]["CAR"].iloc[-1], 2),
        "t_statistic":    round(t_stat, 3),
        "p_value":        round(p_val, 4),
        "significant":    p_val < 0.05,
        "cohens_d":       round(cohens_d, 3),
        "r_squared_baseline": round(r**2, 3),
        "n_estimation":   len(estimation),
        "n_event":        len(post),
    }

    return event_window, stats_summary


def phase_comparison(df):
    """Compare key metrics across the 4 conflict phases."""
    metrics = ["de_price_eur_mwh", "ttf_eur_mwh", "brent_usd_bbl"]
    phases  = ["pre_conflict", "acute_shock", "sustained_crisis", "post_ceasefire"]

    rows = []
    baseline = df[df["conflict_phase"] == "pre_conflict"]

    for phase in phases:
        subset = df[df["conflict_phase"] == phase]
        row = {"phase": phase, "n_days": len(subset)}
        for m in metrics:
            row[f"{m}_mean"]   = round(subset[m].mean(), 2)
            row[f"{m}_std"]    = round(subset[m].std(), 2)
            base_mean = baseline[m].mean()
            row[f"{m}_pct_change"] = round((subset[m].mean() - base_mean) / base_mean * 100, 1)
        rows.append(row)

    return pd.DataFrame(rows)


def renewable_premium_analysis(df):
    """
    Calculate the 'renewable premium' — the avoided cost per MWh of wind generation
    when electricity prices spike. This directly models Enercast customer value.

    When gas drives electricity prices up, accurate wind forecasting = more revenue
    for generators and more cost savings for grid operators.
    """
    df = df.copy()

    # Baseline electricity price (pre-conflict average)
    baseline_price = df[df["conflict_phase"] == "pre_conflict"]["de_price_eur_mwh"].mean()

    # Premium per MWh = (current price - baseline) × renewable generation
    df["price_premium_eur_mwh"] = df["de_price_eur_mwh"] - baseline_price
    df["avoided_cost_eur_day"]  = df["price_premium_eur_mwh"] * df["total_wind_gwh"] * 1000  # GWh → MWh

    # For Enercast specifically: a 1% improvement in forecast accuracy for their
    # 240 GW portfolio translates directly to avoided imbalance costs.
    # German imbalance prices ~ 1.2× spot price during high-volatility periods
    ENERCAST_PORTFOLIO_GW = 240
    FORECAST_IMPROVEMENT  = 0.01  # 1% better MAPE
    IMBALANCE_MULTIPLIER  = 1.2

    df["enercast_value_improvement_eur_day"] = (
        df["de_price_eur_mwh"] *
        IMBALANCE_MULTIPLIER *
        FORECAST_IMPROVEMENT *
        ENERCAST_PORTFOLIO_GW * 1000 *  # GW → MW → MWh per hour
        24  # hours per day
        / 1e6  # → millions
    )

    return df


if __name__ == "__main__":
    print("=" * 60)
    print("Event Study Analysis — Iran-USA Conflict Energy Impact")
    print("=" * 60)

    df = load_master()
    print(f"\nLoaded master dataset: {len(df)} rows")

    # ── Run event studies for all three price metrics
    print("\n── Running event studies ──")
    studies = [
        ("de_price_eur_mwh", "German electricity price (€/MWh)"),
        ("ttf_eur_mwh",      "Dutch TTF gas price (€/MWh)"),
        ("brent_usd_bbl",    "Brent crude oil ($/bbl)"),
    ]

    all_events = []
    all_stats  = []

    for metric, label in studies:
        ev_df, stats_row = run_event_study(df, metric, label)
        all_events.append(ev_df)
        all_stats.append(stats_row)

        sig = "✓ SIGNIFICANT" if stats_row["significant"] else "✗ not significant"
        print(f"\n  {label}")
        print(f"    Pre-event AR mean:  {stats_row['pre_mean']:+.2f}")
        print(f"    Post-event AR mean: {stats_row['post_mean']:+.2f}")
        print(f"    Total CAR:          {stats_row['total_CAR']:+.2f}")
        print(f"    t={stats_row['t_statistic']}, p={stats_row['p_value']} → {sig}")
        print(f"    Cohen's d:          {stats_row['cohens_d']:.3f} (effect size)")

    # Save event study results
    pd.concat(all_events).to_csv(f"{PROCESSED}/event_study_timeseries.csv", index=False)
    pd.DataFrame(all_stats).to_csv(f"{PROCESSED}/event_study_stats.csv", index=False)
    print(f"\n  ✓ Saved to {PROCESSED}/event_study_*.csv")

    # ── Phase comparison
    print("\n── Phase comparison ──")
    phase_df = phase_comparison(df)
    phase_df.to_csv(f"{PROCESSED}/phase_comparison.csv", index=False)
    print(phase_df[["phase", "n_days", "de_price_eur_mwh_mean",
                    "de_price_eur_mwh_pct_change", "ttf_eur_mwh_mean",
                    "brent_usd_bbl_mean"]].to_string(index=False))

    # ── Renewable premium
    print("\n── Renewable premium analysis (Enercast business impact) ──")
    rp_df = renewable_premium_analysis(df)
    rp_df.to_csv(f"{PROCESSED}/renewable_premium.csv", index=False)

    crisis_days = rp_df[rp_df["conflict_phase"].isin(["acute_shock", "sustained_crisis"])]
    print(f"\n  Avg avoided cost during crisis: €{crisis_days['avoided_cost_eur_day'].mean():,.0f}/day")
    print(f"  Enercast portfolio value uplift: €{crisis_days['enercast_value_improvement_eur_day'].mean():.2f}M/day")
    print(f"  Over crisis period ({len(crisis_days)} days): €{crisis_days['enercast_value_improvement_eur_day'].sum():.1f}M additional value")

    print("\n  ✓ Saved to", f"{PROCESSED}/renewable_premium.csv")
    print("\nNext: run  python src/03_scenario_forecasting.py")
