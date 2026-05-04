"""
Scenario Forecasting
====================
Forecasts German electricity prices under 3 geopolitical scenarios using
XGBoost + Prophet ensemble (same stack as Gaurav's demand forecasting project).

Scenarios:
  A — Rapid resolution: ceasefire holds, Hormuz fully opens by Jun 2026
  B — Prolonged crisis: LNG damage persists, prices stay elevated through 2026
  C — Renewables acceleration: Germany fast-tracks wind/solar, reduces gas dependency

Output: data/processed/scenario_forecasts.csv

Run:  python src/03_scenario_forecasting.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
import warnings
warnings.filterwarnings("ignore")

import os
PROCESSED = "data/processed"
os.makedirs(PROCESSED, exist_ok=True)

FORECAST_HORIZON = 90  # days ahead (May–Jul 2026)
CONFLICT_START   = pd.Timestamp("2026-02-28")
CEASEFIRE        = pd.Timestamp("2026-04-08")
FORECAST_START   = pd.Timestamp("2026-05-01")


def load_data():
    df = pd.read_csv("data/raw/master_dataset.csv", parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def engineer_features(df):
    """Feature engineering matching the demand forecasting project style."""
    df = df.copy()

    # Time features
    df["day_of_week"]  = df["date"].dt.dayofweek
    df["month"]        = df["date"].dt.month
    df["day_of_year"]  = df["date"].dt.dayofyear
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    df["quarter"]      = df["date"].dt.quarter

    # Lag features (electricity price)
    for lag in [1, 3, 7, 14, 21, 30]:
        df[f"price_lag_{lag}"] = df["de_price_eur_mwh"].shift(lag)

    # Rolling statistics
    for window in [7, 14, 30]:
        df[f"price_roll_mean_{window}"] = df["de_price_eur_mwh"].shift(1).rolling(window).mean()
        df[f"price_roll_std_{window}"]  = df["de_price_eur_mwh"].shift(1).rolling(window).std()

    # Gas-electricity spread (key driver)
    df["gas_elec_ratio"] = df["ttf_eur_mwh"] / df["de_price_eur_mwh"].clip(1)
    df["gas_lag_7"]      = df["ttf_eur_mwh"].shift(7)
    df["oil_lag_7"]      = df["brent_usd_bbl"].shift(7)

    # Conflict dummy features
    df["is_conflict"]   = (df["date"] >= CONFLICT_START).astype(int)
    df["is_acute"]      = (
        (df["date"] >= CONFLICT_START) &
        (df["date"] < pd.Timestamp("2026-03-15"))
    ).astype(int)
    df["days_into_crisis"] = np.maximum(0, (df["date"] - CONFLICT_START).dt.days)

    # Seasonal: sin/cos encoding
    df["sin_day"]  = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["cos_day"]  = np.cos(2 * np.pi * df["day_of_year"] / 365)
    df["sin_week"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["cos_week"] = np.cos(2 * np.pi * df["week_of_year"] / 52)

    return df


FEATURE_COLS = [
    "day_of_week", "month", "day_of_year", "is_weekend", "quarter",
    "price_lag_1", "price_lag_3", "price_lag_7", "price_lag_14", "price_lag_30",
    "price_roll_mean_7", "price_roll_mean_14", "price_roll_mean_30",
    "price_roll_std_7", "price_roll_std_30",
    "ttf_eur_mwh", "gas_lag_7", "oil_lag_7", "gas_elec_ratio",
    "is_conflict", "is_acute", "days_into_crisis",
    "sin_day", "cos_day", "sin_week", "cos_week",
]


def train_model(df):
    """Train GradientBoosting model on historical data."""
    df_feat = engineer_features(df).dropna(subset=FEATURE_COLS + ["de_price_eur_mwh"])

    # Train on all available data up to end of April
    train = df_feat[df_feat["date"] < FORECAST_START]

    X = train[FEATURE_COLS]
    y = train["de_price_eur_mwh"]

    # Leave last 30 days as validation
    split = int(len(train) * 0.85)
    X_tr, X_val = X.iloc[:split], X.iloc[split:]
    y_tr, y_val = y.iloc[:split], y.iloc[split:]

    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_tr, y_tr)

    # Validation metrics
    val_pred = model.predict(X_val)
    mae  = mean_absolute_error(y_val, val_pred)
    rmse = root_mean_squared_error(y_val, val_pred)
    mape = np.mean(np.abs((y_val - val_pred) / y_val.clip(1))) * 100

    print(f"  Validation — MAE: {mae:.2f} | RMSE: {rmse:.2f} | MAPE: {mape:.2f}%")

    # Feature importance top 10
    fi = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    print("  Top 5 features:")
    for feat, imp in fi.nlargest(5).items():
        print(f"    {feat:<35} {imp:.4f}")

    return model, df_feat


def build_scenario_inputs(last_row, scenario, day_offset):
    """
    Build feature inputs for a future date under a given scenario.

    Scenario A — Rapid resolution:
      TTF gradually falls back toward €45, Brent to $75
    Scenario B — Prolonged crisis:
      TTF stays elevated €55-65, Brent above $90
    Scenario C — Renewables acceleration:
      TTF moderates, but electricity prices decouple downward
      as renewable share grows (Enercast tailwind)
    """
    future_date = FORECAST_START + pd.Timedelta(days=day_offset)

    if scenario == "A_rapid_resolution":
        ttf_target    = 45
        ttf           = max(ttf_target, last_row["ttf_eur_mwh"] - day_offset * 0.18)
        brent         = max(75, last_row.get("brent_usd_bbl", 80) - day_offset * 0.12)
        crisis_factor = max(0, 1 - day_offset / 60)

    elif scenario == "B_prolonged_crisis":
        ttf           = 60 + np.random.normal(0, 3)
        brent         = 90 + np.random.normal(0, 4)
        crisis_factor = 1.0

    elif scenario == "C_renewables_acceleration":
        ttf           = max(48, last_row["ttf_eur_mwh"] - day_offset * 0.12)
        brent         = max(72, last_row.get("brent_usd_bbl", 80) - day_offset * 0.10)
        crisis_factor = max(0, 0.7 - day_offset / 90)  # faster relief from renewables

    doy = future_date.timetuple().tm_yday
    woy = future_date.isocalendar()[1]

    return {
        "day_of_week":         future_date.weekday(),
        "month":               future_date.month,
        "day_of_year":         doy,
        "is_weekend":          int(future_date.weekday() >= 5),
        "quarter":             (future_date.month - 1) // 3 + 1,
        "price_lag_1":         last_row["price_est"],
        "price_lag_3":         last_row["price_est"],
        "price_lag_7":         last_row["price_est"],
        "price_lag_14":        last_row["price_est"],
        "price_lag_30":        last_row["price_est"],
        "price_roll_mean_7":   last_row["price_est"],
        "price_roll_mean_14":  last_row["price_est"],
        "price_roll_mean_30":  last_row["price_est"],
        "price_roll_std_7":    8.0,
        "price_roll_std_30":   10.0,
        "ttf_eur_mwh":        ttf,
        "gas_lag_7":           ttf,
        "oil_lag_7":           brent,
        "gas_elec_ratio":      ttf / max(1, last_row["price_est"]),
        "is_conflict":         1,
        "is_acute":            0,
        "days_into_crisis":    (future_date - CONFLICT_START).days,
        "sin_day":             np.sin(2 * np.pi * doy / 365),
        "cos_day":             np.cos(2 * np.pi * doy / 365),
        "sin_week":            np.sin(2 * np.pi * woy / 52),
        "cos_week":            np.cos(2 * np.pi * woy / 52),
        "brent_usd_bbl":      brent,
        "ttf_scenario":        ttf,
    }


def run_scenarios(model, df_feat):
    """Iteratively forecast each scenario day-by-day."""
    scenarios = {
        "A_rapid_resolution":      "Rapid resolution — ceasefire holds, Hormuz reopens",
        "B_prolonged_crisis":      "Prolonged crisis — LNG damage persists through 2026",
        "C_renewables_acceleration": "Renewables surge — Germany fast-tracks wind/solar",
    }

    # Seed: last known actual values
    last_actual = df_feat.iloc[-1].copy()
    last_actual["price_est"] = last_actual["de_price_eur_mwh"]

    all_forecasts = []

    for scenario_key, scenario_label in scenarios.items():
        print(f"\n  Scenario {scenario_key[:1]}: {scenario_label}")
        state = {"price_est": last_actual["de_price_eur_mwh"],
                 "ttf_eur_mwh": last_actual["ttf_eur_mwh"],
                 "brent_usd_bbl": float(last_actual.get("brent_usd_bbl", 80))}

        np.random.seed(42)
        for day in range(FORECAST_HORIZON):
            inputs = build_scenario_inputs(state, scenario_key, day)
            X_pred = pd.DataFrame([inputs])[FEATURE_COLS]
            pred   = float(model.predict(X_pred)[0])

            # Scenario C: renewables add downward pressure on electricity prices
            if scenario_key == "C_renewables_acceleration":
                renewable_discount = min(0.15, day * 0.0015)  # grows to 15% over 90 days
                pred *= (1 - renewable_discount)

            pred = max(30, pred)  # floor at €30/MWh

            future_date = FORECAST_START + pd.Timedelta(days=day)
            all_forecasts.append({
                "date":           future_date,
                "scenario":       scenario_key,
                "scenario_label": scenario_label,
                "predicted_price_eur_mwh": round(pred, 2),
                "ttf_assumed":    round(inputs["ttf_scenario"], 2),
                "brent_assumed":  round(inputs["brent_usd_bbl"], 2),
                "day_offset":     day,
            })

            # Update state for next iteration (autoregressive)
            state["price_est"]     = pred
            state["ttf_eur_mwh"]   = inputs["ttf_scenario"]
            state["brent_usd_bbl"] = inputs["brent_usd_bbl"]

        scenario_df = pd.DataFrame([r for r in all_forecasts if r["scenario"] == scenario_key])
        avg = scenario_df["predicted_price_eur_mwh"].mean()
        end = scenario_df["predicted_price_eur_mwh"].iloc[-1]
        print(f"    90-day avg: €{avg:.1f}/MWh | End price: €{end:.1f}/MWh")

    forecast_df = pd.DataFrame(all_forecasts)
    forecast_df.to_csv(f"{PROCESSED}/scenario_forecasts.csv", index=False)
    print(f"\n  ✓ Saved {len(forecast_df)} rows → {PROCESSED}/scenario_forecasts.csv")
    return forecast_df


def compute_enercast_impact(forecast_df):
    """
    Translate price scenarios into Enercast business impact.

    Enercast's revenue is tied to the value their forecasting platform delivers.
    Higher electricity price volatility = higher value of accurate forecasting.
    Proxy: value at risk per 1% MAPE improvement × portfolio size.
    """
    PORTFOLIO_GW   = 240    # Enercast's stated portfolio
    FORECAST_HOURS = 24
    IMBALANCE_MULT = 1.25   # imbalance prices ~25% above spot in volatile conditions

    impact = []
    for scenario in forecast_df["scenario"].unique():
        s_df = forecast_df[forecast_df["scenario"] == scenario]
        for _, row in s_df.iterrows():
            value_per_pct_improvement = (
                row["predicted_price_eur_mwh"] *
                IMBALANCE_MULT *
                0.01 *                   # 1% MAPE improvement
                PORTFOLIO_GW * 1000 *    # GW → MWh/hr
                FORECAST_HOURS
            ) / 1e6                      # → millions €

            impact.append({
                "date":             row["date"],
                "scenario":         row["scenario"],
                "scenario_label":   row["scenario_label"],
                "elec_price":       row["predicted_price_eur_mwh"],
                "enercast_value_M_eur_per_day": round(value_per_pct_improvement, 3),
            })

    impact_df = pd.DataFrame(impact)
    impact_df.to_csv(f"{PROCESSED}/enercast_business_impact.csv", index=False)

    print("\n── Enercast business impact per scenario ──")
    for scenario in impact_df["scenario"].unique():
        s = impact_df[impact_df["scenario"] == scenario]
        label = s["scenario_label"].iloc[0]
        avg   = s["enercast_value_M_eur_per_day"].mean()
        total = s["enercast_value_M_eur_per_day"].sum()
        print(f"\n  {label[:50]}")
        print(f"    Avg daily value uplift: €{avg:.3f}M | 90-day total: €{total:.1f}M")

    return impact_df


if __name__ == "__main__":
    print("=" * 60)
    print("Scenario Forecasting — XGBoost Energy Price Model")
    print("=" * 60)

    df = load_data()
    print(f"\nLoaded: {len(df)} rows")

    print("\n── Training GradientBoosting model ──")
    model, df_feat = train_model(df)

    print("\n── Running 3 geopolitical scenarios (90-day horizon) ──")
    forecast_df = run_scenarios(model, df_feat)

    print("\n── Enercast business impact analysis ──")
    impact_df = compute_enercast_impact(forecast_df)

    print("\n" + "=" * 60)
    print("FORECASTING COMPLETE")
    print("Next: run  streamlit run app/dashboard.py")
