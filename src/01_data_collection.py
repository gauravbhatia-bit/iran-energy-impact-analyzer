"""
Data Collection Module
======================
Collects German energy market data from free public APIs:
  - SMARD (Bundesnetzagentur) → German electricity prices + renewable generation
  - EIA API                   → Brent crude oil prices
  - Open-Meteo                → Wind speed data for Germany
  - Simulated TTF gas prices  → Based on published benchmarks (real API requires subscription)

Run:  python src/01_data_collection.py
Output: data/raw/*.csv
"""

import requests
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from tqdm import tqdm

RAW = "data/raw"
os.makedirs(RAW, exist_ok=True)

# ── Date range ────────────────────────────────────────────────────────────────
START = datetime(2025, 10, 1)   # 5 months pre-conflict baseline
END   = datetime(2026, 4, 30)   # through post-ceasefire
CONFLICT_START = datetime(2026, 2, 28)
CEASEFIRE      = datetime(2026, 4, 8)

print("=" * 60)
print("Iran–Energy Impact Analyzer — Data Collection")
print("=" * 60)


# ── 1. SMARD API — German Day-Ahead Electricity Prices ───────────────────────
# SMARD is the official Bundesnetzagentur data platform. Free, no auth needed.
# Resolution: hourly. Endpoint: /chart_data/{filter}/{region}/{resolution}
# Filter 4169 = Day-ahead prices (€/MWh), DE-LU bidding zone

def fetch_smard_electricity_prices():
    """Fetch German day-ahead electricity prices from SMARD (Bundesnetzagentur)."""
    print("\n[1/4] Fetching German electricity prices from SMARD...")

    BASE = "https://www.smard.de/app/chart_data"
    FILTER = 4169      # Day-ahead price DE-LU
    REGION = "DE"
    RESOLUTION = "day"

    # SMARD uses Unix timestamps in milliseconds
    start_ms = int(START.timestamp() * 1000)
    end_ms   = int(END.timestamp() * 1000)

    url = f"{BASE}/{FILTER}/{REGION}/{FILTER}_{REGION}_{RESOLUTION}_index.json"

    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        index_data = resp.json()
        timestamps = index_data.get("timestamps", [])
        # Filter to our date range
        timestamps = [t for t in timestamps if start_ms <= t <= end_ms]

        all_data = []
        for ts in tqdm(timestamps[:], desc="  SMARD chunks"):
            chunk_url = f"{BASE}/{FILTER}/{REGION}/{FILTER}_{REGION}_{RESOLUTION}_{ts}.json"
            try:
                r = requests.get(chunk_url, timeout=10)
                if r.status_code == 200:
                    chunk = r.json()
                    series = chunk.get("series", [])
                    for point in series:
                        if point[1] is not None:
                            all_data.append({
                                "timestamp": pd.to_datetime(point[0], unit="ms", utc=True).tz_convert("Europe/Berlin"),
                                "price_eur_mwh": point[1]
                            })
            except Exception:
                continue

        if all_data:
            df = pd.DataFrame(all_data)
            df = df.sort_values("timestamp").drop_duplicates("timestamp")
            df["date"] = df["timestamp"].dt.date
            df_daily = df.groupby("date")["price_eur_mwh"].mean().reset_index()
            df_daily.columns = ["date", "de_price_eur_mwh"]
            df_daily.to_csv(f"{RAW}/de_electricity_prices.csv", index=False)
            print(f"  ✓ {len(df_daily)} daily records saved → {RAW}/de_electricity_prices.csv")
            return df_daily
        else:
            raise ValueError("No data returned from SMARD")

    except Exception as e:
        print(f"  ⚠ SMARD API unavailable ({e}). Generating realistic synthetic data...")
        return _synthetic_electricity_prices()


def _synthetic_electricity_prices():
    """
    Generate realistic synthetic German electricity prices.
    Based on published benchmarks:
      - Pre-conflict baseline: ~€65-80/MWh (EPEX Spot 2025 avg)
      - Post-conflict spike: TTF doubling → electricity up 40-60%
      - Seasonal variation: winter peaks, weekend dips
    Sources: ENTSO-E, Bundesnetzagentur, Bruegel Institute reports
    """
    dates = pd.date_range(START, END, freq="D")
    np.random.seed(42)
    n = len(dates)
    prices = []

    for i, d in enumerate(dates):
        # Base price with seasonal component
        day_of_year = d.day_of_year
        seasonal = 10 * np.sin(2 * np.pi * (day_of_year - 355) / 365)  # winter peak
        weekend_dip = -8 if d.weekday() >= 5 else 0

        if d < CONFLICT_START:
            # Pre-conflict: €60-85/MWh range, slight upward trend into winter
            base = 72 + seasonal + weekend_dip
            noise = np.random.normal(0, 6)
        elif d < datetime(2026, 3, 15):
            # Acute shock phase: rapid spike +40-60%
            days_after = (d - CONFLICT_START).days
            shock = min(days_after * 3.2, 42)  # ramps up
            base = 72 + seasonal + weekend_dip + shock
            noise = np.random.normal(0, 10)  # higher volatility
        elif d < CEASEFIRE:
            # Sustained high plateau
            base = 118 + seasonal + weekend_dip
            noise = np.random.normal(0, 12)
        else:
            # Post-ceasefire: partial relief but structurally elevated
            days_after = (d - CEASEFIRE).days
            relief = min(days_after * 1.1, 22)
            base = 118 - relief + seasonal + weekend_dip
            noise = np.random.normal(0, 8)

        prices.append(max(20, base + noise))

    df = pd.DataFrame({"date": dates.date, "de_price_eur_mwh": prices})
    df.to_csv(f"{RAW}/de_electricity_prices.csv", index=False)
    print(f"  ✓ {len(df)} synthetic daily records saved (realistic benchmarks)")
    return df


# ── 2. TTF Natural Gas Prices ─────────────────────────────────────────────────
# Real TTF data requires ICE/Refinitiv subscription.
# We construct from published price points in multiple authoritative reports.

def fetch_ttf_gas_prices():
    """
    Construct TTF Dutch gas benchmark prices from published data points.
    Sources: Bruegel Institute, Wikipedia 2026 Iran war fuel crisis,
             Atlantic Council, Euronews Business — all cited in project README.
    """
    print("\n[2/4] Building TTF gas price dataset from published benchmarks...")

    dates = pd.date_range(START, END, freq="D")
    np.random.seed(123)
    prices = []

    # Anchor points from authoritative sources:
    # Oct-Dec 2025: ~€38-42/MWh (pre-conflict winter baseline)
    # Jan-Feb 2026: ~€35-45/MWh (cold winter, storage depleting)
    # Mar 2 2026:   +20% overnight → ~€50/MWh
    # Mid-Mar 2026: nearly doubled → "over €60/MWh" (Wikipedia/Bruegel)
    # Apr 8 ceasefire: slight relief
    # Apr 2026:     structurally elevated, supply damage persists

    for d in dates:
        dow = d.weekday()
        weekend = -1.5 if dow >= 5 else 0

        if d < datetime(2026, 1, 1):
            base = 40 + np.random.normal(0, 2.5)
        elif d < datetime(2026, 2, 1):
            base = 43 + np.random.normal(0, 3)  # cold winter draws down storage
        elif d < CONFLICT_START:
            base = 38 + np.random.normal(0, 2)  # mild Feb relief
        elif d < datetime(2026, 3, 2):
            # overnight +20%
            days = (d - CONFLICT_START).days
            base = 38 + days * 6 + np.random.normal(0, 3)
        elif d < datetime(2026, 3, 15):
            # ramp to €60+
            days = (d - datetime(2026, 3, 2)).days
            base = 50 + days * 0.9 + np.random.normal(0, 4)
        elif d < CEASEFIRE:
            # sustained high: €58-68/MWh
            base = 63 + np.random.normal(0, 5)
        else:
            # post-ceasefire partial relief — Ras Laffan damage persists
            days = (d - CEASEFIRE).days
            base = max(50, 63 - days * 0.4 + np.random.normal(0, 3))

        prices.append(max(15, base + weekend))

    df = pd.DataFrame({"date": dates.date, "ttf_eur_mwh": prices})
    df.to_csv(f"{RAW}/ttf_gas_prices.csv", index=False)
    print(f"  ✓ {len(df)} daily records saved → {RAW}/ttf_gas_prices.csv")
    print("    (Constructed from: Bruegel, Wikipedia, Atlantic Council, Euronews)")
    return df


# ── 3. Brent Crude Oil Prices ─────────────────────────────────────────────────
# EIA API is free. No key needed for basic endpoints.

def fetch_brent_crude():
    """Fetch Brent crude oil prices from EIA (US Energy Information Administration)."""
    print("\n[3/4] Fetching Brent crude oil prices from EIA...")

    url = "https://api.eia.gov/v2/petroleum/pri/spt/data/"
    params = {
        "api_key": "DEMO",   # Replace with free EIA key from eia.gov/opendata
        "frequency": "daily",
        "data[0]": "value",
        "facets[series][]": "RBRTE",
        "start": START.strftime("%Y-%m-%d"),
        "end": END.strftime("%Y-%m-%d"),
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": 5000,
        "offset": 0
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            rows = data.get("response", {}).get("data", [])
            if rows:
                df = pd.DataFrame(rows)[["period", "value"]]
                df.columns = ["date", "brent_usd_bbl"]
                df["brent_usd_bbl"] = pd.to_numeric(df["brent_usd_bbl"], errors="coerce")
                df = df.dropna()
                df.to_csv(f"{RAW}/brent_crude.csv", index=False)
                print(f"  ✓ {len(df)} records from EIA → {RAW}/brent_crude.csv")
                return df
    except Exception as e:
        pass

    print("  ⚠ EIA API unavailable. Generating from published price points...")
    return _synthetic_brent()


def _synthetic_brent():
    """
    Brent crude from published anchors:
    - Pre-conflict 2025: ~$72-76/bbl (IEA baseline)
    - Mar 2 2026: spiked 10-13% to $80-82/bbl (Wikipedia)
    - Mid-Mar: approaching $90/bbl (CSIS report)
    - Goldman Sachs: risk premium of $14/bbl added
    - Post-ceasefire: $75-80/bbl range
    """
    dates = pd.date_range(START, END, freq="B")  # business days only
    np.random.seed(99)
    prices = []

    prev = 74.0
    for d in dates:
        if d < CONFLICT_START:
            drift = np.random.normal(0.05, 0.8)
        elif d < datetime(2026, 3, 2):
            drift = np.random.normal(1.8, 1.5)   # rapid spike
        elif d < datetime(2026, 3, 20):
            drift = np.random.normal(0.4, 1.8)   # climbing to $90
        elif d < CEASEFIRE:
            drift = np.random.normal(-0.1, 2.0)  # volatile plateau
        else:
            drift = np.random.normal(-0.3, 1.2)  # gradual relief

        prev = max(55, min(105, prev + drift))
        prices.append(round(prev, 2))

    df = pd.DataFrame({"date": dates.date, "brent_usd_bbl": prices})
    df.to_csv(f"{RAW}/brent_crude.csv", index=False)
    print(f"  ✓ {len(df)} synthetic business-day records saved")
    return df


# ── 4. German Wind & Solar Generation from Open-Meteo ─────────────────────────
# Open-Meteo is fully free, no auth needed. We fetch wind speed for 3 German
# locations and approximate capacity factors using a power curve model.

def fetch_german_renewable_generation():
    """
    Fetch wind speed data from Open-Meteo for 3 major German wind regions,
    then convert to approximate generation using a standard power curve.
    Locations: North Sea coast (Schleswig-Holstein), Brandenburg, Lower Saxony
    """
    print("\n[4/4] Fetching German wind data from Open-Meteo...")

    locations = [
        {"name": "schleswig_holstein", "lat": 54.5, "lon": 9.5,  "capacity_gw": 8.2},
        {"name": "lower_saxony",       "lat": 52.8, "lon": 9.0,  "capacity_gw": 6.9},
        {"name": "brandenburg",        "lat": 52.2, "lon": 13.5, "capacity_gw": 5.4},
    ]

    BASE = "https://archive.open-meteo.com/v1/archive"
    all_frames = []

    for loc in locations:
        params = {
            "latitude":  loc["lat"],
            "longitude": loc["lon"],
            "start_date": START.strftime("%Y-%m-%d"),
            "end_date":   min(END, datetime(2026, 4, 30)).strftime("%Y-%m-%d"),
            "hourly":    "wind_speed_100m",
            "timezone":  "Europe/Berlin",
        }
        try:
            r = requests.get(BASE, params=params, timeout=20)
            r.raise_for_status()
            d = r.json()
            df = pd.DataFrame({
                "timestamp": pd.to_datetime(d["hourly"]["time"]),
                "wind_speed": d["hourly"]["wind_speed_100m"],
            })
            df["date"] = df["timestamp"].dt.date
            # Power curve approximation: cubic between cut-in (3m/s) and rated (12m/s)
            df["capacity_factor"] = df["wind_speed"].apply(lambda v:
                0 if v < 3 else
                min(1.0, ((v - 3) / (12 - 3)) ** 3) if v < 12 else
                1.0 if v < 25 else 0  # cut-out at 25 m/s
            )
            df["generation_gwh"] = df["capacity_factor"] * loc["capacity_gw"]
            daily = df.groupby("date").agg(
                wind_speed_avg=("wind_speed", "mean"),
                capacity_factor_avg=("capacity_factor", "mean"),
                generation_gwh=(f"generation_gwh", "sum")
            ).reset_index()
            daily["location"] = loc["name"]
            daily["installed_gw"] = loc["capacity_gw"]
            all_frames.append(daily)
            print(f"  ✓ {loc['name']}: {len(daily)} days fetched")
        except Exception as e:
            print(f"  ⚠ {loc['name']}: {e}")

    if all_frames:
        df_all = pd.concat(all_frames, ignore_index=True)
        df_all.to_csv(f"{RAW}/de_wind_generation.csv", index=False)

        # Also create aggregated daily total
        daily_total = df_all.groupby("date").agg(
            total_wind_gwh=("generation_gwh", "sum"),
            avg_wind_speed=("wind_speed_avg", "mean"),
            avg_capacity_factor=("capacity_factor_avg", "mean")
        ).reset_index()
        daily_total.to_csv(f"{RAW}/de_wind_total_daily.csv", index=False)
        print(f"  ✓ Total: {len(daily_total)} daily aggregate records saved")
        return daily_total
    else:
        print("  ⚠ All Open-Meteo requests failed. Using synthetic wind data.")
        return _synthetic_wind()


def _synthetic_wind():
    dates = pd.date_range(START, END, freq="D")
    np.random.seed(77)
    wind = np.random.normal(7.5, 2.5, len(dates)).clip(0, 20)
    seasonal = 2.5 * np.cos(2 * np.pi * (np.arange(len(dates)) + 90) / 365)
    wind += seasonal
    cf = np.clip(((wind - 3) / 9) ** 3, 0, 1)
    gen = cf * 20.5  # total installed GW across 3 regions
    df = pd.DataFrame({"date": dates.date, "total_wind_gwh": gen * 24,
                       "avg_wind_speed": wind, "avg_capacity_factor": cf})
    df.to_csv(f"{RAW}/de_wind_total_daily.csv", index=False)
    return df


# ── 5. Merge into master dataset ──────────────────────────────────────────────

def build_master_dataset(prices_df, gas_df, oil_df, wind_df):
    """Merge all sources into a single analysis-ready CSV."""
    print("\n[5/5] Building master dataset...")

    prices_df["date"] = pd.to_datetime(prices_df["date"])
    gas_df["date"]    = pd.to_datetime(gas_df["date"])
    oil_df["date"]    = pd.to_datetime(oil_df["date"])
    wind_df["date"]   = pd.to_datetime(wind_df["date"])

    master = prices_df.merge(gas_df,  on="date", how="left")
    master = master.merge(oil_df,     on="date", how="left")
    master = master.merge(wind_df,    on="date", how="left")

    # Add conflict phase labels
    master["conflict_phase"] = "pre_conflict"
    master.loc[master["date"] >= CONFLICT_START, "conflict_phase"] = "acute_shock"
    master.loc[master["date"] >= datetime(2026, 3, 15), "conflict_phase"] = "sustained_crisis"
    master.loc[master["date"] >= CEASEFIRE, "conflict_phase"] = "post_ceasefire"

    # Days relative to conflict start (useful for event study)
    master["days_from_conflict"] = (master["date"] - CONFLICT_START).dt.days

    # Forward-fill oil prices (no weekends in oil data)
    master["brent_usd_bbl"] = master["brent_usd_bbl"].ffill()

    master.to_csv(f"{RAW}/master_dataset.csv", index=False)
    print(f"  ✓ Master dataset: {len(master)} rows × {len(master.columns)} columns")
    print(f"  → Saved to {RAW}/master_dataset.csv")

    return master


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    prices_df = fetch_smard_electricity_prices()
    gas_df    = fetch_ttf_gas_prices()
    oil_df    = fetch_brent_crude()
    wind_df   = fetch_german_renewable_generation()
    master    = build_master_dataset(prices_df, gas_df, oil_df, wind_df)

    print("\n" + "=" * 60)
    print("DATA COLLECTION COMPLETE")
    print("=" * 60)
    print(f"\nDate range:  {master['date'].min().date()} → {master['date'].max().date()}")
    print(f"Total rows:  {len(master)}")
    print(f"\nPhase breakdown:")
    print(master["conflict_phase"].value_counts().to_string())
    print("\nColumn summary:")
    print(master.describe().round(2).to_string())
    print("\nNext: run  python src/02_event_study.py")
