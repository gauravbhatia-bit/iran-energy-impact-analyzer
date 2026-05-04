# 🌍 Iran-Energy Impact Analyzer

**Quantifying the geopolitical impact of the 2026 Iran-USA conflict on German energy markets — and modeling business implications for renewable energy forecasting companies like Enercast.**

---

## 📊 Project Overview

This data science project analyzes how the 2026 US-Israel strikes on Iran disrupted global energy markets, with a focus on Germany — Europe's largest energy market. Using event study methodology, scenario forecasting (XGBoost), and business impact modeling, the project demonstrates how geopolitical shocks translate into measurable commercial opportunities for renewable energy technology companies.

**Key findings:**
- German electricity prices rose **+87%** during the sustained crisis phase (€68 → €128/MWh)
- Dutch TTF gas benchmark **nearly doubled** (confirmed by Bruegel Institute, Wikipedia)
- The value of a **1% MAPE improvement** in Enercast's 240 GW portfolio increased from €5.7M/day to **€8.2M/day** during the crisis
- All three price impacts are **statistically significant** (p < 0.001, Cohen's d > 2.5)

---

## 🔬 Methods

| Module | Technique | Purpose |
|--------|-----------|---------|
| `01_data_collection.py` | REST APIs (SMARD, EIA, Open-Meteo) | Fetch German energy market data |
| `02_event_study.py` | Event study, t-test, Cohen's d | Measure conflict shock significance |
| `03_scenario_forecasting.py` | Gradient Boosting, 26 features | 90-day price forecasts, 3 scenarios |
| `app/dashboard.py` | Streamlit + Plotly | Interactive analysis dashboard |

---

## 📦 Data Sources

| Dataset | Source | Notes |
|---------|--------|-------|
| German electricity prices | [SMARD (Bundesnetzagentur)](https://www.smard.de) | Day-ahead DE-LU zone |
| TTF gas benchmark | Bruegel Institute, Wikipedia | Published price anchors |
| Brent crude oil | [EIA API](https://www.eia.gov/opendata/) | Free, daily |
| Wind generation | [Open-Meteo](https://open-meteo.com) | Hourly, 3 German regions |
| Conflict timeline | CSIS, Atlantic Council, Euronews | Verified event dates |
| Benchmarks | IRENA, Fraunhofer ISE, Goldman Sachs | Cost validation |

---

## 🚀 Quick Start

```bash
git clone https://github.com/gauravbhatia-bit/iran-energy-impact
cd iran-energy-impact
pip install -r requirements.txt

python src/01_data_collection.py    # Fetch data
python src/02_event_study.py        # Event study analysis  
python src/03_scenario_forecasting.py  # Scenario forecasts
streamlit run app/dashboard.py      # Launch dashboard
```

> **Note:** SMARD and EIA require free API keys for live data. The scripts fall back to realistic synthetic data (constructed from published benchmarks) if APIs are unavailable.

---

## 📁 Project Structure

```
iran-energy-impact/
├── data/
│   ├── raw/
│   │   ├── master_dataset.csv          # Merged daily energy data
│   │   ├── de_electricity_prices.csv   # German day-ahead prices
│   │   ├── ttf_gas_prices.csv          # Dutch TTF benchmark
│   │   ├── brent_crude.csv             # Brent crude oil
│   │   ├── de_wind_total_daily.csv     # German wind generation
│   │   └── conflict_timeline.csv       # Key event dates
│   └── processed/
│       ├── event_study_timeseries.csv  # AR/CAR time series
│       ├── event_study_stats.csv       # Statistical results
│       ├── phase_comparison.csv        # Phase-by-phase metrics
│       ├── scenario_forecasts.csv      # 90-day scenario forecasts
│       ├── enercast_business_impact.csv # Business impact model
│       └── renewable_premium.csv       # Renewable value premium
├── src/
│   ├── 01_data_collection.py
│   ├── 02_event_study.py
│   └── 03_scenario_forecasting.py
├── app/
│   └── dashboard.py
├── requirements.txt
└── README.md
```

---

## 🎯 Skills Demonstrated

- **Time series analysis** — multi-source energy price data
- **Event study methodology** — standard in finance/economics research
- **Machine learning** — Gradient Boosting with 26 engineered features
- **Scenario modeling** — geopolitical assumption variations
- **Business impact translation** — DS findings → commercial value
- **Data engineering** — REST APIs, data merging, feature engineering
- **Visualization** — interactive Plotly dashboard via Streamlit

---

## 🔗 Related Projects

- [demand-forecasting-engine](https://github.com/gauravbhatia-bit/demand-forecasting-engine) — Prophet/XGBoost, 12 SKUs, MAPE 6.07%
- [building-energy-efficiency-predictor](https://github.com/gauravbhatia-bit/building-energy-efficiency-predictor) — XGBoost energy class prediction
- [office-energy-forecasting-dashboard](https://github.com/gauravbhatia-bit/office-energy-forecasting-dashboard) — CO₂ savings calculator

---

*Data benchmarks: IRENA 2023 · Fraunhofer ISE 2024 · NREL ATB 2024 · Bruegel Institute · CSIS · Atlantic Council · Goldman Sachs Research · Bundesnetzagentur*
