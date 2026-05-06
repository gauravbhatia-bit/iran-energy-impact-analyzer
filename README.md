# 🌍 Iran–Energy Impact Analyzer

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.22-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-27AE60?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)

**Quantifying the geopolitical impact of the 2026 Iran-USA conflict on German energy markets — and modeling business implications for renewable energy forecasting companies like Enercast.**

[🔬 Methodology](#-methodology) · [📦 Data Sources](#-data-sources) · [🚀 Quick Start](#-quick-start) · [👤 Author](#-author)

</div>

---

## 📌 Project Overview

The 2026 US-Israel strikes on Iran triggered one of the most significant energy supply shocks in recent European history. With the Strait of Hormuz disrupted and Qatar's Ras Laffan LNG facility damaged, German electricity and gas prices surged dramatically — reaching levels not seen since the 2022 energy crisis.

This project uses **event study methodology**, **Gradient Boosting scenario forecasting**, and **business impact modeling** to quantify the shock across three markets (electricity, gas, oil) and translate findings into commercial insights for **Enercast** — a leading AI-powered renewable energy forecasting platform managing 240 GW across 30 countries.

**Dataset:** 212 daily rows · Oct 2025 → Apr 2026 · 4 conflict phases · 9 features

---

## 🔑 Key Findings

### Price Impact by Conflict Phase

| Phase | Days | Electricity (€/MWh) | vs Baseline | TTF Gas (€/MWh) | Brent ($/bbl) |
|-------|------|---------------------|-------------|-----------------|----------------|
| Pre-conflict (baseline) | 150 | €68.4 | — | €40.0 | $78.7 |
| Acute shock | 15 | €102.9 | **+50.4%** | €52.7 | $91.9 |
| Sustained crisis | 24 | €128.0 | **+87.1%** | €61.8 | $102.6 |
| Post-ceasefire | 23 | €114.3 | **+67.1%** | €58.2 | $97.0 |

> **Peak values recorded:** Electricity **€152.6/MWh** · Brent crude **$105.0/bbl**

### Event Study — Statistical Significance

| Market | Mean Abnormal Return | Total CAR | t-statistic | p-value | Cohen's d |
|--------|---------------------|-----------|-------------|---------|-----------|
| German electricity | +€32.6/MWh | +1,983 | 6.38 | **< 0.001** | 2.70 ✅ |
| Dutch TTF gas | +€19.3/MWh | +1,167 | 9.83 | **< 0.001** | 4.33 ✅ |
| Brent crude oil | +$12.5/bbl | +785 | 6.64 | **< 0.001** | 2.78 ✅ |

> All three markets show statistically significant conflict-driven shocks with **large effect sizes** (Cohen's d > 2.5)

### Enercast Business Impact

| Phase | Daily Value of 1% MAPE Improvement | vs Baseline |
|-------|-------------------------------------|-------------|
| Pre-conflict baseline | €4.73M / day | — |
| Acute shock | €7.12M / day | +50.5% |
| Sustained crisis | **€8.85M / day** | **+87.1%** |
| Post-ceasefire | €7.90M / day | +67.0% |

> The crisis nearly **doubled the commercial value** of accurate renewable energy forecasting — directly linked to rising electricity prices amplifying grid imbalance penalty costs across Enercast's 240 GW portfolio.

---

## 🔮 Scenario Forecasts (May–Jul 2026, 90-day horizon)

| Scenario | Description | 90-day Avg | End Price | Enercast 90-day Value |
|----------|-------------|------------|-----------|----------------------|
| 🟢 A — Rapid resolution | Ceasefire holds, Hormuz reopens Jun 2026 | €130.3/MWh | €129.9/MWh | €844M |
| 🔴 B — Prolonged crisis | LNG damage persists, supply stays tight | €140.3/MWh | €143.1/MWh | €909M |
| 🔵 C — Renewables surge | Germany fast-tracks wind/solar buildout | €124.5/MWh | €114.5/MWh | €807M |

> Even the most optimistic scenario (C) keeps electricity **82% above pre-conflict levels** through July 2026 — structural LNG damage takes years to repair.

---

## 🗂️ Project Structure

```
iran-energy-impact-analyzer/
│
├── data/
│   ├── raw/
│   │   ├── master_dataset.csv             # 212 rows × 9 cols, Oct 2025–Apr 2026
│   │   ├── de_electricity_prices.csv      # German day-ahead prices (SMARD)
│   │   ├── ttf_gas_prices.csv             # Dutch TTF gas benchmark
│   │   ├── brent_crude.csv                # Brent crude oil (EIA)
│   │   ├── de_wind_total_daily.csv        # German wind generation (Open-Meteo)
│   │   └── conflict_timeline.csv          # 16 key geopolitical event dates
│   │
│   └── processed/
│       ├── event_study_timeseries.csv     # Abnormal Returns + CAR time series
│       ├── event_study_stats.csv          # t-test, Cohen's d, significance results
│       ├── phase_comparison.csv           # Phase-by-phase price comparison table
│       ├── scenario_forecasts.csv         # 90-day price forecasts × 3 scenarios
│       ├── enercast_business_impact.csv   # Daily value uplift model output
│       └── renewable_premium.csv          # Renewable avoided cost per day
│
├── src/
│   ├── 01_data_collection.py              # SMARD + EIA + Open-Meteo APIs
│   ├── 02_event_study.py                  # Statistical event analysis
│   └── 03_scenario_forecasting.py         # Gradient Boosting + 3 scenarios
│
├── app/
│   └── dashboard.py                       # 5-page Streamlit + Plotly dashboard
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🔬 Methodology

### Module 1 — Data Collection (`01_data_collection.py`)

Fetches and merges German energy market data from 4 public sources into a single master dataset. All 212 rows are labelled by conflict phase for downstream analysis.

| Source | Dataset | Access |
|--------|---------|--------|
| [SMARD — Bundesnetzagentur](https://www.smard.de) | German day-ahead electricity, DE-LU zone | Free, no key |
| [EIA Open Data](https://www.eia.gov/opendata/) | Brent crude oil, daily $/bbl | Free, API key |
| [Open-Meteo Archive](https://open-meteo.com) | Wind speed, 3 German regions → generation | Free, no key |
| Bruegel · Wikipedia · Atlantic Council | TTF gas benchmark (published anchors) | Public reports |

Scripts fall back to benchmark-accurate synthetic data if APIs are unavailable — the full pipeline runs either way.

---

### Module 2 — Event Study (`02_event_study.py`)

Standard financial event study — isolates the conflict-driven price shock from pre-existing trends.

```
Estimation window : 90 days pre-conflict (Oct 2025 – Jan 2026)
Event window      : −10 to +60 days around 28 Feb 2026 conflict start
Normal price      : Linear trend fitted on estimation window
Abnormal Return   : Actual price − Expected (normal) price
CAR               : Cumulative Abnormal Return (rolling sum of AR)
Significance test : Independent t-test (pre vs post event window)
Effect size       : Cohen's d
```

All three markets confirmed significant at p < 0.001 with large Cohen's d (2.70 – 4.33).

---

### Module 3 — Scenario Forecasting (`03_scenario_forecasting.py`)

**Model:** `GradientBoostingRegressor` (scikit-learn) · n_estimators=300 · learning_rate=0.05 · max_depth=4  
**Validation:** Last 15% holdout · **MAPE ~11.8%**  
**Forecasting method:** Autoregressive iteration — each day uses the previous day's predicted price as lag input

**26 engineered features:**

| Category | Features |
|----------|----------|
| Lag features | Price lags: 1, 3, 7, 14, 30 days |
| Rolling statistics | 7, 14, 30-day rolling mean + std |
| Commodity inputs | TTF gas, Brent crude + 7-day lags, gas/electricity ratio |
| Conflict dummies | `is_conflict`, `is_acute`, `days_into_crisis` |
| Seasonality | sin/cos encoding of day-of-year, week-of-year |
| Calendar | day-of-week, month, quarter, is_weekend |

---

### Module 4 — Enercast Business Impact Model

Translates electricity price levels into the daily commercial value of Enercast's forecasting accuracy.

```
Daily value (€M) = electricity_price (€/MWh)
                 × imbalance_multiplier (1.25)
                 × mape_improvement (1%)
                 × portfolio_capacity (240 GW × 1,000 × 24 hrs)
                 ÷ 1,000,000
```

Higher spot prices → higher grid imbalance penalties → higher revenue value of accurate wind/solar forecasting. Validated against Enercast's stated 240 GW portfolio across 30 countries.

---

## 📊 Dashboard — 5 Pages

| Page | Content |
|------|---------|
| 📊 Overview & Timeline | Phase-coloured price charts, 4 KPI cards, 16-event conflict timeline |
| 📉 Event Study | AR/CAR charts + t-stat/Cohen's d results, switchable across 3 markets |
| 🔮 Scenario Forecasts | 3-scenario price forecast with confidence bands, assumption table |
| 🏭 Enercast Business Impact | Daily value uplift over time, 90-day scenario totals |
| 📖 Methodology | Data sources, formulas, pipeline instructions |

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/gauravbhatia-bit/iran-energy-impact-analyzer.git
cd iran-energy-impact-analyzer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the data pipeline (in order)
python src/01_data_collection.py        # fetches all data sources
python src/02_event_study.py            # event study + statistics
python src/03_scenario_forecasting.py   # trains model + runs forecasts

# 4. Launch the dashboard
streamlit run app/dashboard.py
```

> **API note:** SMARD is fully free with no key. EIA requires a free key from [eia.gov/opendata](https://www.eia.gov/opendata/). Scripts auto-fallback to synthetic data if keys are missing.

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.11 |
| Machine learning | scikit-learn — GradientBoostingRegressor |
| Statistics | scipy (t-test, Cohen's d), statsmodels |
| Data wrangling | pandas, numpy |
| Visualization | plotly, streamlit |
| Data collection | requests, REST APIs (SMARD, EIA, Open-Meteo) |

---

## 🎯 Skills Demonstrated

- **Time series analysis** — 212-day multi-source energy dataset across 4 conflict phases
- **Event study methodology** — standard technique in quantitative finance and economics
- **Machine learning** — Gradient Boosting with 26 features, autoregressive forecasting, MAPE 11.8%
- **Statistical testing** — t-test + Cohen's d, validated at p < 0.001 across all markets
- **Scenario modeling** — 3 geopolitical paths, 90-day horizon, assumption sensitivity
- **Business impact modeling** — translating price forecasts into commercial revenue metrics
- **Data engineering** — multi-source API integration, feature engineering, phase labelling
- **Dashboard deployment** — 5-page Streamlit app with interactive Plotly charts

---

## 🔗 Related Projects

| Project | Description | Key result |
|---------|-------------|------------|
| [demand-forecasting-engine](https://github.com/gauravbhatia-bit/demand-forecasting-engine) | Prophet/XGBoost across 12 SKUs, 5 years of weekly data | MAPE **6.07%** |
| [building-energy-efficiency-predictor](https://github.com/gauravbhatia-bit/building-energy-efficiency-predictor) | XGBoost energy class (A–E) from building physics features | Civil domain + ML |
| [office-energy-forecasting-dashboard](https://github.com/gauravbhatia-bit/office-energy-forecasting-dashboard) | Office energy + live CO₂ savings estimator | MAE 22.98 Wh |

---

## 👤 Author

**Gaurav Bhatia**  
Data Scientist · MSc Data Science (GISMA University of Applied Sciences, Berlin)  
📍 Berlin, Germany  
📧 gauravbhatia.gb6@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/your-linkedin-profile) ← *update this*  
🐙 [GitHub](https://github.com/gauravbhatia-bit)

*Open to Werkstudent (20 hrs/week) and internship opportunities — available immediately in Berlin.*

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

*Sources: IRENA 2023 · Fraunhofer ISE 2024 · NREL ATB 2024 · Bruegel Institute · CSIS · Atlantic Council · Goldman Sachs Research · Bundesnetzagentur*

⭐ If this project helped you, consider giving it a star!

</div>
