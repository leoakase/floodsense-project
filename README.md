# 🌊 FloodSense — AI Flood Risk Intelligence System

> Probabilistic flood risk prediction, infrastructure policy simulation, and regional early warning — built for Nigeria.

[![3MTT NextGen Fellowship](https://img.shields.io/badge/3MTT-NextGen%20Fellowship-0080ff?style=flat-square)](https://3mtt.nitda.gov.ng)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Demo-ff4b4b?style=flat-square&logo=streamlit)](https://your-app-url.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## Overview

FloodSense is an AI-powered flood risk intelligence system that combines **Monte Carlo probabilistic simulation** with **machine learning** to predict flood risk, simulate the impact of infrastructure interventions, and monitor flood conditions across major Nigerian cities in real time.

Unlike deterministic flood models that produce a single fixed prediction, FloodSense accounts for **uncertainty in environmental conditions** — modelling rainfall, drainage, infiltration, runoff, and terrain as probability distributions, running 10,000 simulations per scenario to produce a statistically grounded risk estimate.

Built as an MVP for the **3MTT NextGen Fellowship Knowledge Showcase**, this system demonstrates how AI can directly support climate resilience, disaster preparedness, and infrastructure planning across Nigeria.

---

## Features

### 🔮 Module 1 — Flood Risk Prediction
- Input environmental parameters (rainfall, drainage, infiltration, runoff coefficient, slope)
- Outputs flood probability with an animated visual risk gauge
- Colour-coded risk levels: Low / Moderate / High
- Natural language insight explaining the dominant risk drivers
- Feature importance breakdown showing which variables drive the prediction most

### 🏗️ Module 2 — Policy & Infrastructure Simulator
- Set a baseline environmental scenario for any location
- Simulate the effect of three infrastructure interventions:
  - Drainage capacity upgrade
  - Surface runoff reduction (paving / green cover)
  - Soil infiltration improvement (afforestation / soil treatment)
  - Combined application of all three
- Visual before-vs-after horizontal bar chart
- Identifies the most effective single intervention and quantifies the combined impact

### 🚨 Module 3 — Regional Early Warning Monitor
- Monitors multiple Nigerian cities simultaneously
- Probabilistic rainfall forecast simulated per city (Gamma-distributed surge on climatological baseline)
- Geographic features (drainage, slope, infiltration, runoff coefficient) are fixed per city — reflecting real terrain and infrastructure conditions
- Add and remove monitored cities dynamically from a pre-loaded list of 10 cities
- Add completely custom locations with user-defined parameters
- Live alert banner triggered when any city crosses into high-risk territory
- Adjustable rainfall surge slider to simulate different storm forecast intensities

---

## Pre-loaded Nigerian Cities

FloodSense ships with estimated environmental parameters for 10 major Nigerian cities, reflecting their real flood risk profiles:

| City | Risk Profile |
|------|-------------|
| Lagos | Dense coastal urban — poor drainage, high runoff |
| Port Harcourt | Niger Delta — high rainfall, waterlogged soils |
| Kogi / Lokoja | River Niger confluence — historically flood-prone |
| Anambra / Onitsha | River Niger floodplain — high inundation risk |
| Ibadan | Hilly terrain — moderate urban density |
| Benin City | Forested region — moderate flood exposure |
| Abuja | Planned city — better drainage infrastructure |
| Kaduna | Savanna belt — moderate conditions |
| Kano | Arid north — low rainfall, good soil absorption |
| Maiduguri | Semi-arid — low flood risk baseline |

> **Note:** Geographic parameters (drainage, infiltration, runoff, slope) are estimated from general knowledge of each city's terrain, infrastructure, and hydrology. In a production system, these would be calibrated using NIHSA hydrological station data and GIS datasets. Rainfall is simulated probabilistically per monitoring cycle; in production it would be sourced from the NIMET or OpenWeatherMap API.

---

## How It Works

### Monte Carlo Simulation Engine (`floodnew.py`)

The core simulation models flood occurrence as a physical process under uncertainty. For each scenario:

1. **Base values** are defined for each environmental variable
2. **Uncertainty distributions** are applied per variable:
   - Rainfall → Gamma distribution
   - Infiltration → Log-normal distribution
   - Drainage capacity → Log-normal distribution
   - Runoff coefficient → Normal distribution
   - Slope → Normal distribution
3. **10,000 simulations** are run per scenario
4. A flood event is registered when:

```
effective_runoff = rainfall × runoff_coeff × (1 + 0.05 × slope)
flood occurs if: effective_runoff > (infiltration + drainage_capacity)
```

5. **Flood probability** = proportion of simulations where a flood occurs

This approach — borrowed from quantitative risk analysis — produces probabilistic estimates rather than binary yes/no predictions, enabling more honest and actionable decision-making under uncertainty.

### Surrogate ML Model (`train_model.py`)

Running 10,000 simulations per prediction is too slow for an interactive app. A **Gradient Boosting Regressor** (or Random Forest, whichever scores higher on the test set) is trained on 1,000 pre-simulated scenarios to learn the mapping:

```
[rainfall, infiltration, drainage, runoff, slope] → flood_probability
```

This surrogate model achieves **R² ≈ 0.97** on the held-out test set — making it a highly accurate and near-instant approximation of the Monte Carlo engine, suitable for real-time interactive use.

---

## Project Structure

```
floodsense/
├── floodnew.py          # Monte Carlo simulation engine (hand-written)
├── train_model.py       # Dataset generation, model training, model export
├── floodsense_app.py    # Streamlit web application
├── model.pkl            # Trained model (generated by train_model.py)
├── requirements.txt     # Python dependencies
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/yourusername/floodsense.git
cd floodsense
pip install -r requirements.txt
```

### Train the Model

Run this once to generate the dataset and save the trained model:

```bash
python train_model.py
```

This will:
- Generate 1,000 flood scenarios via Monte Carlo simulation (~60 seconds)
- Train and compare Random Forest vs Gradient Boosting models
- Print R² and MSE scores for both models
- Save the best-performing model to `model.pkl`
- Export a feature importance chart

### Run the App

```bash
streamlit run floodsense_app.py
```

App opens at `http://localhost:8501`

---

## Requirements

```
streamlit
scikit-learn
matplotlib
numpy
pandas
```

Install all at once:

```bash
pip install streamlit scikit-learn matplotlib numpy pandas
```

---

## Key Design Decisions

**Why Monte Carlo simulation?**
Most flood prediction tools use deterministic models — they give one answer assuming exact input values. Real environmental conditions are inherently uncertain. Monte Carlo simulation explicitly models this uncertainty, producing a probability estimate that is more honest and more useful for planning and early warning.

**Why a surrogate ML model?**
Running 10,000 simulations per prediction is too slow for an interactive app. The ML model is trained on Monte Carlo outputs, learning to approximate the simulation with sub-millisecond inference. This combines the statistical rigour of the simulation with the speed needed for a real-time dashboard.

**Why fix geographic features per city in the Early Warning module?**
Drainage infrastructure, terrain slope, and soil type change on timescales of years — they are fundamentally geographic. Only rainfall changes day-to-day. Fixing these features per city reflects real-world data architecture, where geographic attributes come from GIS/infrastructure databases and only weather inputs are updated from forecast APIs.

**Why simulate rainfall instead of using a live API?**
This is an MVP. The forecast rainfall simulation uses the same Gamma distribution as the Monte Carlo engine, producing realistic probabilistic surges per city. In production, this single function would be replaced with a NIMET or OpenWeatherMap API call — the rest of the system remains unchanged.

---

## Future Work

- Integration with NIMET / OpenWeatherMap API for live per-city rainfall forecasts
- Calibration of city parameters using NIHSA hydrological station data
- Interactive map visualization (Folium / Plotly) showing geographic risk distribution
- Historical flood event validation against NEMA records
- SMS / email alert system for high-risk cities
- Expansion to LGA-level granularity within each state

---

## Built With

- [Streamlit](https://streamlit.io) — web application framework
- [scikit-learn](https://scikit-learn.org) — machine learning (Gradient Boosting / Random Forest)
- [NumPy](https://numpy.org) — Monte Carlo simulation and numerical computing
- [Matplotlib](https://matplotlib.org) — data visualization
- [Pandas](https://pandas.pydata.org) — data handling and model training pipeline

---

## Acknowledgements

Built as part of the **3MTT NextGen Fellowship**, powered by the Airtel Africa Foundation and the Federal Ministry of Communications, Innovation & Digital Economy (FMCIDE).

The Monte Carlo simulation engine (`floodnew.py`) was written entirely from scratch, including a custom `lognormal_params()` function that correctly derives log-normal distribution parameters (μ, σ) from physical mean and standard deviation values — ensuring the simulation reflects realistic environmental behaviour rather than arbitrary distributions.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
