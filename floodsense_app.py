"""
FloodSense — AI Flood Risk Intelligence System
================================================
Streamlit app that imports the user's own Monte Carlo simulation
from floodnew.py and loads the trained model from model.pkl.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from floodnew import monte_carlo, lognormal_params

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FloodSense | AI Flood Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Sora:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] {
    background-color: #050d1a !important;
    color: #e0eaf8 !important;
    font-family: 'Sora', sans-serif !important;
}
.stApp {
    background: radial-gradient(ellipse at 20% 0%, #0a1f3d 0%, #050d1a 60%) !important;
}

/* HEADER */
.flood-header {
    background: linear-gradient(135deg, #051630 0%, #0a2545 50%, #051630 100%);
    border: 1px solid #1a3a5c;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.flood-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00b4ff, #0066ff, transparent);
}
.flood-header h1 {
    font-family: 'Sora', sans-serif !important;
    font-size: 2.6rem !important;
    font-weight: 800 !important;
    color: #ffffff !important;
    margin: 0 0 0.3rem 0 !important;
    letter-spacing: -0.5px;
}
.flood-header p { color: #6b9fc4 !important; font-size: 1.05rem !important; margin: 0 !important; font-weight: 300; }
.header-badge {
    display: inline-block;
    background: rgba(0, 180, 255, 0.1);
    border: 1px solid rgba(0, 180, 255, 0.3);
    color: #00b4ff;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 1rem;
    letter-spacing: 2px;
}

/* TABS */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    gap: 0.5rem !important;
    border-bottom: 1px solid #1a3a5c !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #4a7a9b !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 1.5rem !important;
    border-radius: 8px 8px 0 0 !important;
    border: 1px solid transparent !important;
    border-bottom: none !important;
}
.stTabs [aria-selected="true"] {
    background: #0a1f3d !important;
    color: #00b4ff !important;
    border-color: #1a3a5c !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: transparent !important;
    padding-top: 1.5rem !important;
}

/* CARDS */
.metric-card {
    background: linear-gradient(135deg, #0a1f3d 0%, #071628 100%);
    border: 1px solid #1a3a5c;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.metric-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,180,255,0.4), transparent);
}
.metric-card .value {
    font-family: 'Space Mono', monospace;
    font-size: 3rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.metric-card .label { font-size: 0.8rem; color: #4a7a9b; text-transform: uppercase; letter-spacing: 2px; }

.risk-low    { color: #00e676 !important; }
.risk-medium { color: #ffab00 !important; }
.risk-high   { color: #ff3d57 !important; }

.risk-banner {
    border-radius: 10px;
    padding: 1rem 1.5rem;
    font-weight: 700;
    font-size: 1.1rem;
    text-align: center;
    margin: 1rem 0;
    letter-spacing: 0.5px;
}
.risk-banner.low    { background: rgba(0,230,118,0.1); border: 1px solid #00e676; color: #00e676; }
.risk-banner.medium { background: rgba(255,171,0,0.1); border: 1px solid #ffab00; color: #ffab00; }
.risk-banner.high   { background: rgba(255,61,87,0.1);  border: 1px solid #ff3d57; color: #ff3d57; }

/* SLIDERS & INPUTS */
.stSlider > div > div > div > div { background: #00b4ff !important; }
.stSlider > div > div > div { background: #1a3a5c !important; }
.stSlider label, .stSelectbox label, .stTextInput label, .stNumberInput label {
    color: #6b9fc4 !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
}
.stSelectbox > div > div { background: #0a1f3d !important; border-color: #1a3a5c !important; color: #e0eaf8 !important; }
.stTextInput > div > div > input { background: #0a1f3d !important; border-color: #1a3a5c !important; color: #e0eaf8 !important; }
.stNumberInput > div > div > input { background: #0a1f3d !important; border-color: #1a3a5c !important; color: #e0eaf8 !important; }

/* BUTTONS */
.stButton > button {
    background: linear-gradient(135deg, #0055cc, #0080ff) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0066e0, #0099ff) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(0,128,255,0.3) !important;
}

/* MISC */
.section-label { font-family: 'Space Mono', monospace; font-size: 0.7rem; color: #00b4ff; text-transform: uppercase; letter-spacing: 3px; margin-bottom: 0.5rem; }
.section-title { font-size: 1.4rem; font-weight: 700; color: #ffffff; margin-bottom: 0.3rem; }
.section-desc  { color: #4a7a9b; font-size: 0.9rem; margin-bottom: 1.5rem; }
.insight-box {
    background: rgba(0,180,255,0.05);
    border-left: 3px solid #00b4ff;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
    font-size: 0.95rem;
    color: #a0c8e8;
    font-style: italic;
}
.styled-divider { border: none; height: 1px; background: linear-gradient(90deg, transparent, #1a3a5c, transparent); margin: 1.5rem 0; }
.fi-bar-wrap { margin-bottom: 0.6rem; }
.fi-label { font-size: 0.8rem; color: #6b9fc4; margin-bottom: 0.15rem; display: flex; justify-content: space-between; }
.fi-bar-bg { background: #0d2240; border-radius: 4px; height: 8px; overflow: hidden; }
.fi-bar-fill { height: 100%; border-radius: 4px; background: linear-gradient(90deg, #0055cc, #00b4ff); }
.location-card {
    background: linear-gradient(135deg, #0a1f3d 0%, #071628 100%);
    border: 1px solid #1a3a5c;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.alert-critical {
    background: rgba(255,61,87,0.08);
    border: 1px solid rgba(255,61,87,0.4);
    border-radius: 10px;
    padding: 1rem 1.5rem;
    color: #ff3d57;
    font-weight: 700;
    font-size: 1rem;
    text-align: center;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { border-color: rgba(255,61,87,0.4); }
    50%       { border-color: rgba(255,61,87,0.9); }
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem !important; max-width: 1200px !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists("model.pkl"):
        st.error("⚠️ model.pkl not found. Please run:  python train_model.py")
        st.stop()
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#050d1a', 'axes.facecolor': '#071628',
    'axes.edgecolor': '#1a3a5c',   'axes.labelcolor': '#6b9fc4',
    'xtick.color': '#4a7a9b',      'ytick.color': '#4a7a9b',
    'text.color': '#e0eaf8',       'grid.color': '#0d2240',
    'grid.linestyle': '--',        'grid.alpha': 0.7,
    'font.family': 'monospace',
})

def predict(rain, infi, drain, runoff, slope):
    val = model.predict([[rain, infi, drain, runoff, slope]])[0]
    return float(np.clip(val, 0, 1))

def risk_label(p):
    if p < 0.35: return "LOW",      "low",    "🟢"
    if p < 0.65: return "MODERATE", "medium", "🟡"
    return               "HIGH",    "high",   "🔴"

def risk_color(p):
    if p < 0.35: return '#00e676'
    if p < 0.65: return '#ffab00'
    return '#ff3d57'

def explain(rain, drain, runoff, prob):
    factors = []
    if rain > 110:    factors.append("extreme rainfall intensity")
    if drain < 25:    factors.append("critically low drainage capacity")
    if runoff > 0.75: factors.append("high surface runoff coefficient")
    if not factors:
        if prob < 0.35:
            return "Environmental conditions are within manageable limits. Drainage and soil absorption are keeping flood risk in check."
        return "Moderate interplay between rainfall and drainage — conditions are manageable but warrant monitoring."
    return f"Elevated risk is driven by {', '.join(factors)}. Targeted infrastructure improvements could significantly reduce exposure."

def simulate_forecast_rainfall(base_rain, surge_mean=15, surge_std=5):
    """
    Simulates forecast rainfall for a city.
    Adds a probabilistic surge (Gamma-distributed, matching floodnew.py's approach)
    on top of the city's climatological baseline rainfall.
    In production: replace with NIMET / OpenWeatherMap API call.
    """
    surge = np.random.gamma(shape=3, scale=max(surge_mean / 3, 0.1))
    noise = np.random.normal(0, surge_std)
    return float(np.clip(base_rain + surge + noise, 0, 300))


# ─────────────────────────────────────────────
# NIGERIAN CITY DATABASE
# Geo-features (drain, infi, runoff, slope) are fixed per city —
# they represent physical infrastructure and terrain.
# base_rain is the climatological average; actual forecast rain
# is simulated fresh each monitoring cycle.
# ─────────────────────────────────────────────
CITY_DB = {
    "Lagos": dict(
        base_rain=110, infi=14, drain=18, runoff=0.82, slope=1.2,
        note="Dense coastal urban — poor drainage, high runoff"
    ),
    "Port Harcourt": dict(
        base_rain=105, infi=16, drain=20, runoff=0.78, slope=1.5,
        note="Niger Delta — high rainfall, waterlogged soils"
    ),
    "Kogi / Lokoja": dict(
        base_rain=95, infi=18, drain=22, runoff=0.72, slope=3.0,
        note="River Niger confluence — historically flood-prone"
    ),
    "Anambra / Onitsha": dict(
        base_rain=98, infi=17, drain=21, runoff=0.74, slope=2.5,
        note="River Niger floodplain — high inundation risk"
    ),
    "Ibadan": dict(
        base_rain=85, infi=20, drain=26, runoff=0.65, slope=4.5,
        note="Hilly terrain — moderate urban density"
    ),
    "Benin City": dict(
        base_rain=88, infi=19, drain=24, runoff=0.68, slope=3.5,
        note="Forested region — moderate flood exposure"
    ),
    "Abuja": dict(
        base_rain=72, infi=25, drain=32, runoff=0.55, slope=5.5,
        note="Planned city — better drainage infrastructure"
    ),
    "Kaduna": dict(
        base_rain=60, infi=26, drain=30, runoff=0.50, slope=5.0,
        note="Savanna belt — moderate conditions"
    ),
    "Kano": dict(
        base_rain=48, infi=30, drain=35, runoff=0.42, slope=4.0,
        note="Arid north — low rainfall, good soil absorption"
    ),
    "Maiduguri": dict(
        base_rain=38, infi=32, drain=38, runoff=0.35, slope=3.5,
        note="Semi-arid — low flood risk baseline"
    ),
}


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="flood-header">
    <div class="header-badge">AI FLOOD INTELLIGENCE SYSTEM · 3MTT NEXTGEN FELLOWSHIP</div>
    <h1>🌊 FloodSense</h1>
    <p>Probabilistic flood risk prediction, infrastructure policy simulation, and regional early warning —
    powered by Monte Carlo simulation and machine learning.</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔮  Flood Prediction",
    "🏗️  Policy Simulator",
    "🚨  Early Warning Monitor",
])


# ══════════════════════════════════════════════
# TAB 1 — FLOOD PREDICTION
# ══════════════════════════════════════════════
with tab1:
    st.markdown("""
    <div class='section-label'>Module 01</div>
    <div class='section-title'>Flood Risk Prediction</div>
    <div class='section-desc'>Adjust environmental parameters to estimate flood probability under uncertain conditions using your Monte Carlo simulation engine.</div>
    """, unsafe_allow_html=True)

    col_in, _, col_out = st.columns([1.2, 0.1, 1])

    with col_in:
        st.markdown("<div style='color:#4a7a9b;font-size:0.8rem;text-transform:uppercase;letter-spacing:2px;margin-bottom:1rem'>Environmental Parameters</div>", unsafe_allow_html=True)
        rain1   = st.slider("Rainfall (mm)",      0,   200,  85,   key="r1")
        drain1  = st.slider("Drainage Capacity",  0,   60,   28,   key="d1")
        infi1   = st.slider("Soil Infiltration",  0,   50,   18,   key="i1")
        runoff1 = st.slider("Runoff Coefficient", 0.0, 1.0,  0.72, step=0.01, key="ro1")
        slope1  = st.slider("Terrain Slope",      0.0, 10.0, 3.0,  step=0.1,  key="s1")
        btn1    = st.button("⚡ Analyse Flood Risk", key="btn1")

    with col_out:
        if btn1:
            prob = predict(rain1, infi1, drain1, runoff1, slope1)
            label, cls, icon = risk_label(prob)
            color = risk_color(prob)

            # Gauge chart
            fig, ax = plt.subplots(figsize=(5, 3))
            fig.patch.set_facecolor('#050d1a')
            ax.set_facecolor('#050d1a')
            ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis('off')
            theta = np.linspace(np.pi, 0, 300)
            cx, cy, r = 5, 0.5, 3.8
            for i in range(len(theta) - 1):
                t = i / len(theta)
                c = plt.cm.RdYlGn_r(t)
                ax.plot([cx + r*np.cos(theta[i]),  cx + r*np.cos(theta[i+1])],
                        [cy + r*np.sin(theta[i]),  cy + r*np.sin(theta[i+1])],
                        color=c, linewidth=14, solid_capstyle='round')
            needle_angle = np.pi - prob * np.pi
            ax.annotate('', xy=(cx + 3.2*np.cos(needle_angle), cy + 3.2*np.sin(needle_angle)),
                        xytext=(cx, cy),
                        arrowprops=dict(arrowstyle='->', color='white', lw=2.5))
            ax.plot(cx, cy, 'o', color='white', markersize=10, zorder=5)
            ax.text(cx, cy+1.2, f"{prob:.0%}", ha='center', va='center',
                    fontsize=26, fontweight='bold', color=color, fontfamily='monospace')
            ax.text(cx, cy-0.3, 'FLOOD PROBABILITY', ha='center', va='center',
                    fontsize=7, color='#4a7a9b', fontfamily='monospace')
            ax.text(1.0, 0.2, 'LOW',  fontsize=7, color='#00e676', fontfamily='monospace')
            ax.text(8.2, 0.2, 'HIGH', fontsize=7, color='#ff3d57', fontfamily='monospace')
            st.pyplot(fig, use_container_width=True)
            plt.close()

            st.markdown(f"<div class='risk-banner {cls}'>{icon} {label} FLOOD RISK</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='insight-box'>💡 {explain(rain1, drain1, runoff1, prob)}</div>", unsafe_allow_html=True)

            st.markdown("<hr class='styled-divider'>", unsafe_allow_html=True)
            st.markdown("<div style='color:#4a7a9b;font-size:0.75rem;text-transform:uppercase;letter-spacing:2px;margin-bottom:0.8rem'>Key Risk Drivers</div>", unsafe_allow_html=True)
            features = ['Rainfall', 'Infiltration', 'Drainage', 'Runoff', 'Slope']
            for feat, imp in sorted(zip(features, model.feature_importances_), key=lambda x: -x[1]):
                pct = int(imp * 100)
                st.markdown(f"""
                <div class='fi-bar-wrap'>
                    <div class='fi-label'><span>{feat}</span><span style='color:#00b4ff'>{pct}%</span></div>
                    <div class='fi-bar-bg'><div class='fi-bar-fill' style='width:{pct}%'></div></div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='height:320px;display:flex;align-items:center;justify-content:center;
                        border:1px dashed #1a3a5c;border-radius:12px;flex-direction:column;gap:0.5rem'>
                <div style='font-size:2.5rem'>🌊</div>
                <div style='color:#4a7a9b;font-size:0.9rem'>Set parameters and click Analyse</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2 — POLICY SIMULATOR
# ══════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class='section-label'>Module 02</div>
    <div class='section-title'>Policy & Infrastructure Simulator</div>
    <div class='section-desc'>Simulate how specific infrastructure interventions reduce flood risk and quantify the impact of each policy decision.</div>
    """, unsafe_allow_html=True)

    col_p1, col_p2 = st.columns([1, 1.3])

    with col_p1:
        st.markdown("<div style='color:#4a7a9b;font-size:0.8rem;text-transform:uppercase;letter-spacing:2px;margin-bottom:1rem'>Baseline Conditions</div>", unsafe_allow_html=True)
        rain2   = st.slider("Rainfall (mm)",      0,   200,  120,  key="r2")
        drain2  = st.slider("Drainage Capacity",  0,   60,   22,   key="d2")
        infi2   = st.slider("Soil Infiltration",  0,   50,   15,   key="i2")
        runoff2 = st.slider("Runoff Coefficient", 0.0, 1.0,  0.80, step=0.01, key="ro2")
        slope2  = st.slider("Terrain Slope",      0.0, 10.0, 4.0,  step=0.1,  key="s2")

        st.markdown("<div style='color:#4a7a9b;font-size:0.8rem;text-transform:uppercase;letter-spacing:2px;margin:1.2rem 0 0.8rem'>Intervention Strengths</div>", unsafe_allow_html=True)
        drain_boost = st.slider("Drainage upgrade (capacity units)", 5,    30,   15,   key="db")
        runoff_cut  = st.slider("Runoff reduction (paving/green)",   0.05, 0.30, 0.12, step=0.01, key="rc")
        infi_boost  = st.slider("Soil improvement (infiltration+)",  2,    20,   8,    key="ib")
        btn2 = st.button("🏗️ Run Policy Simulation", key="btn2")

    with col_p2:
        if btn2:
            base = predict(rain2, infi2, drain2, runoff2, slope2)
            scenarios = {
                "Baseline":                   (rain2, infi2,              drain2,               runoff2,                        slope2),
                "Upgrade Drainage":           (rain2, infi2,              drain2 + drain_boost,  runoff2,                        slope2),
                "Reduce Surface Runoff":      (rain2, infi2,              drain2,               max(0.05, runoff2 - runoff_cut), slope2),
                "Improve Soil / Green Cover": (rain2, infi2 + infi_boost, drain2,               runoff2,                        slope2),
                "Combined Interventions":     (rain2, infi2+infi_boost//2, drain2+drain_boost//2, max(0.05, runoff2-runoff_cut/2), slope2),
            }
            results = {k: predict(*v) for k, v in scenarios.items()}

            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            labels = list(results.keys())
            values = list(results.values())
            bar_colors = [risk_color(v) for v in values]
            bars = ax.barh(labels[::-1], values[::-1], color=bar_colors[::-1], height=0.55, edgecolor='none')
            for bar, val in zip(bars, values[::-1]):
                ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{val:.0%}', va='center', fontsize=9, color='#e0eaf8', fontfamily='monospace')
            ax.axvline(base, color='#ffffff', linewidth=1, linestyle='--', alpha=0.4)
            ax.text(base + 0.01, len(labels) - 0.3, 'Baseline', fontsize=7, color='#ffffff', alpha=0.5)
            ax.set_xlim(0, 1.08)
            ax.set_xlabel("Flood Probability", fontsize=9)
            ax.set_title("Impact of Infrastructure Interventions", fontsize=11, color='#e0eaf8', pad=12)
            ax.grid(axis='x', alpha=0.3)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            ax.tick_params(axis='y', labelsize=8.5)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

            best_k = min(results, key=results.get)
            best_v = results[best_k]
            reduction = base - best_v
            lbl_b, cls_b, _ = risk_label(base)
            lbl_a, cls_a, _ = risk_label(best_v)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"<div class='metric-card'><div class='value risk-{cls_b}'>{base:.0%}</div><div class='label'>Current Risk</div></div>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"<div class='metric-card'><div class='value risk-{cls_a}'>{best_v:.0%}</div><div class='label'>Best Outcome</div></div>", unsafe_allow_html=True)
            with c3:
                st.markdown(f"<div class='metric-card'><div class='value' style='color:#00b4ff'>{reduction:.0%}</div><div class='label'>Risk Reduction</div></div>", unsafe_allow_html=True)

            st.markdown(f"""
            <div class='insight-box'>
                🏗️ Most effective intervention: <strong style='color:#00b4ff'>{best_k}</strong> —
                reduces flood probability from <strong>{base:.0%}</strong> to <strong>{best_v:.0%}</strong>.
                Combined measures achieve the greatest risk reduction.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='height:380px;display:flex;align-items:center;justify-content:center;
                        border:1px dashed #1a3a5c;border-radius:12px;flex-direction:column;gap:0.5rem'>
                <div style='font-size:2.5rem'>🏗️</div>
                <div style='color:#4a7a9b;font-size:0.9rem'>Set baseline conditions and run simulation</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 3 — EARLY WARNING MONITOR
# ══════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div class='section-label'>Module 03</div>
    <div class='section-title'>Regional Early Warning Monitor</div>
    <div class='section-desc'>
        Automatic flood risk monitoring across Nigerian cities using simulated forecast rainfall.
        Geographic features (drainage, slope, infiltration, runoff) are fixed per city —
        reflecting real infrastructure and terrain. Only rainfall varies with each forecast cycle.
        <span style='color:#00b4ff;font-size:0.82rem'>&nbsp;[Production: connects to NIMET / OpenWeatherMap API]</span>
    </div>
    """, unsafe_allow_html=True)

    # Session state init
    if 'monitored_cities' not in st.session_state:
        st.session_state.monitored_cities = ["Lagos", "Port Harcourt", "Kogi / Lokoja", "Abuja"]
    if 'city_db' not in st.session_state:
        st.session_state.city_db = {k: dict(v) for k, v in CITY_DB.items()}

    col_w1, col_w2 = st.columns([1, 1.8])

    with col_w1:
        st.markdown("<div style='color:#4a7a9b;font-size:0.8rem;text-transform:uppercase;letter-spacing:2px;margin-bottom:0.8rem'>Manage Monitored Cities</div>", unsafe_allow_html=True)

        # Monitored city list with remove buttons
        for city in list(st.session_state.monitored_cities):
            ca, cb = st.columns([3, 1])
            with ca:
                st.markdown(f"<div style='color:#e0eaf8;font-size:0.9rem;padding:0.3rem 0'>📍 {city}</div>", unsafe_allow_html=True)
            with cb:
                if st.button("✕", key=f"rm_{city}", help=f"Remove {city}"):
                    st.session_state.monitored_cities.remove(city)
                    st.rerun()

        # Adjust geo-features for existing city
        with st.expander("⚙️ Adjust city conditions"):
            if st.session_state.monitored_cities:
                edit_city = st.selectbox("Select city", st.session_state.monitored_cities, key="edit_city_sel")
                p = st.session_state.city_db[edit_city]
                new_drain  = st.number_input("Drainage capacity",  0,   60,   int(p['drain']),        key="adj_d")
                new_infi   = st.number_input("Infiltration",       0,   50,   int(p['infi']),         key="adj_i")
                new_runoff = st.number_input("Runoff coefficient",  0.0, 1.0,  float(p['runoff']),    step=0.01, key="adj_ro")
                new_slope  = st.number_input("Slope",              0.0, 10.0, float(p['slope']),      step=0.1,  key="adj_s")
                if st.button("Apply changes", key="apply_adj"):
                    st.session_state.city_db[edit_city].update(
                        drain=new_drain, infi=new_infi, runoff=new_runoff, slope=new_slope
                    )
                    st.success(f"✓ Updated {edit_city}")

        st.markdown("<hr class='styled-divider'>", unsafe_allow_html=True)

        # Add known city
        available = [c for c in st.session_state.city_db if c not in st.session_state.monitored_cities]
        if available:
            add_city = st.selectbox("Add a Nigerian city", ["— select —"] + available, key="add_known")
            if st.button("➕ Add City", key="add_known_btn"):
                if add_city != "— select —":
                    st.session_state.monitored_cities.append(add_city)
                    st.rerun()

        # Add custom location
        with st.expander("🗺️ Add custom location"):
            cust_name   = st.text_input("Location name",         key="cust_name")
            cust_rain   = st.number_input("Base rainfall (mm)",  0,   200,  80,   key="cust_r")
            cust_drain  = st.number_input("Drainage capacity",   0,   60,   25,   key="cust_d")
            cust_infi   = st.number_input("Soil infiltration",   0,   50,   18,   key="cust_i")
            cust_runoff = st.number_input("Runoff coefficient",  0.0, 1.0,  0.65, step=0.01, key="cust_ro")
            cust_slope  = st.number_input("Terrain slope",       0.0, 10.0, 3.0,  step=0.1,  key="cust_s")
            cust_note   = st.text_input("Notes (optional)",      key="cust_note")
            if st.button("Add Location", key="add_custom_btn"):
                if cust_name.strip():
                    st.session_state.city_db[cust_name] = dict(
                        base_rain=cust_rain, infi=cust_infi, drain=cust_drain,
                        runoff=cust_runoff, slope=cust_slope,
                        note=cust_note or "Custom location"
                    )
                    st.session_state.monitored_cities.append(cust_name)
                    st.rerun()

        st.markdown("<hr class='styled-divider'>", unsafe_allow_html=True)

        # Forecast settings
        st.markdown("<div style='color:#4a7a9b;font-size:0.8rem;text-transform:uppercase;letter-spacing:2px;margin-bottom:0.5rem'>Forecast Settings</div>", unsafe_allow_html=True)
        surge_mean = st.slider("Forecast rainfall surge (mm)", 0, 60, 15, key="surge",
                               help="Additional rainfall above each city's baseline — simulates a storm forecast event")
        auto_refresh = st.toggle("Auto-run on load", value=True, key="auto_refresh")
        run_btn = st.button("🚨 Run Monitoring Cycle", key="btn3")

    with col_w2:
        should_run = run_btn or (auto_refresh and 'last_monitor_results' not in st.session_state)

        if should_run and st.session_state.monitored_cities:
            np.random.seed()  # fresh seed so each cycle gives a new forecast
            results_ew      = {}
            forecasted_rain = {}
            for city in st.session_state.monitored_cities:
                p = st.session_state.city_db[city]
                f_rain = simulate_forecast_rainfall(p['base_rain'], surge_mean=surge_mean)
                forecasted_rain[city] = f_rain
                results_ew[city] = predict(f_rain, p['infi'], p['drain'], p['runoff'], p['slope'])
            st.session_state.last_monitor_results = results_ew
            st.session_state.last_forecasted_rain = forecasted_rain

        if 'last_monitor_results' in st.session_state and st.session_state.monitored_cities:
            results_ew      = {c: v for c, v in st.session_state.last_monitor_results.items() if c in st.session_state.monitored_cities}
            forecasted_rain = {c: v for c, v in st.session_state.last_forecasted_rain.items()  if c in st.session_state.monitored_cities}

            sorted_res = sorted(results_ew.items(), key=lambda x: -x[1])
            names   = [r[0] for r in sorted_res]
            probs   = [r[1] for r in sorted_res]
            bcolors = [risk_color(p) for p in probs]

            # Bar chart
            fig, ax = plt.subplots(figsize=(6.5, max(3.5, len(names) * 0.65 + 1.2)))
            bars = ax.barh(names[::-1], probs[::-1], color=bcolors[::-1], height=0.55)
            for bar, val in zip(bars, probs[::-1]):
                ax.text(min(val + 0.01, 0.97), bar.get_y() + bar.get_height()/2,
                        f'{val:.0%}', va='center', fontsize=8.5, color='#e0eaf8', fontfamily='monospace')
            ax.axvline(0.35, color='#00e676', linewidth=0.8, linestyle=':', alpha=0.5)
            ax.axvline(0.65, color='#ffab00', linewidth=0.8, linestyle=':', alpha=0.5)
            ax.set_xlim(0, 1.1)
            ax.set_xlabel("Flood Probability", fontsize=9)
            ax.set_title(f"Regional Flood Risk Monitor  |  +{surge_mean}mm Forecast Surge", fontsize=10.5, color='#e0eaf8', pad=12)
            ax.grid(axis='x', alpha=0.3)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            ax.tick_params(axis='y', labelsize=9)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

            high_risk   = [(c, p) for c, p in results_ew.items() if p >= 0.65]
            medium_risk = [(c, p) for c, p in results_ew.items() if 0.35 <= p < 0.65]
            low_risk    = [(c, p) for c, p in results_ew.items() if p < 0.35]

            if high_risk:
                st.markdown("<div class='alert-critical'>🚨 HIGH FLOOD RISK ALERT — IMMEDIATE ATTENTION REQUIRED</div>", unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"<div class='metric-card'><div class='value risk-high'>{len(high_risk)}</div><div class='label'>High Risk Zones</div></div>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"<div class='metric-card'><div class='value risk-medium'>{len(medium_risk)}</div><div class='label'>Moderate Risk</div></div>", unsafe_allow_html=True)
            with c3:
                st.markdown(f"<div class='metric-card'><div class='value risk-low'>{len(low_risk)}</div><div class='label'>Low Risk Zones</div></div>", unsafe_allow_html=True)

            st.markdown("<hr class='styled-divider'>", unsafe_allow_html=True)
            st.markdown("<div style='color:#4a7a9b;font-size:0.75rem;text-transform:uppercase;letter-spacing:2px;margin-bottom:0.8rem'>Location Detail</div>", unsafe_allow_html=True)

            for city, prob in sorted(results_ew.items(), key=lambda x: -x[1]):
                lbl, cls, icon = risk_label(prob)
                note  = st.session_state.city_db[city].get('note', '')
                color = risk_color(prob)
                frain = forecasted_rain.get(city, 0)
                st.markdown(f"""
                <div class='location-card'>
                    <div>
                        <div style='font-weight:600;color:#e0eaf8'>{icon} {city}</div>
                        <div style='font-size:0.75rem;color:#4a7a9b;margin-top:2px'>{note}</div>
                        <div style='font-size:0.72rem;color:#2a5a7c;margin-top:3px'>📡 Forecast rainfall: {frain:.1f} mm</div>
                    </div>
                    <div style='text-align:right'>
                        <div style='font-family:monospace;font-size:1.3rem;font-weight:700;color:{color}'>{prob:.0%}</div>
                        <div style='font-size:0.7rem;color:{color};text-transform:uppercase;letter-spacing:1px'>{lbl}</div>
                    </div>
                </div>""", unsafe_allow_html=True)

        elif not st.session_state.monitored_cities:
            st.markdown("""
            <div style='height:300px;display:flex;align-items:center;justify-content:center;
                        border:1px dashed #1a3a5c;border-radius:12px;flex-direction:column;gap:0.5rem'>
                <div style='font-size:2rem'>📍</div>
                <div style='color:#4a7a9b;font-size:0.9rem'>Add at least one city to monitor</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='height:380px;display:flex;align-items:center;justify-content:center;
                        border:1px dashed #1a3a5c;border-radius:12px;flex-direction:column;gap:0.5rem'>
                <div style='font-size:2.5rem'>🚨</div>
                <div style='color:#4a7a9b;font-size:0.9rem'>Click Run Monitoring Cycle to begin</div>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;margin-top:3rem;padding-top:1.5rem;
            border-top:1px solid #1a3a5c;color:#2a4a6c;font-size:0.78rem;
            font-family:monospace;letter-spacing:1px'>
    FLOODSENSE · AI FLOOD INTELLIGENCE · 3MTT NEXTGEN FELLOWSHIP · MVP DEMO
</div>
""", unsafe_allow_html=True)