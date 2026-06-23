"""
=============================================================================
  THE LOGPOSE - Bengaluru Parking Intelligence System
  Flipkart GridLock 2.0 - Round 2 | PS1
  Built for the Bengaluru Traffic Police (BTP)

  RUN: streamlit run app.py
=============================================================================
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path

# ── PATHS ───────────────────────────────────────────────────────────────────
# Dynamically get the directory where app.py is located
BASE_DIR  = Path(__file__).parent
DATA_DIR  = BASE_DIR / "data"
CLEAN_DIR = DATA_DIR / "cleaned"
OUT_DIR   = BASE_DIR / "outputs"
DASH_DIR  = OUT_DIR / "dashboard_data"
MAP_DIR   = OUT_DIR / "maps"
PLOT_DIR  = OUT_DIR / "plots"
MODEL_DIR = OUT_DIR / "model"

# ── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="The LogPose - BTP Intelligence",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SESSION STATE DEFAULTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEFAULTS = {
    "dark_mode": True,
    "active_preset": None,
    "active_preset_key": None,
    # Zone drill-down: when set, Offender tab shows these stations
    "drill_stations": None,
    "drill_zone_label": None,
    "page": "🏠 Command Center",
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

def navigate_to(target_page, drill_stations=None, drill_label=None):
    st.session_state.page = target_page
    if drill_stations is not None:
        st.session_state.drill_stations = drill_stations
        st.session_state.drill_zone_label = drill_label
    st.rerun()

# ── THEME ───────────────────────────────────────────────────────────────────
def get_theme():
    if st.session_state.dark_mode:
        return {
            "bg": "#0a0a14", "card_bg": "#13132a", "card_bg2": "#1a1a35",
            "text": "#e8e8f8", "accent": "#e94560", "accent2": "#4cc9f0",
            "accent3": "#f4a261", "border": "#2a2a4a", "neutral": "#8888aa",
            "success": "#52b788", "warning": "#f4a261", "danger": "#e94560",
            "gradient_start": "#e94560", "gradient_end": "#4cc9f0",
            "plotly_template": "plotly_dark",
            "mapstyle": "carto-darkmatter",
        }
    else:
        return {
            "bg": "#f0f2f8", "card_bg": "#ffffff", "card_bg2": "#f8f9fc",
            "text": "#1a1a2e", "accent": "#c0392b", "accent2": "#2980b9",
            "accent3": "#e67e22", "border": "#d0d4e8", "neutral": "#6c7a8d",
            "success": "#27ae60", "warning": "#e67e22", "danger": "#c0392b",
            "gradient_start": "#c0392b", "gradient_end": "#2980b9",
            "plotly_template": "plotly_white",
            "mapstyle": "carto-positron",
        }

T = get_theme()

# ── GLOBAL CSS ───────────────────────────────────────────────────────────────
_dark = st.session_state.dark_mode
_glass_bg = 'rgba(19,19,42,0.65)' if _dark else 'rgba(255,255,255,0.55)'
_glass_border = 'rgba(255,255,255,0.08)' if _dark else 'rgba(0,0,0,0.08)'
_sidebar_bg = 'rgba(8,8,26,0.92)' if _dark else 'rgba(232,236,245,0.95)'

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

/* ── Animated App Background ── */
.stApp {{
    background: {T['bg']};
    background-image: 
        radial-gradient(circle at 15% 50%, {'rgba(233,69,96,0.08)' if _dark else 'rgba(233,69,96,0.05)'}, transparent 40%),
        radial-gradient(circle at 85% 30%, {'rgba(76,201,240,0.08)' if _dark else 'rgba(76,201,240,0.05)'}, transparent 40%);
    background-size: 200% 200%;
    animation: bgShift 15s ease infinite;
    color: {T['text']};
}}

@keyframes bgShift {{
    0% {{ background-position: 0% 50%; }}
    50% {{ background-position: 100% 50%; }}
    100% {{ background-position: 0% 50%; }}
}}

/* ── Sidebar Ultimate Glassmorphism ── */
section[data-testid="stSidebar"] {{
    background: {'rgba(10,10,24,0.7)' if _dark else 'rgba(240,244,250,0.7)'} !important;
    backdrop-filter: blur(28px) saturate(200%);
    -webkit-backdrop-filter: blur(28px) saturate(200%);
    border-right: 1px solid {'rgba(255,255,255,0.1)' if _dark else 'rgba(0,0,0,0.05)'};
    box-shadow: 5px 0 30px rgba(0,0,0,0.1);
}}
section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {{
    background: transparent !important;
}}

/* Hide radio buttons */
section[data-testid="stSidebar"] .stRadio {{ display: none !important; }}

/* ── Sidebar Navigation Buttons ── */
section[data-testid="stSidebar"] [data-testid="stButton"] button {{
    display: flex; justify-content: flex-start; align-items: center; padding: 12px 16px;
    border-radius: 12px; font-size: .95rem; font-weight: 500;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    position: relative; overflow: hidden; height: auto;
}}
section[data-testid="stSidebar"] [data-testid="stButton"] button[kind="secondary"] {{
    background: transparent; border: 1px solid transparent; color: {T['neutral']};
    box-shadow: none;
}}
section[data-testid="stSidebar"] [data-testid="stButton"] button[kind="secondary"]:hover {{
    background: {'rgba(255,255,255,0.08)' if _dark else 'rgba(0,0,0,0.04)'};
    color: {T['text']};
    transform: translateX(4px);
    border-color: {'rgba(255,255,255,0.1)' if _dark else 'rgba(0,0,0,0.08)'};
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}}
section[data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"] {{
    background: {'rgba(233,69,96,0.15)' if _dark else 'rgba(233,69,96,0.1)'} !important;
    border: 1px solid {'rgba(233,69,96,0.4)' if _dark else 'rgba(233,69,96,0.3)'} !important;
    color: {T['accent']} !important; font-weight: 700;
    box-shadow: 0 4px 20px rgba(233,69,96,0.2) !important;
}}
section[data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"]:hover {{
    transform: translateX(4px);
    box-shadow: 0 6px 25px rgba(233,69,96,0.3) !important;
}}
@keyframes pulseDot {{
    0% {{ box-shadow: 0 0 0 0 rgba(233,69,96,0.7); }}
    70% {{ box-shadow: 0 0 0 6px rgba(233,69,96,0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(233,69,96,0); }}
}}

/* ── Shift & Timing Horizontal Pills ── */
[data-testid="stRadio"] > div[role="radiogroup"] {{
    display: flex; flex-direction: row; flex-wrap: wrap; gap: 10px;
}}
[data-testid="stRadio"] > div[role="radiogroup"] > label {{
    display: flex; align-items: center; justify-content: center;
    padding: 10px 20px; border-radius: 30px;
    background: {'rgba(255,255,255,0.05)' if _dark else 'rgba(0,0,0,0.03)'};
    border: 1px solid {'rgba(255,255,255,0.1)' if _dark else 'rgba(0,0,0,0.1)'};
    color: {T['neutral']}; font-weight: 600; font-size: 0.9rem;
    cursor: pointer; transition: all 0.3s ease;
}}
[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {{
    background: {'rgba(255,255,255,0.1)' if _dark else 'rgba(0,0,0,0.08)'};
    color: {T['text']};
    border-color: {'rgba(255,255,255,0.3)' if _dark else 'rgba(0,0,0,0.2)'};
}}
[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"],
[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {{
    background: {T['gradient_start']};
    background: linear-gradient(135deg, {T['gradient_start']}, {T['gradient_end']});
    border-color: transparent;
    color: #fff !important; font-weight: 700;
    box-shadow: 0 4px 15px {'rgba(233,69,96,0.4)' if _dark else 'rgba(233,69,96,0.2)'};
}}
[data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child {{
    display: none !important;
}}

/* ── KPI Cards (Next-Gen Glassmorphism) ── */
.kpi {{
    background: {'rgba(20,20,40,0.6)' if _dark else 'rgba(255,255,255,0.6)'};
    backdrop-filter: blur(24px) saturate(200%);
    -webkit-backdrop-filter: blur(24px) saturate(200%);
    border: 1px solid {'rgba(255,255,255,0.15)' if _dark else 'rgba(0,0,0,0.1)'};
    border-radius: 16px;
    padding: 22px 16px;
    text-align: center;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    height: 100%;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
}}
.kpi::before {{
    content: '';
    position: absolute; top: 0; left: 0; right: 0;
    height: 4px;
    background: linear-gradient(90deg, {T['gradient_start']}, {T['gradient_end']});
    opacity: 0.7; transition: all .4s;
}}
.kpi:hover {{
    transform: translateY(-8px) scale(1.02);
    box-shadow: 0 15px 40px rgba(233,69,96,.15), 0 8px 16px rgba(0,0,0,.2);
    border-color: {'rgba(233,69,96,0.3)' if _dark else 'rgba(233,69,96,0.2)'};
}}
.kpi:hover::before {{ opacity: 1; height: 6px; box-shadow: 0 0 15px {T['gradient_start']}; }}
.kpi-val {{
    font-size: 2.2rem; font-weight: 900;
    background: linear-gradient(135deg, {T['gradient_start']}, {T['gradient_end']});
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 8px 0 4px;
    font-family: 'Space Grotesk', sans-serif;
    line-height: 1.2;
    filter: drop-shadow(0px 4px 6px rgba(0,0,0,0.1));
}}
.kpi-lbl {{
    font-size: .75rem; color: {T['neutral']};
    text-transform: uppercase; letter-spacing: 1.5px;
    font-weight: 700;
}}

/* ── Alert KPI ── */
.kpi-alert {{
    border-color: {'rgba(233,69,96,0.5)' if _dark else 'rgba(233,69,96,0.4)'} !important;
    box-shadow: 0 0 25px rgba(233,69,96,0.15) inset, 0 8px 32px rgba(233,69,96,0.2);
    animation: alertPulse 3s infinite alternate;
}}
@keyframes alertPulse {{
    0% {{ box-shadow: 0 0 15px rgba(233,69,96,0.1) inset, 0 4px 20px rgba(233,69,96,0.1); }}
    100% {{ box-shadow: 0 0 30px rgba(233,69,96,0.25) inset, 0 10px 40px rgba(233,69,96,0.3); }}
}}
.kpi-alert::before {{ opacity: 1 !important; background: {T['danger']}; box-shadow: 0 0 20px {T['danger']}; }}
.kpi-alert .kpi-val {{
    background: linear-gradient(135deg, #ff4b4b, #ff8b8b);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}}

/* ── High-Tech Section Headers ── */
.sec-hdr {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.15rem; font-weight: 800;
    color: {T['text']};
    padding: 14px 0 10px;
    border-bottom: 2px solid transparent;
    border-image: linear-gradient(90deg, {T['accent']}, {T['accent2']}, transparent) 1;
    margin-bottom: 20px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}}

/* ── Plate cards (Deep Glassmorphism) ── */
.plate-card {{
    background: {'rgba(25,25,45,0.6)' if _dark else 'rgba(255,255,255,0.6)'};
    backdrop-filter: blur(16px) saturate(180%);
    -webkit-backdrop-filter: blur(16px) saturate(180%);
    border: 1px solid {'rgba(255,255,255,0.15)' if _dark else 'rgba(0,0,0,0.1)'};
    border-radius: 14px;
    padding: 16px 20px;
    margin: 8px 0;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
}}
.plate-card:hover {{
    box-shadow: 0 10px 30px rgba(76,201,240,.15);
    border-color: {'rgba(76,201,240,0.4)' if _dark else 'rgba(76,201,240,0.3)'};
    transform: translateY(-3px) scale(1.01);
}}
.plate-num {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.3rem; font-weight: 800;
    color: {T['accent2']};
    letter-spacing: 2.5px;
    text-shadow: 0 0 10px {'rgba(76,201,240,0.3)' if _dark else 'transparent'};
}}
.plate-badge {{
    display: inline-block; padding: 4px 12px;
    border-radius: 20px; font-size: 0.75rem; font-weight: 800;
    text-transform: uppercase; letter-spacing: 1px;
    backdrop-filter: blur(8px);
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}}
.badge-hab {{ background: linear-gradient(135deg, rgba(233,69,96,0.9), rgba(155,35,53,0.9)); color: white; box-shadow: 0 0 15px rgba(233,69,96,0.4); }}
.badge-rep {{ background: linear-gradient(135deg, rgba(244,162,97,0.9), rgba(230,126,34,0.9)); color: white; }}
.badge-ok  {{ background: linear-gradient(135deg, rgba(82,183,136,0.9), rgba(39,174,96,0.9)); color: white; }}
.badge-heavy {{ background: linear-gradient(135deg, rgba(76,201,240,0.9), rgba(41,128,185,0.9)); color: white; box-shadow: 0 0 15px rgba(76,201,240,0.3); }}

/* ── Directive cards (Neon glowing) ── */
.dir-card {{
    background: {'rgba(20,20,40,0.6)' if _dark else 'rgba(255,255,255,0.7)'};
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid {'rgba(255,255,255,0.1)' if _dark else 'rgba(0,0,0,0.1)'};
    border-left: 4px solid {T['accent']};
    border-radius: 12px;
    padding: 16px 20px; margin: 8px 0;
    font-size: .9rem; color: {T['text']};
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
}}
.dir-card:hover {{
    transform: translateX(6px);
    box-shadow: -4px 8px 25px rgba(233,69,96,.2);
    border-color: {'rgba(233,69,96,0.3)' if _dark else 'rgba(233,69,96,0.2)'};
    background: {'rgba(233,69,96,0.05)' if _dark else 'rgba(233,69,96,0.02)'};
}}
.dir-rank {{ font-weight: 900; color: {T['accent']}; font-size: .8rem; letter-spacing: 0.8px; }}

/* ── Anomaly banner (Pulsing neon) ── */
.anomaly-banner {{
    background: linear-gradient(90deg, rgba(123,13,30,0.9), rgba(233,69,96,0.9));
    color: white; padding: 16px 26px; border-radius: 14px;
    font-weight: 700; margin-bottom: 20px;
    animation: errorPulse 2s infinite;
    box-shadow: 0 4px 30px rgba(233,69,96,0.5);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.2);
    font-size: 1.05rem;
}}
@keyframes errorPulse {{ 
    0%,100% {{ box-shadow: 0 0 20px rgba(233,69,96,0.4); transform: scale(1); }} 
    50% {{ box-shadow: 0 0 40px rgba(233,69,96,0.8); transform: scale(1.005); }} 
}}

/* ── Drill-down banner (Dynamic Glassmorphism) ── */
.drill-banner {{
    background: {'rgba(30,30,60,0.7)' if _dark else 'rgba(240,248,255,0.8)'};
    backdrop-filter: blur(24px) saturate(200%);
    -webkit-backdrop-filter: blur(24px) saturate(200%);
    border: 1px solid {'rgba(76,201,240,0.4)' if _dark else 'rgba(76,201,240,0.3)'};
    border-radius: 16px; padding: 22px 28px; margin: 15px 0;
    box-shadow: 0 8px 32px rgba(76,201,240,0.15);
    position: relative; overflow: hidden;
}}
.drill-banner::after {{
    content: ''; position: absolute; top: -50%; left: -50%; width: 200%; height: 200%;
    background: radial-gradient(circle, {'rgba(76,201,240,0.1)' if _dark else 'rgba(76,201,240,0.05)'} 0%, transparent 70%);
    animation: rotateGlow 10s linear infinite;
    pointer-events: none;
}}
@keyframes rotateGlow {{ 100% {{ transform: rotate(360deg); }} }}
.drill-title {{ font-family: 'Space Grotesk', sans-serif; font-size: 1.4rem; font-weight: 800; color: {T['accent2']}; text-shadow: 0 0 10px rgba(76,201,240,0.3); }}
.drill-meta  {{ font-size: .85rem; color: {T['neutral']}; margin-top: 8px; letter-spacing: 0.5px; font-weight: 500; }}

/* ── Scrollable table ── */
.scroll-table {{ max-height: 480px; overflow-y: auto; }}

/* ── Next-Gen Page title ── */
.page-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.6rem; font-weight: 900;
    background: linear-gradient(135deg, {T['gradient_start']}, {T['gradient_end']}, {T['accent3']});
    background-size: 200% auto;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 5px;
    letter-spacing: -0.5px;
    animation: gradientShift 5s ease infinite;
    filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1));
}}
@keyframes gradientShift {{
    0% {{ background-position: 0% 50%; }}
    50% {{ background-position: 100% 50%; }}
    100% {{ background-position: 0% 50%; }}
}}
.page-sub {{ font-size: .95rem; color: {T['neutral']}; margin-top: 5px; letter-spacing: 0.3px; font-weight: 500; }}

/* ── Premium Animated Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 6px;
    background: {'rgba(20,20,40,0.5)' if _dark else 'rgba(255,255,255,0.5)'};
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-radius: 14px;
    padding: 6px;
    border: 1px solid {'rgba(255,255,255,0.1)' if _dark else 'rgba(0,0,0,0.05)'};
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 10px;
    padding: 10px 24px;
    font-weight: 700;
    font-size: .9rem;
    transition: all .3s cubic-bezier(0.25, 0.8, 0.25, 1);
    border: none !important;
    background: transparent;
    color: {T['neutral']};
}}
.stTabs [data-baseweb="tab"]:hover {{
    background: {'rgba(255,255,255,0.1)' if _dark else 'rgba(0,0,0,0.05)'};
    color: {T['text']};
    transform: translateY(-2px);
}}
.stTabs [aria-selected="true"] {{
    background: {'rgba(233,69,96,0.15)' if _dark else 'rgba(233,69,96,0.1)'} !important;
    color: {T['accent']} !important;
    box-shadow: 0 4px 15px rgba(233,69,96,0.2) !important;
    border: 1px solid {'rgba(233,69,96,0.3)' if _dark else 'rgba(233,69,96,0.2)'} !important;
}}
.stTabs [data-baseweb="tab-highlight"] {{ display: none; }}
.stTabs [data-baseweb="tab-border"] {{ display: none; }}

/* ── Ultimate Glass Expander ── */
.streamlit-expanderHeader {{
    background: {'rgba(25,25,45,0.6)' if _dark else 'rgba(255,255,255,0.7)'} !important;
    backdrop-filter: blur(16px);
    border-radius: 12px !important;
    border: 1px solid {'rgba(255,255,255,0.15)' if _dark else 'rgba(0,0,0,0.1)'} !important;
    font-weight: 700;
    font-size: 1.05rem;
    padding: 12px 18px !important;
    transition: all 0.3s ease;
}}
.streamlit-expanderHeader:hover {{
    background: {'rgba(35,35,60,0.8)' if _dark else 'rgba(240,248,255,0.9)'} !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    border-color: {'rgba(76,201,240,0.4)' if _dark else 'rgba(76,201,240,0.3)'} !important;
}}

/* ── Misc ── */
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
.stDataFrame {{
    border-radius: 16px; overflow: hidden;
    border: 1px solid {'rgba(255,255,255,0.1)' if _dark else 'rgba(0,0,0,0.1)'};
    box-shadow: 0 8px 30px rgba(0,0,0,0.1);
}}

/* ── Interactive Glowing Buttons ── */
.stButton > button {{
    background: {'rgba(25,25,45,0.7)' if _dark else 'rgba(255,255,255,0.8)'};
    border-radius: 12px;
    font-weight: 800;
    letter-spacing: 0.5px;
    transition: all .3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    backdrop-filter: blur(12px);
    border: 1px solid {'rgba(255,255,255,0.2)' if _dark else 'rgba(0,0,0,0.15)'};
    padding: 8px 24px;
    color: {T['text']};
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    position: relative;
    overflow: hidden;
    z-index: 1;
}}
.stButton > button::before {{
    content: ''; position: absolute; top: 0; left: 0; width: 100%; height: 100%;
    background: linear-gradient(135deg, {'rgba(233,69,96,0.1)' if _dark else 'rgba(233,69,96,0.05)'}, {'rgba(76,201,240,0.1)' if _dark else 'rgba(76,201,240,0.05)'});
    z-index: -1; transition: opacity 0.3s ease; opacity: 0;
}}
.stButton > button:hover::before {{ opacity: 1; }}
.stButton > button:hover {{
    transform: translateY(-3px) scale(1.02);
    box-shadow: 0 8px 25px rgba(76,201,240,.25), 0 0 10px rgba(233,69,96,.2);
    border-color: {'rgba(76,201,240,0.5)' if _dark else 'rgba(76,201,240,0.4)'};
    color: {T['accent2']};
}}
.stButton > button:active {{
    transform: translateY(1px) scale(0.98);
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}}

/* ── Primary Buttons ── */
.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, {T['gradient_start']}, {T['gradient_end']});
    color: white !important;
    border: none;
    box-shadow: 0 4px 20px rgba(233,69,96,0.4);
}}
.stButton > button[kind="primary"]:hover {{
    box-shadow: 0 10px 30px rgba(233,69,96,0.6), 0 0 20px rgba(76,201,240,0.4);
    transform: translateY(-4px) scale(1.03);
}}

/* ── Sidebar filter label ── */
.filter-label {{
    font-size: .8rem; font-weight: 800;
    color: {T['text']};
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 10px;
    padding: 0 4px;
    border-bottom: 1px solid {'rgba(255,255,255,0.1)' if _dark else 'rgba(0,0,0,0.1)'};
    padding-bottom: 6px;
}}

/* ── Sidebar stats ── */
.sidebar-stat {{
    display: flex; align-items: center; gap: 12px;
    font-size: .85rem; font-weight: 600; color: {T['neutral']};
    padding: 6px 0; transition: color 0.2s;
}}
.sidebar-stat:hover {{ color: {T['text']}; }}
.sidebar-stat-icon {{ font-size: 1.1rem; filter: drop-shadow(0 0 5px rgba(255,255,255,0.2)); }}

/* ── Vehicle profile card (Advanced Glassmorphism) ── */
.vehicle-profile {{
    background: {'rgba(20,20,40,0.6)' if _dark else 'rgba(255,255,255,0.7)'};
    backdrop-filter: blur(24px) saturate(200%);
    -webkit-backdrop-filter: blur(24px) saturate(200%);
    border: 1px solid {'rgba(255,255,255,0.15)' if _dark else 'rgba(0,0,0,0.1)'};
    border-radius: 20px;
    padding: 26px 30px;
    margin: 15px 0;
    transition: all .35s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    box-shadow: 0 10px 40px rgba(0,0,0,0.1);
}}
.vehicle-profile:hover {{
    box-shadow: 0 15px 50px rgba(76,201,240,.2), 0 0 20px rgba(233,69,96,.1);
    border-color: {'rgba(76,201,240,0.4)' if _dark else 'rgba(76,201,240,0.3)'};
    transform: translateY(-5px);
}}

/* ── High-Tech Glowing Status dot ── */
.status-dot {{
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 50%;
    margin-right: 8px;
    animation: neonGlow 2s ease-in-out infinite;
    background: currentColor;
}}
@keyframes neonGlow {{
    0%, 100% {{ box-shadow: 0 0 5px currentColor, 0 0 10px currentColor; transform: scale(1); }}
    50% {{ box-shadow: 0 0 15px currentColor, 0 0 25px currentColor; transform: scale(1.2); }}
}}

/* ── Custom CSS Loader Overlay (Next Level) ── */
[data-testid="stSpinner"] {{
    background: {'rgba(10,10,24,0.6)' if _dark else 'rgba(255,255,255,0.6)'};
    backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 20px 30px;
    border: 1px solid {'rgba(76,201,240,0.3)' if _dark else 'rgba(76,201,240,0.2)'};
    box-shadow: 0 10px 40px rgba(76,201,240,0.15);
    display: flex; align-items: center; justify-content: center;
    animation: spinnerPop 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards;
}}
[data-testid="stSpinner"] > div > div > div {{
    color: {T['accent2']} !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.1rem !important;
    font-weight: 800 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    text-shadow: 0 0 10px rgba(76,201,240,0.5) !important;
    animation: textPulse 1.5s infinite !important;
}}
@keyframes spinnerPop {{
    0% {{ transform: scale(0.8); opacity: 0; }}
    100% {{ transform: scale(1); opacity: 1; }}
}}
@keyframes textPulse {{
    0%, 100% {{ opacity: 1; text-shadow: 0 0 10px rgba(76,201,240,0.5); }}
    50% {{ opacity: 0.6; text-shadow: 0 0 20px rgba(76,201,240,0.8); }}
}}

/* ── Professional Enterprise UI Overrides ── */
.status-strip {{ display: flex; gap: 20px; margin-bottom: 15px; padding: 10px 0; border-bottom: 1px solid {T['border']}; }}
.status-pill {{
    font-weight: 600; font-size: 0.85rem; color: {T['neutral']};
    display: flex; align-items: center; gap: 6px; text-transform: uppercase; letter-spacing: 0.5px;
}}

.urgent-card {{
    background: {'rgba(233, 69, 96, 0.05)' if _dark else 'rgba(233, 69, 96, 0.02)'} !important;
    border: 1px solid {'rgba(233, 69, 96, 0.3)' if _dark else 'rgba(233, 69, 96, 0.2)'} !important;
    border-left: 4px solid #e94560 !important;
    border-radius: 4px !important;
}}

.glance-card {{
    background: {'#13132a' if _dark else '#ffffff'};
    border: 1px solid {T['border']};
    border-radius: 4px; padding: 12px 16px; margin-bottom: 10px;
    display: flex; align-items: center; justify-content: space-between; gap: 15px;
}}
.glance-text {{ font-size: 0.85rem; color: {T['text']}; font-weight: 500; }}

</style>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA LOADING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import os
import sqlite3
import zipfile
import json
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine
from pymongo import MongoClient
import certifi

load_dotenv(override=True)

try:
    SUPABASE_URI = st.secrets.get("SUPABASE_URI", os.getenv("SUPABASE_URI", ""))
    MONGO_URI = st.secrets.get("MONGO_URI", os.getenv("MONGO_URI", ""))
    MONGO_DB = st.secrets.get("MONGO_DB", os.getenv("MONGO_DB", "btp_db"))
except Exception:
    SUPABASE_URI = os.getenv("SUPABASE_URI", "")
    MONGO_URI = os.getenv("MONGO_URI", "")
    MONGO_DB = os.getenv("MONGO_DB", "btp_db")

DB_PATH = BASE_DIR / "btp_database.db"
ZIP_PATH = BASE_DIR / "btp_database.zip"

PARQUET_PATH = BASE_DIR / "data" / "features_slim.parquet"
if not MONGO_URI and not SUPABASE_URI and not DB_PATH.exists() and not ZIP_PATH.exists() and not PARQUET_PATH.exists():
    st.error("🚨 **CRITICAL DEPLOYMENT ERROR: No Database Found!** 🚨")
    st.markdown("""
    Your Streamlit Cloud app does not know how to connect to MongoDB because **the credentials are missing!**

    ### How to fix this RIGHT NOW:
    1. Open your Streamlit Cloud Dashboard.
    2. Click the three dots (**⋮**) next to your app and select **Settings**.
    3. Go to the **Secrets** tab.
    4. Paste the following exact text into the box, replacing the placeholders with your actual credentials:
    ```toml
    MONGO_URI="mongodb+srv://<username>:<password>@cluster0.../LogPose?retryWrites=true&w=majority"
    MONGO_DB="btp_db"
    ```
    5. Click **Save** and refresh this page.
    """)
    st.stop()

@st.cache_resource
def get_mongo_client():
    if MONGO_URI:
        try:
            client = MongoClient(MONGO_URI, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=5000)
            client.admin.command('ping')
            return client
        except Exception as e:
            st.error(f"MongoDB Connection Error (Check IP Whitelist or Secrets): {e}")
            pass
    return None

@st.cache_resource
def get_supabase_conn():
    if SUPABASE_URI:
        try:
            engine = create_engine(SUPABASE_URI)
            conn = engine.connect()
            return conn
        except Exception as e:
            st.error(f"Supabase Connection Error: {e}")
            pass
    return None

@st.cache_resource
def get_sqlite_conn():
    if not DB_PATH.exists() and ZIP_PATH.exists():
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(BASE_DIR)
    
    if DB_PATH.exists():
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        return conn
    return None

@st.cache_data(ttl=3600)
def load_features():
    parquet_path = BASE_DIR / "data" / "features_slim.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    mongo_client = get_mongo_client()
    if mongo_client:
        try:
            db = mongo_client[MONGO_DB]
            COLS = {
                "id":1, "latitude":1, "longitude":1,
                "vehicle_number":1, "vehicle_type":1, "vehicle_category":1,
                "police_station":1, "validation_status":1, "created_datetime_ist":1,
                "hour":1, "time_bucket":1, "is_heavy_vehicle":1,
                "primary_violation":1, "violation_count":1, "max_severity":1,
                "is_junction":1, "is_habitual_offender":1, "_id":0
            }
            data = list(db["dataset_features"].find({}, COLS))
            if data:
                return pd.DataFrame(data)
            else:
                st.warning("MongoDB connected, but 'dataset_features' is empty!")
        except Exception as e:
            st.error(f"MongoDB Data Fetch Error: {e}")
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def _load_json(p, coll_name=None):
    if coll_name:
        supa_conn = get_supabase_conn()
        if supa_conn:
            try:
                from sqlalchemy import text
                cursor = supa_conn.execute(text("SELECT data FROM json_store WHERE key=:key"), {"key": coll_name})
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
            except Exception:
                pass

        mongo_client = get_mongo_client()
        if mongo_client:
            try:
                db = mongo_client[MONGO_DB]
                docs = list(db[coll_name].find({}, {"_id": 0}))
                if len(docs) > 1:
                    return docs
                elif len(docs) == 1:
                    return docs[0]
            except Exception:
                pass

        conn = get_sqlite_conn()
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT data FROM json_store WHERE key=?", (coll_name,))
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
            except Exception:
                pass
    with open(p) as f: return json.load(f)

@st.cache_data(ttl=3600)
def _load_csv(p, coll_name=None):
    if coll_name:
        supa_conn = get_supabase_conn()
        if supa_conn:
            try:
                return pd.read_sql(f"SELECT * FROM {coll_name}", supa_conn)
            except Exception:
                pass

        mongo_client = get_mongo_client()
        if mongo_client:
            try:
                db = mongo_client[MONGO_DB]
                data = list(db[coll_name].find({}, {"_id": 0}))
                if data: return pd.DataFrame(data)
            except Exception:
                pass

        conn = get_sqlite_conn()
        if conn:
            try:
                return pd.read_sql(f"SELECT * FROM {coll_name}", conn)
            except Exception:
                pass
    return pd.read_csv(p)

def jload(p, coll_name=None):
    try: return _load_json(p, coll_name)
    except: return {}

def cload(p, coll_name=None):
    try: return _load_csv(p, coll_name)
    except: return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def build_vehicle_summary(reg_df, target_col):
    if "vehicle_number" not in reg_df.columns or len(reg_df) == 0:
        return pd.DataFrame()
    _count_col = "id" if "id" in reg_df.columns else reg_df.columns[0]
    _agg = {"total_violations": (_count_col, "count")}
    if "vehicle_category" in reg_df.columns:     _agg["vehicle_category"]  = ("vehicle_category", "first")
    if "vehicle_type" in reg_df.columns:          _agg["vehicle_type"]       = ("vehicle_type", "first")
    if target_col in reg_df.columns:              _agg["top_violation"]      = (target_col, lambda x: x.value_counts().index[0] if len(x) > 0 else "-")
    if "police_station" in reg_df.columns:        _agg["top_station"]        = ("police_station", lambda x: x.value_counts().index[0] if len(x) > 0 else "-")
    if "max_severity" in reg_df.columns:          _agg["avg_severity"]       = ("max_severity", "mean")
    if "is_habitual_offender" in reg_df.columns:  _agg["is_habitual"]        = ("is_habitual_offender", "max")
    if "is_heavy_vehicle" in reg_df.columns:      _agg["is_heavy"]           = ("is_heavy_vehicle", "max")
    if "violation_count" in reg_df.columns:       _agg["multi_offence"]      = ("violation_count", lambda x: int((x > 1).sum()))
    if "created_datetime_ist" in reg_df.columns:  _agg["last_seen"]          = ("created_datetime_ist", "max")
    veh_summary = (
        reg_df.groupby("vehicle_number")
        .agg(**_agg)
        .reset_index()
        .sort_values("total_violations", ascending=False)
    )
    if "avg_severity" in veh_summary.columns:
        veh_summary["avg_severity"] = veh_summary["avg_severity"].round(2)
    if "last_seen" in veh_summary.columns:
        veh_summary["last_seen"] = veh_summary["last_seen"].astype(str).str[:10]
    return veh_summary

df              = load_features()
model_summary   = jload(OUT_DIR / "model_summary.json", "model_summary")
priority_df     = cload(OUT_DIR / "enforcement_priority_ranked.csv", "enforcement_priority_ranked")
hotspot_df      = cload(OUT_DIR / "hotspot_clusters.csv", "hotspot_clusters")
peak_time_df    = cload(DASH_DIR / "peak_time_forecast.csv", "peak_time_forecast")
global_peak_df  = cload(DASH_DIR / "global_peak_hours.csv", "global_peak_hours")
hour_viol_mat   = cload(DASH_DIR / "hour_vs_violation_matrix.csv", "hour_vs_violation_matrix")
patrol_gap_df   = cload(DASH_DIR / "patrol_gap_analysis.csv", "patrol_gap_analysis")
reactive_data   = jload(DASH_DIR / "reactive_vs_proactive.json", "reactive_vs_proactive")
anomaly_data    = jload(DASH_DIR / "anomaly_detection.json", "anomaly_detection")
scita_data      = jload(DASH_DIR / "scita_sync.json", "scita_sync")
vehicle_idx     = cload(DASH_DIR / "vehicle_lookup_index.csv", "vehicle_lookup_index")
multi_viol_df   = cload(DASH_DIR / "multi_violation_profiles.csv", "multi_violation_profiles")
multi_summary   = jload(DASH_DIR / "multi_violation_summary.json", "multi_violation_summary")
veh_viol_mat    = cload(DASH_DIR / "vehicle_vs_violation_matrix.csv", "vehicle_vs_violation_matrix")
geohash_df      = cload(DASH_DIR / "geohash_grid_overlay.csv", "geohash_grid_overlay")
recommendations = jload(DASH_DIR / "recommendations.json", "recommendations")
time_block_df   = cload(DASH_DIR / "time_block_shifts.csv", "time_block_shifts")
habitual_df     = cload(DASH_DIR / "habitual_offenders.csv", "habitual_offenders")
station_ref_df  = cload(DASH_DIR / "station_reference.csv", "station_reference")
junction_ref_df = cload(DASH_DIR / "junction_reference.csv", "junction_reference")
quick_views     = jload(DASH_DIR / "quick_view_presets.json", "quick_view_presets")
dow_df          = cload(DASH_DIR / "day_of_week_trends.csv", "day_of_week_trends")
validation_data = jload(DASH_DIR / "validation_status.json", "validation_status")
offence_ref_df  = cload(DASH_DIR / "offence_filter_reference.csv", "offence_filter_reference")
live_pred_cfg   = jload(DASH_DIR / "live_prediction_config.json", "live_prediction_config")

target_col = "primary_violation_final" if "primary_violation_final" in df.columns else "primary_violation"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def kpi(label, value, col, alert=False):
    cls = "kpi kpi-alert" if alert else "kpi"
    col.markdown(f'<div class="{cls}"><div class="kpi-lbl">{label}</div><div class="kpi-val">{value}</div></div>',
                 unsafe_allow_html=True)

def sec(text):
    st.markdown(f'<div class="sec-hdr">{text}</div>', unsafe_allow_html=True)

def embed_map(html_path, height=540):
    p = Path(html_path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            html = f.read()
        st.components.v1.html(html, height=height, scrolling=False)
    else:
        st.warning(f"Map not found: {p.name}. Run parking_intelligence_pipeline.py first.")

def embed_plot(png_path, caption=""):
    p = Path(png_path)
    if p.exists():
        st.image(str(p), caption=caption, use_container_width=True)
    else:
        st.info(f"Plot not found: {p.name}")

def navigate_to(page, drill_stations=None, drill_label=None):
    """Programmatically navigate and optionally set drill-down context."""
    st.session_state.page = page
    st.session_state.drill_stations = drill_stations
    st.session_state.drill_zone_label = drill_label
    st.rerun()

def severity_badge(sev):
    sev = float(sev) if sev else 0
    if sev >= 4: return f'<span style="color:#e94560;font-weight:700;">{"●"*int(sev)} {sev:.1f}</span>'
    if sev >= 3: return f'<span style="color:#f4a261;font-weight:700;">{"●"*int(sev)} {sev:.1f}</span>'
    return f'<span style="color:#52b788;font-weight:700;">{"●"*int(sev)} {sev:.1f}</span>'


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIDEBAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PAGES = [
    "🏠 Command Center",
    "🗺️ Zone Maps",
    "📊 Priority Board",
    "🚨 Offender Registry",
    "⏱️ Shift & Timing",
    "🤖 AI Model",
    "⚙️ System",
]

with st.sidebar:
    # Brand
    st.markdown(f"""
    <div style="text-align:center; padding:18px 0 12px;">
        <div style="font-family:'Space Grotesk',sans-serif; font-size:1.8rem; font-weight:800;
                    background:linear-gradient(135deg,{T['gradient_start']},{T['gradient_end']});
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                    letter-spacing:-0.5px;">
            🧭 The LogPose
        </div>
        <div style="color:{T['neutral']}; font-size:.68rem; margin-top:4px; letter-spacing:1.5px;
                    text-transform:uppercase;">
            Bengaluru Traffic Police · Intelligence<br>
            <span style="color:{T['accent']}; font-weight:600; text-transform:none; letter-spacing:normal; font-size:.75rem;">~ by The LogPose Crew</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div style="height:1px;background:linear-gradient(90deg,transparent,{T["border"]},transparent);margin:8px 0 12px;"></div>', unsafe_allow_html=True)

    # ── Sidebar Navigation ──
    for p_name in PAGES:
        is_active = (st.session_state.page == p_name)
        if st.button(p_name, use_container_width=True, type="primary" if is_active else "secondary"):
            st.session_state.page = p_name
            if p_name != "🚨 Offender Registry":
                st.session_state.drill_stations = None
                st.session_state.drill_zone_label = None
            st.rerun()

    st.markdown(f'<div style="height:1px;background:linear-gradient(90deg,transparent,{T["border"]},transparent);margin:12px 0;"></div>', unsafe_allow_html=True)

    # Theme toggle
    dark = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
    if dark != st.session_state.dark_mode:
        st.session_state.dark_mode = dark
        st.rerun()

    st.markdown(f'<div style="height:1px;background:linear-gradient(90deg,transparent,{T["border"]},transparent);margin:12px 0;"></div>', unsafe_allow_html=True)

    # Global filters
    st.markdown(f'<div class="filter-label">Data Filters</div>', unsafe_allow_html=True)

    data_quality = st.selectbox("Data Quality", ["All Records", "Approved Only"],
                                index=0, label_visibility="collapsed")

    all_violations = offence_ref_df["violation_type"].tolist() if not offence_ref_df.empty else []
    selected_violations = st.multiselect("Violation Types", all_violations,
                                         default=[], placeholder="All violation types")

    st.markdown(f'<div style="height:1px;background:linear-gradient(90deg,transparent,{T["border"]},transparent);margin:12px 0;"></div>', unsafe_allow_html=True)



# ── Apply filters ────────────────────────────────────────────────────────────
with st.spinner("Processing data filters..."):
    dff = df
    if data_quality == "Approved Only" and "validation_status" in dff.columns:
        dff = dff[dff["validation_status"] == "approved"]
    if selected_violations and target_col in dff.columns:
        dff = dff[dff[target_col].isin(selected_violations)]

# alias
page = st.session_state.page


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ██████████████████  PAGE: COMMAND CENTER  ██████████████████████████████████
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if page == "🏠 Command Center":
    last_updated = df["created_datetime_ist"].max()[:10] if "created_datetime_ist" in df.columns else "2024-05-18"

    st.markdown(f'''
    <div style="display:flex;justify-content:space-between;align-items:flex-end;">
        <div>
            <div class="page-title" style="font-weight: 700; letter-spacing: -0.5px;">BTP Intelligence Operations</div>
            <div class="page-sub" style="color:{T['neutral']};">Bengaluru Traffic Police Command Center</div>
        </div>
        <div style="color:{T['neutral']}; font-size:0.8rem; padding-bottom:5px;">DATA CURRENT AS OF: <strong>{last_updated}</strong></div>
    </div>
    ''', unsafe_allow_html=True)
    
    # ── 1. Top-of-page Status Strip ──
    v_pct = validation_data.get("approved_pct", 0) if validation_data else 0
    s_pct = scita_data.get("sync_pct", 0) if scita_data else 0
    has_anom = anomaly_data.get("has_anomalies", False) if anomaly_data else False
    anom_msg = anomaly_data.get("warning_message", "1 Alert") if anomaly_data else ""

    st.markdown(f'''
    <div class="status-strip">
        <div class="status-pill">
            <span style="color:{'#52b788' if v_pct>=80 else '#e94560'}">{'🟢' if v_pct>=80 else '🔴'}</span> 
            Data Health: {v_pct:.1f}% Valid
        </div>
        <div class="status-pill">
            <span style="color:{'#52b788' if s_pct>=80 else '#e94560'}">{'🟢' if s_pct>=80 else '🔴'}</span> 
            SCITA Sync: {s_pct:.1f}%
        </div>
        <div class="status-pill">
            <span style="color:{'#e94560' if has_anom else '#52b788'}">{'🔴' if has_anom else '🟢'}</span> 
            {'Anomaly: ' + anom_msg if has_anom else 'System Normal'}
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # ── Split KPI Row ──
    # ── Top Row: Map + Directives ──
    col_left, col_right = st.columns([6, 5])

    with col_left:
        sec("Operations Map Viewer")
        if "mini_map_idx" not in st.session_state:
            st.session_state.mini_map_idx = 0
        mini_maps = [
            {"title": "Congestion Heatmap", "file": "01_congestion_heatmap.html"},
            {"title": "Hotspot Clusters & Priority", "file": "02_hotspot_clusters_priority.html"},
            {"title": "Night vs Day Patrol", "file": "03_night_vs_day.html"}
        ]
        current_map = mini_maps[st.session_state.mini_map_idx]
        st.markdown(f"<div style='text-align:center; font-weight:600; color:{T['text']};'>{current_map['title']}</div>", unsafe_allow_html=True)
        embed_map(MAP_DIR / current_map["file"], height=500)
        st.markdown(f"<div style='text-align:center; font-size:0.75rem; color:{T['neutral']}; padding: 6px 0 4px 0;'>Use the buttons below to switch maps &nbsp;·&nbsp; Showing {st.session_state.mini_map_idx + 1} of {len(mini_maps)}</div>", unsafe_allow_html=True)
        nav1, nav2, nav3 = st.columns(3)
        with nav1:
            if st.button("🔥 Congestion Heatmap", key="btn_map_0", use_container_width=True):
                st.session_state.mini_map_idx = 0
                st.rerun()
        with nav2:
            if st.button("🎯 Hotspot Clusters", key="btn_map_1", use_container_width=True):
                st.session_state.mini_map_idx = 1
                st.rerun()
        with nav3:
            if st.button("🌙 Night vs Day", key="btn_map_2", use_container_width=True):
                st.session_state.mini_map_idx = 2
                st.rerun()

    with col_right:
        sec("AI Enforcement Directives")
        if isinstance(recommendations, list) and recommendations:
            for rec in recommendations[:5]:
                pri   = rec.get("priority", "")
                stn   = rec.get("station", "")
                direc = rec.get("directive", "")
                u_cls = "urgent-card" if pri == 1 else ""
                tag = f"<b>[PRIORITY {pri}]</b> {stn}" if pri else f"{stn}"
                st.markdown(f'<div class="dir-card {u_cls}" style="padding:10px 14px;"><span class="dir-rank" style="color:{T["text"]}; font-size:0.8rem;">{tag}</span><br><span style="color:{T["neutral"]};font-size:0.9rem;">{direc}</span></div>',
                            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Bottom Row: KPIs + Stats ──
    bot_left, bot_right = st.columns([5, 6])

    with bot_left:
        sec("System Exports")
        ex1, ex2 = st.columns(2)
        with ex1:
            if not priority_df.empty:
                st.download_button("Download Priority CSV", priority_df.to_csv(index=False).encode(),
                                   "priority_ranked.csv", "text/csv", use_container_width=True)
        with ex2:
            if not hotspot_df.empty:
                st.download_button("Download Hotspots CSV", hotspot_df.to_csv(index=False).encode(),
                                   "hotspot_clusters.csv", "text/csv", use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        sec("Data Health & Integrity")
        dh1, dh2, dh3 = st.columns(3)
        kpi("Total Records", f"{len(dff):,}", dh1)
        kpi("Approval Rate", f"{v_pct:.1f}%", dh2, alert=(v_pct < 80))
        kpi("Anomalies Detected", "Yes" if has_anom else "None", dh3, alert=has_anom)

        st.markdown("<br>", unsafe_allow_html=True)
        sec("Operations Panel")
        scita_synced = scita_data.get('sent_to_scita', 0) if scita_data else 0
        scita_pend = scita_data.get('not_sent', 0) if scita_data else 0
        scita_alert = scita_pend > 0
        o1, o2 = st.columns(2)
        kpi("SCITA Synced", f"{scita_synced:,}", o1)
        kpi("SCITA Pending", f"{scita_pend:,}", o2, alert=scita_alert)

        st.markdown("<br>", unsafe_allow_html=True)
        hvy_c = int(dff['is_heavy_vehicle'].sum()) if "is_heavy_vehicle" in dff.columns else 0
        hvy_pct = (hvy_c / len(dff) * 100) if len(dff) > 0 else 0
        k_mv = multi_summary.get('total_multi_violation_records',0) if multi_summary else 0
        o3, o4 = st.columns(2)
        kpi("Heavy Vehicles", f"{hvy_c:,} <span style='font-size:0.5em;color:{T['neutral']}'>({hvy_pct:.1f}%)</span>", o3)
        kpi("Multi-Violation", f"{k_mv:,}", o4)

    with bot_right:
        sec("Quick Navigation")
        qcols1, qcols2 = st.columns(2), st.columns(2)
        if qcols1[0].button("Zone Maps", use_container_width=True): navigate_to("🗺️ Zone Maps")
        if qcols1[1].button("Offender Registry", use_container_width=True): navigate_to("🚨 Offender Registry")
        if qcols2[0].button("Priority Board", use_container_width=True): navigate_to("📊 Priority Board")
        if qcols2[1].button("Shift & Timing", use_container_width=True): navigate_to("⏱️ Shift & Timing")

        st.markdown("<br>", unsafe_allow_html=True)
        sec("AI & Operations Overview")
        ao1, ao2, ao3 = st.columns(3)
        acc = model_summary.get('ensemble_accuracy', 0) if model_summary else 0
        uniq_veh = dff["vehicle_number"].nunique() if "vehicle_number" in dff.columns else 1
        hab_c = int(dff["is_habitual_offender"].sum()) if "is_habitual_offender" in dff.columns else 0
        hab_pct = (hab_c / uniq_veh * 100) if uniq_veh else 0
        ao1.markdown(f'''
        <div class="kpi" title="Note: Class performance is uneven. Minor offences exhibit lower recall rates.">
            <div class="kpi-lbl">Model Accuracy *</div>
            <div class="kpi-val">{acc:.1%}</div>
        </div>
        ''', unsafe_allow_html=True)
        n_clus = model_summary.get('n_clusters','-') if model_summary else '-'
        kpi("Hotspot Clusters", f"{n_clus}", ao2)
        kpi("Habitual Offenders", f"{hab_c:,} <span style='font-size:0.5em;color:{T['neutral']}'>({hab_pct:.1f}%)</span>", ao3, alert=hab_c > 0)

        st.markdown("<br>", unsafe_allow_html=True)
        reac = reactive_data.get("reactive_pct", 0) if reactive_data else 0
        proa = reactive_data.get("proactive_pct", 0) if reactive_data else 0
        g_zone = patrol_gap_df.iloc[0]["geohash5"] if not patrol_gap_df.empty else "N/A"
        p_hour = global_peak_df.iloc[0]["hour"] if not global_peak_df.empty else "N/A"
        gi1, gi2, gi3 = st.columns(3)
        with gi1:
            st.markdown(f'<div class="glance-card"><div class="glance-text">Enforcement:<br><b>{proa:.1f}% proactive vs {reac:.1f}% reactive</b></div></div>', unsafe_allow_html=True)
            if st.button("Shift & Timing", key="g_shift1", use_container_width=True): navigate_to("⏱️ Shift & Timing")
        with gi2:
            st.markdown(f'<div class="glance-card urgent-card"><div class="glance-text">Patrol Gap:<br><b>Highest enforcement gap in {g_zone}</b></div></div>', unsafe_allow_html=True)
            if st.button("Zone Maps", key="g_zone1", use_container_width=True): navigate_to("🗺️ Zone Maps")
        with gi3:
            st.markdown(f'<div class="glance-card"><div class="glance-text">Peak Window:<br><b>City-wide violations peak at {p_hour}:00 IST</b></div></div>', unsafe_allow_html=True)
            if st.button("AI Model", key="g_shift2", use_container_width=True): navigate_to("🤖 AI Model")

        st.markdown("<br>", unsafe_allow_html=True)
        sec("Quick Vehicle Search")
        vcols = st.columns([3, 1])
        cmd_search = vcols[0].text_input("Enter Plate No.", placeholder="KA01AB1234", key="cmd_search_input", label_visibility="collapsed")
        if vcols[1].button("Search", use_container_width=True, type="primary"):
            if cmd_search:
                st.session_state.quick_search_vnum = cmd_search.strip().upper()
            navigate_to("🚨 Offender Registry")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ██████████████████  PAGE: ZONE MAPS  ███████████████████████████████████████
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "🗺️ Zone Maps":
    st.markdown('<div class="page-title">🗺️ Zone Maps</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Click a zone preset to zoom in - then tap "Show Offenders in Zone" to pull the full vehicle registry.</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Quick-View Presets ─────────────────────────────────────────────────
    if quick_views:
        sec("📍 Zone Quick-Select - Click to Zoom")
        n = len(quick_views)
        qcols = st.columns(n + 1)
        for i, (key, preset) in enumerate(quick_views.items()):
            is_active = st.session_state.active_preset_key == key
            if qcols[i].button(
                ("✅ " if is_active else "") + preset["label"],
                key=f"qv_{key}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                if is_active:
                    st.session_state.active_preset = None
                    st.session_state.active_preset_key = None
                else:
                    st.session_state.active_preset = preset
                    st.session_state.active_preset_key = key
                st.rerun()

        if st.session_state.active_preset:
            if qcols[-1].button("🔄 Reset City View", use_container_width=True):
                st.session_state.active_preset = None
                st.session_state.active_preset_key = None
                st.rerun()
        else:
            qcols[-1].markdown(
                f'<div style="text-align:center;padding:8px;color:{T["neutral"]};font-size:.78rem;">← Select a zone</div>',
                unsafe_allow_html=True)

    # ── Zone drill-down view ─────────────────────────────────────────────────
    ap = st.session_state.active_preset
    if ap:
        label    = ap["label"]
        center   = ap["center"]
        zoom     = ap["zoom"]
        stations = ap.get("stations", [])

        # filter data
        if stations and "police_station" in dff.columns:
            zone_df = dff[dff["police_station"].isin(stations)]
        else:
            zone_df = dff[
                dff["latitude"].between(center[0]-.02, center[0]+.02) &
                dff["longitude"].between(center[1]-.02, center[1]+.02)
            ]

        # banner
        st.markdown(f"""
        <div class="drill-banner">
            <div class="drill-title">🔎 Zone: {label}</div>
            <div class="drill-meta">
                Jurisdiction: {', '.join(stations) if stations else 'Area scan'}
                &nbsp;·&nbsp; Records: {len(zone_df):,}
                &nbsp;·&nbsp; Centre: {center[0]:.4f}°N, {center[1]:.4f}°E
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Zone KPIs
        zc1,zc2,zc3,zc4,zc5 = st.columns(5)
        kpi("Zone Violations", f"{len(zone_df):,}", zc1)
        kpi("Unique Vehicles",
            f"{zone_df['vehicle_number'].nunique():,}" if "vehicle_number" in zone_df.columns and len(zone_df) > 0 else "0",
            zc2)
        kpi("Avg Severity",
            f"{zone_df['max_severity'].mean():.2f}" if "max_severity" in zone_df.columns and len(zone_df) > 0 else "-",
            zc3)
        heavy = int(zone_df["is_heavy_vehicle"].sum()) if "is_heavy_vehicle" in zone_df.columns and len(zone_df) > 0 else 0
        kpi("Heavy Vehicles", f"{heavy:,}", zc4)
        top_v = zone_df[target_col].value_counts().index[0].replace("_"," ") if len(zone_df) > 0 and target_col in zone_df.columns else "-"
        kpi("Top Offence", top_v[:16], zc5)

        st.markdown("<br>", unsafe_allow_html=True)

        # ★ KEY ACTION BUTTON ★
        if len(zone_df) > 0:
            if st.button(
                f"🚨  SHOW ALL OFFENDERS IN THIS ZONE  ({len(zone_df['vehicle_number'].dropna().unique()) if 'vehicle_number' in zone_df.columns else 0} vehicles)",
                type="primary", use_container_width=True, key="zone_offender_btn",
            ):
                navigate_to("🚨 Offender Registry",
                            drill_stations=stations if stations else None,
                            drill_label=label)

        st.markdown("<br>", unsafe_allow_html=True)

        # Zoomed map
        if len(zone_df) > 0 and "latitude" in zone_df.columns:
            sev_labels = {1:"Very Low", 2:"Low", 3:"Medium", 4:"High", 5:"Very High"}
            sev_cols   = {1:"#52b788", 2:"#90e0ef", 3:"#f4a261", 4:"#e94560", 5:"#9b2335"}
            samp = zone_df.dropna(subset=["latitude","longitude"]).sample(min(6000, len(zone_df)), random_state=42)
            samp["sev_lbl"] = samp["max_severity"].map(sev_labels).fillna("Unknown")


            fig = px.scatter_map(
                samp, lat="latitude", lon="longitude",
                color="sev_lbl",
                color_discrete_map={v: sev_cols[k] for k,v in sev_labels.items()},
                category_orders={"sev_lbl": ["Very Low","Low","Medium","High","Very High"]},
                hover_data={"police_station": True, "vehicle_category": True, target_col: True, "vehicle_type": True, "latitude": False, "longitude": False},
                labels={"sev_lbl": "Severity", "police_station": "Station", "vehicle_category": "Vehicle", target_col: "Offence", "vehicle_type": "Type"},
                zoom=zoom, center={"lat": center[0], "lon": center[1]},
                map_style=T["mapstyle"],
                height=580,
                title=f"🔎 {label} - {len(zone_df):,} violations (color = severity)",
            )

            fig.update_layout(
                margin=dict(l=0,r=0,t=40,b=0),
                paper_bgcolor=T["bg"], font_color=T["text"],
                legend=dict(yanchor="top",y=.99,xanchor="left",x=.01,bgcolor="rgba(10,10,20,.7)"),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Hourly area chart
            if "hour" in zone_df.columns:
                hourly = zone_df.groupby("hour").size().reset_index(name="count")
                peak_h = int(hourly.loc[hourly["count"].idxmax(),"hour"])
                fig2 = px.area(
                    hourly, x="hour", y="count",
                    color_discrete_sequence=[T["accent"]],
                    title=f"Hourly Pattern - {label}",
                    template=T["plotly_template"],
                )
                st.success(f"🔮 Peak hour: **{peak_h}:00 IST** - Deploy enforcement between {peak_h}:00 - {(peak_h+2)%24}:00 IST for max impact.")
                fig2.update_layout(paper_bgcolor=T["bg"], plot_bgcolor=T["card_bg2"], font_color=T["text"])
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("No violations in this zone. Try a different preset or reset filters.")

        st.divider()

    # ── Full city map tabs ───────────────────────────────────────────────────
    # When a preset is active, collapse city maps into an expander to save space
    _map_expander = st.expander("🌆 Full City Maps (expand to view all of Bengaluru)", expanded=(ap is None))
    with _map_expander:
        sec("🌆 Full City Maps")
        mtabs = st.tabs(["🔴 Heatmap","🎯 Hotspots","🌙 Night vs Day","🟢 Severity","🔲 Geohash Grid","🔍 Patrol Gap"])

        with mtabs[0]:
            st.caption("Real-time violation density across Bengaluru. Red = highest concentration.")
            embed_map(MAP_DIR / "01_congestion_heatmap.html")

        with mtabs[1]:
            st.caption("DBSCAN clusters + Top 10 priority zones. Click pins for cluster stats.")
            embed_map(MAP_DIR / "02_hotspot_clusters_priority.html")

        with mtabs[2]:
            st.caption("Night patrol (10PM-6AM) vs daytime enforcement split.")
            embed_map(MAP_DIR / "03_night_vs_day.html")

        with mtabs[3]:
            st.caption("Violations color-coded by severity (green=low → red=critical).")
            if "latitude" in dff.columns:
                sev_labels = {1:"Very Low",2:"Low",3:"Medium",4:"High",5:"Very High"}
                sev_cols   = {1:"#52b788",2:"#90e0ef",3:"#f4a261",4:"#e94560",5:"#9b2335"}
                samp = dff.dropna(subset=["latitude","longitude"]).sample(min(8000,len(dff)), random_state=42)
                samp["sev_lbl"] = samp["max_severity"].map(sev_labels).fillna("Unknown")

                fig = px.scatter_map(
                    samp, lat="latitude", lon="longitude",
                    color="sev_lbl",
                    color_discrete_map={v:sev_cols[k] for k,v in sev_labels.items()},
                    category_orders={"sev_lbl":["Very Low","Low","Medium","High","Very High"]},
                    hover_data={"police_station": True, "vehicle_category": True, target_col: True, "latitude": False, "longitude": False},
                    labels={"sev_lbl": "Severity", "police_station": "Station", "vehicle_category": "Vehicle", target_col: "Offence"},
                    zoom=11, center={"lat":12.9716,"lon":77.5946},
                    map_style=T["mapstyle"], height=600,
                    title="🟢 Bengaluru - Violations by Severity",
                )
                fig.update_layout(margin=dict(l=0,r=0,t=36,b=0), paper_bgcolor=T["bg"], font_color=T["text"], title_font_size=13)
                st.plotly_chart(fig, use_container_width=True)

        with mtabs[4]:
            st.caption("Geohash6 grid (~1km² cells). Bubble size = violations. Color = severity.")
            if not geohash_df.empty:

                fig = px.scatter_map(
                    geohash_df.head(500), lat="centroid_lat", lon="centroid_lon",
                    size="violation_count", color="avg_severity",
                    color_continuous_scale=["#52b788","#f4a261","#e94560","#9b2335"],
                    hover_data={"geohash6": True, "violation_count": True, "avg_severity": True, "top_station": True, "centroid_lat": False, "centroid_lon": False},
                    labels={"geohash6": "Grid Cell", "violation_count": "Violations", "avg_severity": "Avg Severity", "top_station": "Top Station"},
                    zoom=11, center={"lat":12.9716,"lon":77.5946},
                    map_style=T["mapstyle"], height=600,
                    title="🔲 Geohash6 Grid - Violation Density per 1km² Cell",
                )
                fig.update_layout(margin=dict(l=0,r=0,t=36,b=0), paper_bgcolor=T["bg"], font_color=T["text"], title_font_size=13)
                st.plotly_chart(fig, use_container_width=True)
                st.info("Geohash6 bins violations into 1km² spatial cells - the winning feature from Round 1.")

        with mtabs[5]:
            st.caption("Patrol Gap = zones with high violations but low device coverage. Red = urgent.")
            if not patrol_gap_df.empty:
                fig = px.scatter_map(
                    patrol_gap_df.head(150), lat="centroid_lat", lon="centroid_lon",
                    size="violation_count", color="gap_score",
                    color_continuous_scale=["#52b788","#f4a261","#e94560"],
                    hover_data={"geohash5": True, "violation_count": True, "active_devices": True, "gap_score": True, "centroid_lat": False, "centroid_lon": False},
                    labels={"geohash5": "Zone", "violation_count": "Violations", "active_devices": "Active Devices", "gap_score": "Gap Score"},
                    zoom=11, center={"lat":12.9716,"lon":77.5946},
                    map_style=T["mapstyle"], height=600,
                    title="🔍 Patrol Gap - Underpoliced High-Violation Zones",
                )
                fig.update_layout(margin=dict(l=0,r=0,t=36,b=0), paper_bgcolor=T["bg"], font_color=T["text"], title_font_size=13)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Gap Score = violation density minus device coverage. Higher = bigger enforcement gap.")

    st.divider()

    # Junction vs Mid-Road + Station Drill
    st.divider()
    sec("🔀 Location Type & Station Filter")
    junc_col, stn_col = st.columns(2)
    with junc_col:
        junc_toggle = st.radio("Location Type:", ["All","Junction Only","Mid-Road Only"], horizontal=True)

    has_station_col = "police_station" in dff.columns
    stn_sel = "All Stations"
    with stn_col:
        if has_station_col:
            stn_sel = st.selectbox("Police Station:", ["All Stations"] + sorted(dff["police_station"].unique()))

    # Build filtered subset
    loc_df = dff.copy()
    if junc_toggle == "Junction Only" and "is_junction" in loc_df.columns:
        loc_df = loc_df[loc_df["is_junction"] == 1]
    elif junc_toggle == "Mid-Road Only" and "is_junction" in loc_df.columns:
        loc_df = loc_df[loc_df["is_junction"] == 0]
    if stn_sel != "All Stations" and has_station_col:
        loc_df = loc_df[loc_df["police_station"] == stn_sel]

    lc1,lc2,lc3,lc4 = st.columns(4)
    kpi("Records", f"{len(loc_df):,}", lc1)
    kpi("Unique Vehicles", f"{loc_df['vehicle_number'].nunique():,}" if "vehicle_number" in loc_df.columns else "-", lc2)
    kpi("Avg Severity", f"{loc_df['max_severity'].mean():.2f}" if "max_severity" in loc_df.columns and len(loc_df) > 0 else "-", lc3)
    kpi("Heavy", f"{int(loc_df['is_heavy_vehicle'].sum()):,}" if "is_heavy_vehicle" in loc_df.columns else "0", lc4)

    if stn_sel != "All Stations" and len(loc_df) > 0:
        if st.button(f"🚨 View Offenders for {stn_sel}", type="primary", use_container_width=True):
            navigate_to("🚨 Offender Registry", drill_stations=[stn_sel], drill_label=stn_sel)

        if "hour" in loc_df.columns:
            h = loc_df.groupby("hour").size().reset_index(name="count")
            fig = px.bar(h, x="hour", y="count",
                         color_discrete_sequence=[T["accent"]],
                         title=f"Hourly Pattern - {stn_sel}",
                         template=T["plotly_template"])
            fig.update_layout(paper_bgcolor=T["bg"], plot_bgcolor=T["card_bg2"], font_color=T["text"])
            st.plotly_chart(fig, use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ██████████████████  PAGE: PRIORITY BOARD  ██████████████████████████████████
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "📊 Priority Board":
    st.markdown('<div class="page-title">📊 Priority Board</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Ranked enforcement zones · Hotspot leaderboard · Data export</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Priority ranker
    sec("🏆 Enforcement Priority Ranker - Top 10 Zones")
    if not priority_df.empty:
        top10 = priority_df.head(10)
        fig = px.bar(
            top10, x="priority_score", y="police_station",
            orientation="h", color="priority_score",
            color_continuous_scale=["#4cc9f0","#e94560"],
            hover_data=["violation_total","avg_severity","avg_congestion","habitual_count","heavy_veh_count"],
            template=T["plotly_template"],
        )
        fig.update_layout(
            yaxis=dict(categoryorder="total ascending"),
            paper_bgcolor=T["bg"], plot_bgcolor=T["card_bg2"],
            font_color=T["text"], height=420,
            xaxis_title="Priority Score (0-1)",
            yaxis_title="",
            title="",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Clickable table rows - show offender button per station
        st.markdown("**Click a station row to view its offenders:**")
        for _, row in top10.iterrows():
            col_info, col_btn = st.columns([6, 2])
            with col_info:
                hab = int(row.get("habitual_count", 0))
                st.markdown(f"""
                <div class="plate-card">
                  <span style="font-weight:700; color:{T['accent2']};">#{int(row['rank'])} {row['police_station']}</span>
                  &nbsp; Score: <strong>{row['priority_score']:.4f}</strong>
                  &nbsp;·&nbsp; {int(row['violation_total']):,} violations
                  &nbsp;·&nbsp; Avg Sev: {row['avg_severity']:.2f}
                  {"&nbsp;·&nbsp;<span class='plate-badge badge-hab'>"+str(hab)+" HABITUAL</span>" if hab > 0 else ""}
                  &nbsp;·&nbsp; 🚛 {int(row.get('heavy_veh_count',0))} heavy
                </div>
                """, unsafe_allow_html=True)
            with col_btn:
                if st.button(f"🚨 Offenders", key=f"pri_off_{row['rank']}", use_container_width=True):
                    navigate_to("🚨 Offender Registry",
                                drill_stations=[row["police_station"]],
                                drill_label=row["police_station"])

    st.divider()

    # Hotspot leaderboard
    sec("📍 Top 15 Hotspot Clusters")
    if not hotspot_df.empty:
        top15 = hotspot_df.head(15).copy()
        top15.index = range(1, len(top15)+1)
        top15.index.name = "Rank"
        fig = px.bar(
            top15, x=top15.index, y="point_count",
            color="point_count",
            color_continuous_scale=["#4cc9f0","#e94560"],
            labels={"point_count":"Violations","index":"Cluster Rank"},
            template=T["plotly_template"],
        )
        fig.update_layout(paper_bgcolor=T["bg"], plot_bgcolor=T["card_bg2"], font_color=T["text"])
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            top15[["cluster","centroid_lat","centroid_lon","point_count"]],
            use_container_width=True,
            column_config={
                "centroid_lat": st.column_config.NumberColumn("Lat", format="%.4f"),
                "centroid_lon": st.column_config.NumberColumn("Lon", format="%.4f"),
                "point_count": st.column_config.NumberColumn("Violations", format="%d"),
            },
        )

    st.divider()

    # Export
    sec("📥 Download Reports")
    ec1,ec2,ec3 = st.columns(3)
    with ec1:
        if not priority_df.empty:
            st.download_button("📥 Priority Ranker CSV", priority_df.to_csv(index=False).encode(),
                               "priority_ranked.csv", "text/csv", use_container_width=True)
    with ec2:
        if not hotspot_df.empty:
            st.download_button("📥 Hotspot Clusters CSV", hotspot_df.to_csv(index=False).encode(),
                               "hotspot_clusters.csv", "text/csv", use_container_width=True)
    with ec3:
        st.download_button("📥 Full Dataset CSV", dff.to_csv(index=False).encode(),
                           "filtered_dataset.csv", "text/csv", use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ██████████████████  PAGE: OFFENDER REGISTRY  ███████████████████████████████
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "🚨 Offender Registry":

    # ── Determine working dataset ──────────────────────────────────────────
    drill_stations = st.session_state.drill_stations
    drill_label    = st.session_state.drill_zone_label

    if drill_stations and "police_station" in dff.columns:
        reg_df = dff[dff["police_station"].isin(drill_stations)]
        context = f"Zone: {drill_label} ({', '.join(drill_stations)})"
    else:
        reg_df = dff
        context = "All Bengaluru - Full Registry"

    # ── Header ────────────────────────────────────────────────────────────
    st.markdown('<div class="page-title">🚨 Offender Registry</div>', unsafe_allow_html=True)
    if drill_stations:
        st.markdown(f"""
        <div class="drill-banner">
            <div class="drill-title">🔎 Drill-Down Active: {drill_label}</div>
            <div class="drill-meta">Showing vehicles from: {', '.join(drill_stations)} · {len(reg_df):,} records</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🔄 Clear Filter - Show All Offenders", key="clear_drill"):
            st.session_state.drill_stations = None
            st.session_state.drill_zone_label = None
            st.rerun()
    else:
        st.markdown(f'<div class="page-sub">{context}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Registry KPIs ──────────────────────────────────────────────────────
    unique_vehs = reg_df["vehicle_number"].nunique() if "vehicle_number" in reg_df.columns else 0
    hab_in_zone = int(reg_df["is_habitual_offender"].sum()) if "is_habitual_offender" in reg_df.columns else 0
    heavy_in_zone = int(reg_df["is_heavy_vehicle"].sum()) if "is_heavy_vehicle" in reg_df.columns else 0
    multi_in_zone = int((reg_df["violation_count"] > 1).sum()) if "violation_count" in reg_df.columns else 0

    rk1,rk2,rk3,rk4,rk5 = st.columns(5)
    kpi("Total Records",       f"{len(reg_df):,}",      rk1)
    kpi("Unique Vehicles",     f"{unique_vehs:,}",       rk2)
    kpi("Habitual Offenders",  f"{hab_in_zone:,}",       rk3, alert=hab_in_zone>0)
    kpi("Heavy Vehicles",      f"{heavy_in_zone:,}",     rk4)
    kpi("Multi-Offence",       f"{multi_in_zone:,}",     rk5)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Build vehicle-level summary for this context ───────────────────────
    with st.spinner("Aggregating Offender Data..."):
        if "vehicle_number" in reg_df.columns and len(reg_df) > 0:
            veh_summary = build_vehicle_summary(reg_df, target_col)
        else:
            veh_summary = pd.DataFrame()

    # ── Helper: render a single vehicle's full history card ────────────────
    def _render_vehicle_history(vnum, source_df, card_key=""):
        """Render a rich vehicle profile + violation history inline."""
        vdata = source_df[source_df["vehicle_number"] == vnum] if "vehicle_number" in source_df.columns else pd.DataFrame()
        if vdata.empty:
            st.warning(f"No records found for {vnum}")
            return

        is_hab  = int(vdata["is_habitual_offender"].max()) if "is_habitual_offender" in vdata.columns else 0
        is_heav = int(vdata["is_heavy_vehicle"].max()) if "is_heavy_vehicle" in vdata.columns else 0
        avg_sev = float(vdata["max_severity"].mean()) if "max_severity" in vdata.columns else 0
        top_stn = vdata["police_station"].value_counts().index[0] if "police_station" in vdata.columns and len(vdata) > 0 else "-"
        top_vio = vdata[target_col].value_counts().index[0] if target_col in vdata.columns and len(vdata) > 0 else "-"
        veh_cat = vdata["vehicle_category"].iloc[0] if "vehicle_category" in vdata.columns else "-"
        veh_typ = vdata["vehicle_type"].iloc[0] if "vehicle_type" in vdata.columns else "-"

        hab_b  = '<span class="plate-badge badge-hab">⚠ HABITUAL</span>' if is_hab else ""
        hvy_b  = '<span class="plate-badge badge-heavy">🚛 HEAVY</span>' if is_heav else ""
        ok_b   = '<span class="plate-badge badge-ok">✓ NORMAL</span>' if not is_hab and not is_heav else ""

        # ── Profile card ──
        st.markdown(f"""
        <div style="background:{T['card_bg2']};border:1px solid {T['border']};border-radius:14px;
                    padding:20px 24px;margin:8px 0;">
            <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;">
                <span class="plate-num" style="font-size:1.5rem;">{vnum}</span>
                {hab_b} {hvy_b} {ok_b}
            </div>
            <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-top:14px;">
                <div><span style="color:{T['neutral']};font-size:.72rem;text-transform:uppercase;">Violations</span>
                     <div style="font-weight:800;font-size:1.3rem;color:{T['accent']};">{len(vdata)}</div></div>
                <div><span style="color:{T['neutral']};font-size:.72rem;text-transform:uppercase;">Avg Severity</span>
                     <div style="font-weight:800;font-size:1.3rem;color:{'#e94560' if avg_sev>=4 else '#f4a261' if avg_sev>=3 else '#52b788'};">{avg_sev:.2f}</div></div>
                <div><span style="color:{T['neutral']};font-size:.72rem;text-transform:uppercase;">Category</span>
                     <div style="font-weight:700;font-size:1rem;">{veh_cat}</div></div>
                <div><span style="color:{T['neutral']};font-size:.72rem;text-transform:uppercase;">Type</span>
                     <div style="font-weight:700;font-size:1rem;">{veh_typ}</div></div>
                <div><span style="color:{T['neutral']};font-size:.72rem;text-transform:uppercase;">Primary Station</span>
                     <div style="font-weight:700;font-size:1rem;">{top_stn}</div></div>
                <div><span style="color:{T['neutral']};font-size:.72rem;text-transform:uppercase;">Top Offence</span>
                     <div style="font-weight:700;font-size:1rem;">{str(top_vio).replace('_',' ')}</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Violation history table ──
        st.markdown(f"**📜 Complete Violation History - {len(vdata)} records**")
        hist_cols = [c for c in ["created_datetime_ist","police_station","junction_name",
                                  target_col,"violation_type","max_severity","vehicle_category",
                                  "location","is_junction","time_bucket"] if c in vdata.columns]
        if hist_cols:
            sort_col = "created_datetime_ist" if "created_datetime_ist" in vdata.columns else hist_cols[0]
            st.dataframe(
                vdata[hist_cols].sort_values(sort_col, ascending=False),
                use_container_width=True, hide_index=True,
                column_config={
                    "max_severity": st.column_config.NumberColumn("Severity", format="%.1f"),
                },
            )
        st.download_button(
            f"📥 Download {vnum} history",
            vdata.to_csv(index=False).encode(),
            f"violations_{vnum}.csv", "text/csv",
            key=f"dl_{card_key}_{vnum}",
        )

    # ── TABS ──────────────────────────────────────────────────────────────
    reg_tabs = st.tabs([
        "🔍 Search Vehicle",
        "📋 All Vehicles",
        "⚠️ Habitual Offenders",
        "🚛 Heavy Vehicles",
        "🔴 Multi-Offence",
        "📥 Export",
    ])

    # ══════════════════════════════════════════════════════════════════════
    # TAB 0 - SEARCH
    # ══════════════════════════════════════════════════════════════════════
    with reg_tabs[0]:
        sec("🔍 Vehicle Number Plate Lookup")
        
        # Populate from Quick Search if available
        if "quick_search_vnum" in st.session_state:
            def_val = st.session_state.quick_search_vnum
            del st.session_state.quick_search_vnum
        else:
            def_val = ""
            
        search_q = st.text_input(
            "Vehicle Number",
            value=def_val,
            placeholder="e.g.  KA01AB1234  or  GL0042",
            key="reg_search",
        )

        if search_q:
            src = search_q.strip().upper()
            if "vehicle_number" in reg_df.columns:
                hits = reg_df[reg_df["vehicle_number"].str.contains(src, na=False)]
            else:
                hits = pd.DataFrame()

            if hits.empty:
                st.warning(f"No records found for **{src}** in current scope.")
            else:
                uniq = hits["vehicle_number"].unique()
                st.success(f"Found **{len(uniq)} vehicle(s)** matching \"{src}\" · **{len(hits):,} total violations**")
                for idx_v, vnum in enumerate(uniq[:15]):
                    with st.expander(f"🚗  {vnum}  -  {len(hits[hits['vehicle_number']==vnum])} violations", expanded=(idx_v==0)):
                        _render_vehicle_history(vnum, hits, card_key=f"search_{idx_v}")
        else:
            st.info("💡 Type a full or partial plate number above to search. Results show full violation history for each matching vehicle.")

    # ══════════════════════════════════════════════════════════════════════
    # TAB 1 - ALL VEHICLES TABLE
    # ══════════════════════════════════════════════════════════════════════
    with reg_tabs[1]:
        sec(f"📋 All Vehicles - {context}")
        if veh_summary.empty:
            st.info("No vehicle data available.")
        else:
            # ── Filter row ──
            fc1, fc2, fc3 = st.columns(3)
            min_v = fc1.number_input("Min violations", 1, 100, 1, key="all_min_v")
            _cat_opts = (["All"] + sorted(veh_summary["vehicle_category"].dropna().unique().tolist())) if "vehicle_category" in veh_summary.columns else ["All"]
            sel_cat = fc2.selectbox("Vehicle Category", _cat_opts, key="all_cat")
            _sort_opts = [c for c in ["total_violations","avg_severity","multi_offence"] if c in veh_summary.columns]
            if not _sort_opts: _sort_opts = ["total_violations"]
            srt_by = fc3.selectbox("Sort by", _sort_opts, key="all_sort")

            filt = veh_summary[veh_summary["total_violations"] >= min_v]
            if sel_cat != "All" and "vehicle_category" in filt.columns:
                filt = filt[filt["vehicle_category"] == sel_cat]
            filt = filt.sort_values(srt_by, ascending=False).reset_index(drop=True)

            st.caption(f"Showing **{len(filt):,}** vehicles · sorted by {srt_by}")

            # ── Streamlit dataframe with column config (fast, no Styler crash) ──
            _display_cols = [c for c in ["vehicle_number","total_violations","avg_severity","vehicle_category",
                                          "top_station","top_violation","is_habitual","is_heavy","last_seen"]
                             if c in filt.columns]
            _col_cfg = {
                "vehicle_number": st.column_config.TextColumn("Plate Number", width="medium"),
            }
            if "total_violations" in filt.columns:
                _col_cfg["total_violations"] = st.column_config.NumberColumn("Violations", format="%d")
            if "avg_severity" in filt.columns:
                _col_cfg["avg_severity"] = st.column_config.NumberColumn("Avg Sev", format="%.2f")
            if "is_habitual" in filt.columns:
                _col_cfg["is_habitual"] = st.column_config.CheckboxColumn("Habitual", default=False)
            if "is_heavy" in filt.columns:
                _col_cfg["is_heavy"] = st.column_config.CheckboxColumn("Heavy", default=False)

            selected = st.dataframe(
                filt[_display_cols],
                use_container_width=True,
                hide_index=True,
                column_config=_col_cfg,
                height=460,
                on_select="rerun",
                selection_mode="single-row",
            )

            # ── Show details for selected row ──
            sel_rows = selected.get("selection", {}).get("rows", []) if selected else []
            if sel_rows:
                sel_idx = sel_rows[0]
                if sel_idx < len(filt):
                    sel_plate = filt.iloc[sel_idx]["vehicle_number"]
                    st.markdown("---")
                    sec(f"📋 Vehicle Details - {sel_plate}")
                    _render_vehicle_history(sel_plate, reg_df, card_key=f"all_{sel_idx}")
            else:
                st.info("👆 Click a row in the table above to view that vehicle's full violation history.")

            # Download
            st.download_button("📥 Download Filtered List (CSV)", filt.to_csv(index=False).encode(),
                               "offender_plates.csv", "text/csv", use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 2 - HABITUAL OFFENDERS
    # ══════════════════════════════════════════════════════════════════════
    with reg_tabs[2]:
        sec("⚠️ Habitual Offenders (5+ violations on record)")
        hab_zone = veh_summary[veh_summary["is_habitual"] == 1].copy() if not veh_summary.empty and "is_habitual" in veh_summary.columns else pd.DataFrame()

        if hab_zone.empty:
            st.success("✅ No habitual offenders in this scope.")
        else:
            hk1,hk2,hk3 = st.columns(3)
            kpi("Habitual Count", f"{len(hab_zone)}", hk1, alert=True)
            kpi("Highest Violations", f"{int(hab_zone['total_violations'].max())}", hk2, alert=True)
            _hab_sev = f"{hab_zone['avg_severity'].mean():.2f}" if "avg_severity" in hab_zone.columns else "-"
            kpi("Avg Severity", _hab_sev, hk3)
            st.markdown("<br>", unsafe_allow_html=True)

            _hab_disp = [c for c in ["vehicle_number","total_violations","avg_severity",
                                      "vehicle_category","top_station","top_violation","last_seen"]
                         if c in hab_zone.columns]
            _hab_cfg = {"vehicle_number": st.column_config.TextColumn("Plate", width="medium")}
            if "total_violations" in hab_zone.columns:
                _hab_cfg["total_violations"] = st.column_config.NumberColumn("Violations", format="%d")
            if "avg_severity" in hab_zone.columns:
                _hab_cfg["avg_severity"] = st.column_config.NumberColumn("Avg Sev", format="%.2f")

            hab_sel = st.dataframe(
                hab_zone[_hab_disp],
                use_container_width=True, hide_index=True,
                column_config=_hab_cfg, height=400,
                on_select="rerun", selection_mode="single-row",
            )
            hab_sel_rows = hab_sel.get("selection", {}).get("rows", []) if hab_sel else []
            if hab_sel_rows:
                h_idx = hab_sel_rows[0]
                if h_idx < len(hab_zone):
                    h_plate = hab_zone.iloc[h_idx]["vehicle_number"]
                    st.markdown("---")
                    sec(f"📋 Full History - {h_plate}")
                    _render_vehicle_history(h_plate, reg_df, card_key=f"hab_{h_idx}")
            else:
                st.info("👆 Click any habitual offender to see their full violation history.")

            st.download_button("📥 Download Habitual Offenders", hab_zone.to_csv(index=False).encode(),
                               "habitual_offenders.csv", "text/csv", use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 3 - HEAVY VEHICLES
    # ══════════════════════════════════════════════════════════════════════
    with reg_tabs[3]:
        sec("🚛 Heavy Vehicles (5× Congestion Weight)")
        heavy_zone = veh_summary[veh_summary["is_heavy"] == 1].copy() if not veh_summary.empty and "is_heavy" in veh_summary.columns else pd.DataFrame()

        if heavy_zone.empty:
            st.success("No heavy vehicles in this scope.")
        else:
            hv1,hv2 = st.columns(2)
            kpi("Heavy Vehicles", f"{len(heavy_zone)}", hv1)
            _hv_sev = f"{heavy_zone['avg_severity'].mean():.2f}" if "avg_severity" in heavy_zone.columns else "-"
            kpi("Avg Severity", _hv_sev, hv2)

            _hv_disp = [c for c in ["vehicle_number","vehicle_category","total_violations",
                                     "avg_severity","top_station","top_violation","last_seen"]
                        if c in heavy_zone.columns]
            _hv_cfg = {"vehicle_number": st.column_config.TextColumn("Plate", width="medium")}
            if "avg_severity" in heavy_zone.columns:
                _hv_cfg["avg_severity"] = st.column_config.NumberColumn("Avg Sev", format="%.2f")
            if "total_violations" in heavy_zone.columns:
                _hv_cfg["total_violations"] = st.column_config.NumberColumn("Violations", format="%d")

            hv_sel = st.dataframe(
                heavy_zone[_hv_disp],
                use_container_width=True, hide_index=True,
                column_config=_hv_cfg, height=400,
                on_select="rerun", selection_mode="single-row",
            )
            hv_sel_rows = hv_sel.get("selection", {}).get("rows", []) if hv_sel else []
            if hv_sel_rows:
                hv_idx = hv_sel_rows[0]
                if hv_idx < len(heavy_zone):
                    hv_plate = heavy_zone.iloc[hv_idx]["vehicle_number"]
                    st.markdown("---")
                    sec(f"📋 Full History - {hv_plate}")
                    _render_vehicle_history(hv_plate, reg_df, card_key=f"hv_{hv_idx}")
            else:
                st.info("👆 Click any heavy vehicle row to see its full violation history.")

            st.download_button("📥 Download Heavy Vehicle List", heavy_zone.to_csv(index=False).encode(),
                               "heavy_vehicles.csv", "text/csv", use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 4 - MULTI-OFFENCE
    # ══════════════════════════════════════════════════════════════════════
    with reg_tabs[4]:
        sec("🔴 Multi-Offence Records (2+ violations simultaneously)")
        if not multi_viol_df.empty:
            if drill_stations and "police_station" in multi_viol_df.columns:
                multi_zone = multi_viol_df[multi_viol_df["police_station"].isin(drill_stations)]
            else:
                multi_zone = multi_viol_df
            mk1,mk2,mk3 = st.columns(3)
            kpi("Multi-Offence Records", f"{len(multi_zone):,}", mk1)
            kpi("Avg Violations/Record", f"{multi_zone['violation_count'].mean():.1f}" if "violation_count" in multi_zone.columns else "-", mk2)
            kpi("Max Simultaneous", f"{int(multi_zone['violation_count'].max())}" if "violation_count" in multi_zone.columns else "-", mk3, alert=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(multi_zone.head(200), use_container_width=True, hide_index=True, height=400)
            st.download_button("📥 Download Multi-Offence Records", multi_zone.to_csv(index=False).encode(),
                               "multi_offence.csv", "text/csv", use_container_width=True)
        else:
            st.info("No multi-offence data available.")

    # ══════════════════════════════════════════════════════════════════════
    # TAB 5 - EXPORT
    # ══════════════════════════════════════════════════════════════════════
    with reg_tabs[5]:
        sec("📥 Export Zone Data")
        st.markdown(f"**Scope:** {context}")
        x1,x2,x3 = st.columns(3)
        with x1:
            if not veh_summary.empty:
                st.download_button("📥 Vehicle Summary CSV", veh_summary.to_csv(index=False).encode(),
                                   "zone_vehicle_summary.csv", "text/csv", use_container_width=True)
        with x2:
            st.download_button("📥 All Violation Records CSV", reg_df.to_csv(index=False).encode(),
                               "zone_violations.csv", "text/csv", use_container_width=True)
        with x3:
            if "is_habitual_offender" in reg_df.columns:
                hab_export = reg_df[reg_df["is_habitual_offender"] == 1]
                st.download_button("📥 Habitual Offenders CSV", hab_export.to_csv(index=False).encode(),
                                   "zone_habitual.csv", "text/csv", use_container_width=True)

    # ── Vehicle vs Violation matrix ───────────────────────────────────────
    st.divider()
    sec("🔥 Vehicle × Violation Heatmap")
    if not veh_viol_mat.empty:
        mx = veh_viol_mat.set_index(veh_viol_mat.columns[0])
        fig = px.imshow(
            mx.values,
            labels=dict(x="Violation Type", y="Vehicle Category", color="Count"),
            x=mx.columns.tolist(), y=mx.index.tolist(),
            color_continuous_scale=["#0a0a14","#4cc9f0","#e94560"],
            template=T["plotly_template"], aspect="auto",
        )
        fig.update_layout(paper_bgcolor=T["bg"], font_color=T["text"], height=350)
        st.plotly_chart(fig, use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ██████████████████  PAGE: SHIFT & TIMING  ██████████████████████████████████
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "⏱️ Shift & Timing":
    st.markdown('<div class="page-title">⏱️ Shift & Timing Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Peak hour forecasting · Shift analysis · Day-of-week trends · Proactive deployment metrics</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Shift filter
    sec("🕐 Shift Filter")
    SHIFT_MAP = {"All Shifts": None, "🌙 Night (10PM-6AM)": "NIGHT",
                 "🌅 Morning (6AM-10AM)": "MORNING", "☀️ Midday (10AM-2PM)": "MIDDAY",
                 "🌤️ Afternoon (2PM-6PM)": "AFTERNOON", "🌆 Evening (6PM-10PM)": "EVENING"}
    shift_ch = st.radio("Shift", list(SHIFT_MAP.keys()), horizontal=True, label_visibility="collapsed")
    bkt = SHIFT_MAP[shift_ch]
    shift_df = dff[dff["time_bucket"] == bkt] if bkt and "time_bucket" in dff.columns else dff

    sc1,sc2,sc3,sc4 = st.columns(4)
    kpi("Records in Shift", f"{len(shift_df):,}", sc1)
    kpi("Avg Severity", f"{shift_df['max_severity'].mean():.2f}" if "max_severity" in shift_df.columns and len(shift_df)>0 else "-", sc2)
    kpi("Heavy Vehicles", f"{int(shift_df['is_heavy_vehicle'].sum()):,}" if "is_heavy_vehicle" in shift_df.columns else "0", sc3)
    kpi("Unique Vehicles", f"{shift_df['vehicle_number'].nunique():,}" if "vehicle_number" in shift_df.columns else "-", sc4)

    if not time_block_df.empty:
        fig = px.bar(time_block_df, x="time_bucket", y="violation_count",
                     color="avg_severity", color_continuous_scale=["#4cc9f0","#e94560"],
                     hover_data=["ist_range","heavy_count","avg_congestion_weight"],
                     template=T["plotly_template"])
        fig.update_layout(paper_bgcolor=T["bg"], plot_bgcolor=T["card_bg2"], font_color=T["text"])
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Peak hour forecaster
    sec("🔮 Peak Hour Forecaster")
    col_sel, col_chart = st.columns([1, 3])
    with col_sel:
        forecast_stn = st.selectbox("Station:", ["City-Wide"] + sorted(dff["police_station"].unique().tolist()) if "police_station" in dff.columns else ["City-Wide"])

    if forecast_stn == "City-Wide":
        peak_data = global_peak_df
        ptitle = "City-Wide Peak Hours"
    else:
        peak_data = peak_time_df[peak_time_df["police_station"] == forecast_stn] if not peak_time_df.empty else pd.DataFrame()
        ptitle = f"Peak Hours - {forecast_stn}"

    with col_chart:
        if not peak_data.empty:
            fig = px.bar(peak_data, x="hour", y="violation_count",
                         color="violation_count", color_continuous_scale=["#13132a","#e94560"],
                         title=ptitle, template=T["plotly_template"])
            fig.update_layout(paper_bgcolor=T["bg"], plot_bgcolor=T["card_bg2"], font_color=T["text"])
            st.plotly_chart(fig, use_container_width=True)
            peak_h = int(peak_data.loc[peak_data["violation_count"].idxmax(), "hour"])
            st.success(f"🔮 **Deploy at {peak_h}:00 IST** - highest violation probability. Maintain until {(peak_h+2)%24}:00 IST.")

    st.divider()

    # Hour × Violation heatmap
    sec("🔥 Hour × Offence Heat Matrix")
    if not hour_viol_mat.empty:
        mx = hour_viol_mat.set_index(hour_viol_mat.columns[0])
        fig = px.imshow(
            mx.values,
            labels=dict(x="Violation Type", y="Hour (IST)", color="Count"),
            x=mx.columns.tolist(), y=[str(h) for h in mx.index.tolist()],
            color_continuous_scale=["#0a0a14","#4cc9f0","#e94560","#9b2335"],
            template=T["plotly_template"], aspect="auto",
        )
        fig.update_layout(paper_bgcolor=T["bg"], font_color=T["text"], height=520)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    c_dow, c_react = st.columns(2)
    with c_dow:
        sec("📅 Day-of-Week Trend")
        if not dow_df.empty:
            fig = px.bar(dow_df, x="day_name", y="violation_count",
                         color="avg_severity", color_continuous_scale=["#4cc9f0","#e94560"],
                         template=T["plotly_template"])
            fig.update_layout(paper_bgcolor=T["bg"], plot_bgcolor=T["card_bg2"], font_color=T["text"])
            st.plotly_chart(fig, use_container_width=True)

    with c_react:
        sec("⚖️ Reactive vs Proactive")
        if reactive_data:
            rc1,rc2 = st.columns(2)
            kpi("Night Patrol", f"{reactive_data.get('reactive_pct',0):.1f}%", rc1)
            kpi("Rush Hour Enforce.", f"{reactive_data.get('proactive_pct',0):.1f}%", rc2)
            fig = go.Figure(go.Pie(
                labels=["Night (Reactive)","Rush Hour (Proactive)","Other"],
                values=[reactive_data.get("reactive_night_patrol",0),
                        reactive_data.get("proactive_rush_hour",0),
                        reactive_data.get("other_hours",0)],
                marker_colors=["#9b2335","#52b788","#4cc9f0"],
                hole=.55, textinfo="label+percent", textfont_size=10,
            ))
            fig.update_layout(paper_bgcolor=T["bg"], font_color=T["text"], height=320,
                              margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, use_container_width=True)
            st.info(f"💡 {reactive_data.get('insight','')}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ██████████████████  PAGE: AI MODEL  ████████████████████████████████████████
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "🤖 AI Model":
    st.markdown('<div class="page-title">🤖 AI Model Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">LightGBM + XGBoost ensemble · Feature importance · Live prediction · 94.8% accuracy</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    mc1,mc2,mc3,mc4,mc5,mc6 = st.columns(6)
    kpi("Accuracy",        f"{model_summary.get('ensemble_accuracy',0):.2%}",  mc1)
    kpi("F1 (Weighted)",   f"{model_summary.get('ensemble_f1_weighted',0):.2%}", mc2)
    kpi("F1 (Macro)",      f"{model_summary.get('ensemble_f1_macro',0):.2%}",  mc3)
    kpi("Precision",       f"{model_summary.get('ensemble_precision',0):.2%}", mc4)
    kpi("Recall",          f"{model_summary.get('ensemble_recall',0):.2%}",    mc5)
    kpi("Train Rows",      f"{model_summary.get('train_rows',0):,}",           mc6)

    st.markdown("<br>", unsafe_allow_html=True)

    ml_tabs = st.tabs(["📊 Feature Importance","🔲 Confusion Matrix","📈 Model Comparison","🎮 Live Prediction"])

    with ml_tabs[0]:
        embed_plot(PLOT_DIR / "08_lgbm_feature_importance.png", "LightGBM Feature Importance (Top 20, by Gain)")

    with ml_tabs[1]:
        embed_plot(PLOT_DIR / "07_confusion_matrix.png", "Ensemble Confusion Matrix")

    with ml_tabs[2]:
        embed_plot(PLOT_DIR / "10_model_comparison.png", "LightGBM vs XGBoost vs Ensemble")

    with ml_tabs[3]:
        sec("🎮 Live Prediction - Violation Group Classifier")
        st.markdown("Enter a scenario to get an instant AI classification:")

        pred_stns = live_pred_cfg.get("police_stations", ["Upparpet"])
        pred_stn  = st.selectbox("Police Station", pred_stns, key="live_pred_station")
        station_lookup_preview = live_pred_cfg.get("station_lookup", {}).get(pred_stn, {})
        real_junctions = [j["junction_name"] for j in station_lookup_preview.get("junctions", [])]

        with st.form("live_pred"):
            pc1,pc2,pc3 = st.columns(3)
            with pc1:
                pred_hour = st.slider("Hour (IST)", 0, 23, 9)
                pred_day  = st.selectbox("Day", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
            with pc2:
                veh_cats  = live_pred_cfg.get("vehicle_categories", ["CAR","TWO_WHEELER","HEAVY"])
                pred_veh  = st.selectbox("Vehicle Category", veh_cats)
                pred_junc = st.radio("At Junction?", ["Yes","No"])
            with pc3:
                offender_tiers = live_pred_cfg.get("offender_tiers", ["FIRST_TIME"])
                pred_tier = st.selectbox(
                    "Offender History", offender_tiers,
                    help="Vehicle's prior-violation tier - pulled from real repeat-offender stats."
                )
                pred_junction_name = (
                    st.selectbox(f"Junction at {pred_stn}", real_junctions)
                    if real_junctions else None
                )

            go_btn = st.form_submit_button("🔮 Classify Now", type="primary", use_container_width=True)

        if go_btn:
            try:
                import lightgbm as lgb
                import xgboost as xgb_lib

                lgb_model = lgb.Booster(model_file=str(MODEL_DIR / "lgbm_model.txt"))
                xgb_model = xgb_lib.XGBClassifier()
                xgb_model.load_model(str(MODEL_DIR / "xgb_model.json"))

                day_map = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,"Friday":4,"Saturday":5,"Sunday":6}
                dn = day_map.get(pred_day, 0)
                if pred_hour >= 22 or pred_hour < 6: tb = "NIGHT"
                elif pred_hour < 10: tb = "MORNING"
                elif pred_hour < 14: tb = "MIDDAY"
                elif pred_hour < 18: tb = "AFTERNOON"
                else: tb = "EVENING"

                # ── Pull every value from REAL data-derived lookups built by
                # dashboard_data_pipeline.py - no hardcoded placeholder constants.
                fb        = live_pred_cfg.get("global_fallback", {})
                stn_data  = live_pred_cfg.get("station_lookup", {}).get(pred_stn, {})
                veh_data  = live_pred_cfg.get("vehicle_lookup", {}).get(pred_veh, {})
                tb_data   = live_pred_cfg.get("time_bucket_lookup", {}).get(tb, {})
                tier_data = live_pred_cfg.get("offender_tier_lookup", {}).get(pred_tier, {})
                veh_time_lookup = live_pred_cfg.get("veh_time_lookup", {})

                is_junction_flag = 1 if pred_junc == "Yes" else 0
                if is_junction_flag and pred_junction_name:
                    junc_match = next(
                        (j for j in stn_data.get("junctions", []) if j["junction_name"] == pred_junction_name),
                        None,
                    )
                    junction_name_enc = junc_match["junction_name_enc"] if junc_match else stn_data.get("no_junction_enc", 0)
                    is_top_junction   = junc_match["is_top_junction"] if junc_match else 0
                else:
                    junction_name_enc = stn_data.get("no_junction_enc", 0)
                    is_top_junction   = 0

                feat_dict = {
                    "hour_sin": np.sin(2*np.pi*pred_hour/24), "hour_cos": np.cos(2*np.pi*pred_hour/24),
                    "day_sin": np.sin(2*np.pi*dn/7), "day_cos": np.cos(2*np.pi*dn/7),
                    "is_night_shift": 1 if (pred_hour>=22 or pred_hour<6) else 0,
                    "is_weekend": 1 if dn>=5 else 0,
                    "geohash6_density": stn_data.get("geohash6_density_avg", fb.get("geohash6_density", 0.0)),
                    "police_station_density": stn_data.get("police_station_density", fb.get("police_station_density", 0.0)),
                    "vehicle_congestion_weight": veh_data.get("vehicle_congestion_weight", 2),
                    "is_heavy_vehicle": veh_data.get("is_heavy_vehicle", 0),
                    "is_junction": is_junction_flag, "is_top_junction": is_top_junction,
                    "repeat_offender_score": tier_data.get("repeat_offender_score_typical", 1),
                    "is_repeat_offender": tier_data.get("is_repeat_offender", 0),
                    "is_habitual_offender": tier_data.get("is_habitual_offender", 0),
                    "station_hour": stn_data.get("station_hour_by_hour", {}).get(str(pred_hour), fb.get("station_hour", 0.0)),
                    "veh_time": veh_time_lookup.get(f"{pred_veh}_{tb}", fb.get("veh_time", 0.0)),
                    "vehicle_category_enc": veh_data.get("vehicle_category_enc", 0),
                    "police_station_enc": stn_data.get("police_station_enc", 0),
                    "junction_name_enc": junction_name_enc,
                    "repeat_offender_tier_enc": tier_data.get("repeat_offender_tier_enc", 0),
                    "time_bucket_enc": tb_data.get("time_bucket_enc", 0),
                }
                # Feature order MUST exactly match what the model was trained on -
                # persisted by parking_intelligence_pipeline.py into model_summary.json
                # and copied through into this config by dashboard_data_pipeline.py.
                feats = live_pred_cfg.get("feature_columns", [])
                X = np.array([[feat_dict.get(f,0) for f in feats]], dtype="float32")
                lgb_p = lgb_model.predict(X)
                xgb_p = xgb_model.predict_proba(X)
                ens_p = (lgb_p + xgb_p) / 2
                classes = live_pred_cfg.get("target_classes", ["GENERIC_PARKING","SEVERE_OBSTRUCTION","VEHICLE_COMPLIANCE"])

                # ── Threshold-calibrated decision rule (KEY FIX for majority-
                # class bias). Plain argmax(probability) almost always picks
                # GENERIC_PARKING since it's ~92% of the training data, even
                # when a rarer class's probability is genuinely elevated for
                # this specific scenario. Instead, score each class by how far
                # its probability clears its OWN F1-calibrated threshold
                # (computed on held-out data in parking_intelligence_pipeline.py)
                # and pick the largest margin - not the largest raw number.
                thresholds = live_pred_cfg.get("decision_thresholds", {})
                margin_scores = []
                for ci, cname in enumerate(classes):
                    th = max(thresholds.get(cname, 1.0 / len(classes)), 1e-6)
                    margin_scores.append(float(ens_p[0][ci]) / th)
                idx = int(np.argmax(margin_scores))
                pred_cls = classes[idx] if idx < len(classes) else "UNKNOWN"
                conf = float(ens_p[0][idx]) * 100  # confidence shown is still the RAW probability - honest, not inflated by the margin score
                col_map = {"GENERIC_PARKING":"#f4a261","SEVERE_OBSTRUCTION":"#e94560","VEHICLE_COMPLIANCE":"#4cc9f0"}
                clr = col_map.get(pred_cls, T["accent2"])

                st.markdown(f"""
                <div style="background:{T['card_bg2']};border:2px solid {clr};border-radius:14px;
                            padding:28px;text-align:center;margin:14px 0;">
                  <div style="font-size:.78rem;color:{T['neutral']};text-transform:uppercase;letter-spacing:1.5px;">AI Classification Result</div>
                  <div style="font-family:'Space Grotesk',sans-serif;font-size:2.2rem;font-weight:800;
                              color:{clr};margin:10px 0;">{pred_cls.replace("_"," ")}</div>
                  <div style="font-size:1.1rem;color:{T['text']};">Confidence: <strong>{conf:.1f}%</strong></div>
                  <div style="font-size:.78rem;color:{T['neutral']};margin-top:8px;">
                    {pred_veh} · {pred_stn} · {pred_day} {pred_hour}:00 IST · {"Junction: " + (pred_junction_name or "Yes") if pred_junc=="Yes" else "Mid-road"} · {pred_tier.replace("_"," ").title()}
                  </div>
                </div>
                """, unsafe_allow_html=True)

                prob_df = pd.DataFrame({"Class": classes, "Probability (%)": [float(p)*100 for p in ens_p[0]]})
                fig = px.bar(prob_df, x="Class", y="Probability (%)",
                             color="Probability (%)", color_continuous_scale=["#4cc9f0","#e94560"],
                             template=T["plotly_template"])
                fig.update_layout(paper_bgcolor=T["bg"], plot_bgcolor=T["card_bg2"], font_color=T["text"])
                st.plotly_chart(fig, use_container_width=True)

                th_str = " · ".join(
                    f"{c.replace('_',' ').title()} ≥ {thresholds.get(c, 1.0/len(classes)):.2f}"
                    for c in classes
                )
                st.caption(
                    f"Decision uses calibrated per-class thresholds (not raw argmax) to counter "
                    f"the ~92% GENERIC_PARKING class imbalance: {th_str}. "
                    f"Winner = class with the largest probability÷threshold margin."
                )
            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.info("Ensure LightGBM and XGBoost are installed, models exist in outputs/model/, "
                         "and you've re-run parking_intelligence_pipeline.py + dashboard_data_pipeline.py "
                         "after the feature-order fix (model_summary.json must contain 'feature_columns').")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ██████████████████  PAGE: SYSTEM  ██████████████████████████████████████████
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "⚙️ System":
    st.markdown('<div class="page-title">⚙️ System & Data Health</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Data quality · Anomaly monitoring · Pipeline architecture · EDA gallery</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Validation status
    sec("✅ Data Validation Status")
    if validation_data:
        vc1,vc2,vc3 = st.columns(3)
        kpi("Total Records",    f"{validation_data.get('total',0):,}",        vc1)
        kpi("Approved Records", f"{validation_data.get('approved_count',0):,}",vc2)
        kpi("Approval Rate",    f"{validation_data.get('approved_pct',0):.1f}%",vc3)
        counts = validation_data.get("counts", {})
        if counts:
            fig = px.pie(
                names=list(counts.keys()), values=list(counts.values()),
                color_discrete_sequence=["#52b788","#f4a261","#e94560","#4cc9f0","#9b59b6","#6c757d"],
                template=T["plotly_template"], hole=.4,
            )
            fig.update_layout(paper_bgcolor=T["bg"], font_color=T["text"], height=380)
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Anomaly
    sec("📉 Data Anomaly Monitor")
    if anomaly_data.get("has_anomalies"):
        st.error(anomaly_data.get("warning_message",""))
        monthly = anomaly_data.get("monthly_counts",[])
        if monthly:
            mdf = pd.DataFrame(monthly)
            fig = px.bar(mdf, x="_month", y="record_count",
                         color="record_count", color_continuous_scale=["#e94560","#4cc9f0","#52b788"],
                         template=T["plotly_template"])
            fig.update_layout(paper_bgcolor=T["bg"], plot_bgcolor=T["card_bg2"], font_color=T["text"])
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("✅ No anomalies detected in data collection volume.")

    st.divider()

    # Pipeline architecture
    sec("🏗️ Pipeline Architecture")
    st.markdown("""
| Step | Script | Purpose |
|------|--------|---------|
| 1 | `data_cleaning.py` | Device-trust NaN recovery → 93K+ records |
| 2 | `feature_engineering.py` | 28 ML features (geohash, cyclical time, etc.) |
| 3 | `violation_features.py` | Target label engineering (3 enforcement groups) |
| 4 | `parking_intelligence_pipeline.py` | ML training + Folium maps + EDA plots |
| 5 | `dashboard_data_pipeline.py` | Pre-compute 22 dashboard data artifacts |
| 6 | `streamlit run app.py` | **The LogPose** interactive dashboard |
    """)

    st.divider()

    # EDA gallery
    sec("📊 EDA Visualization Gallery")
    g1,g2 = st.columns(2)
    with g1:
        embed_plot(PLOT_DIR / "01_top_violation_classes.png", "Top 15 Violation Classes")
        embed_plot(PLOT_DIR / "03_day_of_week.png", "Day of Week Distribution")
        embed_plot(PLOT_DIR / "05_severity_distribution.png", "Severity Distribution")
        embed_plot(PLOT_DIR / "09_dbscan_cluster_sizes.png", "DBSCAN Cluster Sizes")
    with g2:
        embed_plot(PLOT_DIR / "02_hourly_pattern.png", "Hourly Pattern with Rush Hour Bands")
        embed_plot(PLOT_DIR / "04_vehicle_category.png", "Vehicle Category Distribution")
        embed_plot(PLOT_DIR / "06_top_police_stations.png", "Top 10 Police Stations")

    st.divider()

    # Full export
    sec("📥 System Export")
    ex1,ex2,ex3 = st.columns(3)
    with ex1:
        st.download_button("📥 Full Dataset CSV", dff.to_csv(index=False).encode(),
                           "logpose_full_dataset.csv","text/csv",use_container_width=True)
    with ex2:
        if not vehicle_idx.empty:
            st.download_button("📥 Vehicle Index CSV", vehicle_idx.to_csv(index=False).encode(),
                               "logpose_vehicle_index.csv","text/csv",use_container_width=True)
    with ex3:
        if not habitual_df.empty:
            st.download_button("📥 Habitual Offenders CSV", habitual_df.to_csv(index=False).encode(),
                               "logpose_habitual.csv","text/csv",use_container_width=True)