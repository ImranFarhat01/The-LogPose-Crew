import re
with open(r"d:\Flipkart GridLock 2.0\Round-2\app.py", "r", encoding="utf-8") as f:
    content = f.read()

new_css = '''st.markdown(f"""
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

/* ── Nav Items with Hover Animations ── */
.nav-container {{ display: flex; flex-direction: column; gap: 6px; padding: 0; margin-top: 10px; }}
.nav-item {{
    display: flex; align-items: center; gap: 12px;
    padding: 12px 16px; border-radius: 12px;
    font-size: .9rem; font-weight: 500;
    color: {T['neutral']};
    cursor: pointer; transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    text-decoration: none;
    border: 1px solid transparent;
    background: transparent;
    position: relative;
    overflow: hidden;
}}
.nav-item::before {{
    content: ''; position: absolute; top: 0; left: -100%; width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, {'rgba(255,255,255,0.1)' if _dark else 'rgba(0,0,0,0.05)'}, transparent);
    transition: left 0.5s ease;
}}
.nav-item:hover::before {{ left: 100%; }}
.nav-item:hover {{
    background: {'rgba(255,255,255,0.08)' if _dark else 'rgba(0,0,0,0.04)'};
    color: {T['text']};
    transform: translateX(4px);
    border-color: {'rgba(255,255,255,0.1)' if _dark else 'rgba(0,0,0,0.08)'};
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}}
.nav-item-active {{
    background: {'rgba(233,69,96,0.15)' if _dark else 'rgba(233,69,96,0.1)'} !important;
    border: 1px solid {'rgba(233,69,96,0.4)' if _dark else 'rgba(233,69,96,0.3)'} !important;
    color: {T['accent']} !important;
    font-weight: 700;
    box-shadow: 0 4px 20px rgba(233,69,96,0.2) !important;
}}
.nav-icon {{ font-size: 1.2rem; width: 26px; text-align: center; }}
.nav-label {{ flex: 1; }}
.nav-active-dot {{
    width: 8px; height: 8px; border-radius: 50%;
    background: {T['accent']};
    box-shadow: 0 0 12px {T['accent']}, 0 0 4px {T['accent']};
    animation: pulseDot 2s infinite;
}}

@keyframes pulseDot {{
    0% {{ box-shadow: 0 0 0 0 rgba(233,69,96,0.7); }}
    70% {{ box-shadow: 0 0 0 6px rgba(233,69,96,0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(233,69,96,0); }}
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

</style>
"""'''

new_content = re.sub(
    r'st\.markdown\(f"""\n<style>\n@import url.*?unsafe_allow_html=True\)',
    new_css + ", unsafe_allow_html=True)",
    content,
    flags=re.DOTALL
)

with open(r"d:\Flipkart GridLock 2.0\Round-2\app.py", "w", encoding="utf-8") as f:
    f.write(new_content)
