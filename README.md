# 🚦 PS1 — Parking-Induced Congestion Intelligence System

**Flipkart GridLock Hackathon 2.0 — Round 2**

> AI-driven parking violation hotspot detection and enforcement optimization for Bengaluru Traffic Police (BTP)

---

## 🎯 Problem Statement

On-street illegal parking and spillover parking near commercial areas, metro stations, and events choke carriageways and intersections in Bengaluru. Enforcement is patrol-based and reactive — no heatmap of parking violations vs congestion impact exists, making it difficult to prioritize enforcement zones.

**Our Solution:** An end-to-end AI pipeline that detects illegal parking hotspots, quantifies their congestion impact, and generates targeted enforcement recommendations — all powered by 298,450 real BTP violation records.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA PIPELINE (offline)                      │
│                                                                 │
│  data_cleaning.py → feature_engineering.py → violation_features.py │
│         │                    │                       │          │
│         ▼                    ▼                       ▼          │
│    93K+ rows            28 ML features         8-class target   │
│  (device-trust          (sin/cos time,          (merged to 3    │
│   NaN recovery)          geohash6,              enforcement     │
│                          interactions)           groups)         │
├─────────────────────────────────────────────────────────────────┤
│                    ML PIPELINE (offline)                        │
│                                                                 │
│  parking_intelligence_pipeline.py                               │
│         │                                                       │
│         ├── DBSCAN hotspot clustering (haversine, 300m radius)  │
│         ├── LightGBM + XGBoost soft-voting ensemble             │
│         ├── Congestion impact scoring (1-100)                   │
│         ├── Enforcement priority ranking (54 stations)          │
│         ├── 3 Folium interactive maps                           │
│         └── 10 Matplotlib/Seaborn EDA plots                     │
├─────────────────────────────────────────────────────────────────┤
│                    DASHBOARD PIPELINE (offline)                 │
│                                                                 │
│  dashboard_data_pipeline.py                                     │
│         │                                                       │
│         └── 22 pre-computed data artifacts (JSON/CSV)           │
│             for instant dashboard loading                       │
├─────────────────────────────────────────────────────────────────┤
│                    STREAMLIT DASHBOARD (live)                   │
│                                                                 │
│  streamlit run app.py                                           │
│         │                                                       │
│         └── 35 interactive UI features                          │
│             Maps, Priority Ranker, Vehicle Lookup,              │
│             Peak Time Forecaster, Live ML Prediction,           │
│             Patrol Gap Analysis, etc.                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| **Ensemble Accuracy** | 94.8% |
| **Ensemble F1 (weighted)** | 94.8% |
| **DBSCAN Hotspot Clusters** | 164 |
| **Training Records** | 113,127 |
| **Test Records** | 28,282 |
| **Features Used** | 22 |
| **Target Classes** | 3 enforcement groups |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip

### Install Dependencies
```bash
pip install pandas numpy xgboost lightgbm scikit-learn folium streamlit plotly pygeohash tqdm seaborn matplotlib
```

### Run Pipeline (in order)
```bash
# Step 1: Clean raw data (device-trust NaN recovery)
python data_cleaning.py

# Step 2: Engineer 28 ML features
python feature_engineering.py

# Step 3: Build violation target labels
python violation_features.py

# Step 4: Train ML models + generate maps & plots
python parking_intelligence_pipeline.py

# Step 5: Pre-compute dashboard data artifacts
python dashboard_data_pipeline.py

# Step 6: Launch interactive dashboard
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`.

---

## 🖥️ Dashboard Features (35 total)

### 🗺️ Maps & Spatial (9 features)
1. Interactive Folium heatmap of violation density
2. Junction vs Mid-Road toggle filter
3. Police Station jurisdiction dropdown
4. Custom hotspot radius via DBSCAN clusters
5. Clickable hotspot pins with violation breakdowns
6. Patrol Gap overlay (device activity vs violation density)
7. IT Corridor quick-view preset
8. Commercial Belt quick-view preset
9. Outskirts flagging markers

### 📊 Priority & Insights (6 features)
10. Enforcement Priority Ranker (Top 10 zones)
11. Congestion Impact Score (1-100 per zone)
12. Top 15 Hotspots Leaderboard
13. Offence Severity color-coded map
14. Plain-English enforcement recommendations
15. SCITA Smart City sync status indicator

### 🚗 Vehicle & Offenders (5 features)
16. Habitual Offender Alert Board
17. Vehicle Lookup Search Bar
18. Vehicle Type Congestion filter (Heavy vs 2-Wheeler)
19. Multi-Violation Profile viewer
20. Vehicle vs Violation interactive matrix

### ⏱️ Time & Predictions (5 features)
21. Predictive Peak Time Forecaster per station
22. Time-Block Shift filter (Night/Morning/Midday/Afternoon/Evening)
23. Hour × Violation heatmap matrix
24. Day-of-Week trend graph
25. Reactive vs Proactive enforcement gauge

### 🤖 ML Explainability (5 features)
31. Feature Importance chart
32. Model evaluation metrics dashboard
33. Confusion Matrix explorer
34. Live Prediction demo (select time/vehicle/station → get prediction)
35. Geohash6 Grid overlay

### ⚙️ System Controls (5 features)
26. Data Quality toggle (Approved vs All records)
27. Violation type filter checkboxes
28. CSV/PDF data export buttons
29. Anomaly warning banner (data collection gaps)
30. Dark/Light theme toggle

---

## 📁 Project Structure

```
Round-2/
├── data/
│   ├── jan to may police violation_anonymized791b166.csv   # Raw BTP data (298K rows)
│   ├── cleaned/
│   │   ├── dataset_cleaned.csv          # After device-trust recovery (~93K rows)
│   │   ├── dataset_features.csv         # Full feature matrix (81 columns)
│   │   └── cleaning_audit.json
│   ├── class_weights.json
│   └── label_encoders.pkl
│
├── outputs/
│   ├── maps/
│   │   ├── 01_congestion_heatmap.html
│   │   ├── 02_hotspot_clusters_priority.html
│   │   └── 03_night_vs_day.html
│   ├── plots/                           # 10 EDA visualizations
│   ├── model/
│   │   ├── lgbm_model.txt
│   │   └── xgb_model.json
│   ├── dashboard_data/                  # 22 pre-computed artifacts
│   ├── enforcement_priority_ranked.csv
│   ├── hotspot_clusters.csv
│   ├── classification_report.txt
│   └── model_summary.json
│
├── data_cleaning.py                     # Step 1: Device-trust NaN recovery
├── feature_engineering.py               # Step 2: 28 ML features
├── violation_features.py                # Step 3: Target label engineering
├── parking_intelligence_pipeline.py     # Step 4: ML + maps + plots
├── dashboard_data_pipeline.py           # Step 5: Dashboard artifacts
├── app.py                               # Step 6: Streamlit dashboard
└── README.md
```

---

## 🔬 Technical Highlights

### No Data Leakage
- Device-trust scores computed ONLY from labeled (approved/rejected) records
- NaN records are passive recipients — they never influence trust scores
- Leakage columns (viol_* binary flags) explicitly blocked from ML training
- Class weights fitted on training fold only
- Geohash density replaces raw lat/lon (structural feature, not target-derived)

### Device-Trust NaN Recovery
The raw dataset has only ~8K approved records. Our device-trust scoring promotes high-quality NaN records (from devices with ≥80% approval rates) to the training pool, growing it to ~93K rows.

### Geohash6 Spatial Encoding (Round 1 winning feature)
Instead of raw lat/lon (overfitting risk), we encode locations into geohash6 cells (~1km²) and use per-cell violation density as a normalized feature.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Data Processing | pandas, numpy |
| ML Models | LightGBM + XGBoost (soft-voting ensemble) |
| Spatial Analysis | DBSCAN (haversine), pygeohash |
| Maps | Folium (heatmaps, cluster markers) |
| Visualization | Plotly, Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Language | Python 3.10+ |

---

## 📄 License

This project was created for the Flipkart GridLock Hackathon 2.0 — Round 2.
Dataset provided by Bengaluru Traffic Police (anonymized).
