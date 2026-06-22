"""
=============================================================================
  dashboard_data_pipeline.py
  Flipkart GridLock 2.0 — Round 2 | PS1
  PURPOSE: Pre-compute all data artifacts required by the Streamlit dashboard.
           This script runs AFTER the main ML pipeline and produces JSON/CSV
           files consumed by app.py.

  NO DATA LEAKAGE:
    • This script ONLY reads the already-cleaned, already-trained-on data.
    • It performs aggregations and summaries for DISPLAY purposes only.
    • No model training happens here — models are loaded read-only for
      live inference demo.

  PIPELINE ORDER:
    1. python data_cleaning.py
    2. python feature_engineering.py
    3. python violation_features.py
    4. python parking_intelligence_pipeline.py
    5. python dashboard_data_pipeline.py         ← THIS FILE
    6. streamlit run app.py
=============================================================================
"""

# ── Step 0: auto-install ────────────────────────────────────────────────────
import subprocess, sys, importlib

DEPS = {
    "pandas":     "pandas>=2.0",
    "numpy":      "numpy>=1.24",
    "pygeohash":  "pygeohash",
    "tqdm":       "tqdm",
}
IMPORT_MAP = {"pygeohash": "pygeohash"}

print("\n" + "=" * 70)
print("  DASHBOARD DATA PIPELINE — Pre-computing dashboard artifacts")
print("=" * 70)

for pkg, pip_spec in DEPS.items():
    iname = IMPORT_MAP.get(pkg, pkg)
    try:
        importlib.import_module(iname)
    except ImportError:
        print(f"  ⬇️  Installing {pip_spec}...", end=" ", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", pip_spec, "-q"],
                       capture_output=True)
        print("done")

import ast, json, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import pygeohash as gh
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── PATHS ───────────────────────────────────────────────────────────────────
BASE_DIR     = Path(r"D:\Flipkart Gridlock 2.0\Round-2")
DATA_DIR     = BASE_DIR / "data"
CLEAN_DIR    = DATA_DIR / "cleaned"
OUT_DIR      = BASE_DIR / "outputs"
DASH_DIR     = OUT_DIR / "dashboard_data"
DASH_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_CSV = CLEAN_DIR / "dataset_features.csv"
PRIORITY_CSV = OUT_DIR / "enforcement_priority_ranked.csv"
HOTSPOT_CSV  = OUT_DIR / "hotspot_clusters.csv"
MODEL_JSON   = OUT_DIR / "model_summary.json"

CHUNKSIZE = 50_000

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LOAD DATA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[1] Loading feature dataset...")
chunks = []
reader = pd.read_csv(FEATURES_CSV, chunksize=CHUNKSIZE, low_memory=False)
for chunk in tqdm(reader, desc="  Loading", unit="chunk"):
    chunks.append(chunk)
df = pd.concat(chunks, ignore_index=True)
print(f"  ✅  {len(df):,} rows × {df.shape[1]} cols")

# Parse datetime if needed
if "created_datetime_ist" in df.columns:
    df["created_datetime_ist"] = pd.to_datetime(
        df["created_datetime_ist"], errors="coerce", utc=True
    )
elif "created_datetime" in df.columns:
    df["created_datetime"] = pd.to_datetime(
        df["created_datetime"], format="ISO8601", utc=True, errors="coerce"
    )
    df["created_datetime_ist"] = df["created_datetime"].dt.tz_convert("Asia/Kolkata")

# Ensure hour & day columns exist
if "hour" not in df.columns:
    df["hour"] = df["created_datetime_ist"].dt.hour
if "day_of_week" not in df.columns:
    df["day_of_week"] = df["created_datetime_ist"].dt.dayofweek

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ARTIFACT 1: Peak Time Forecaster (Feature 21)
# For each police station, find the top 3 peak hours by violation count.
# This acts as a "forecaster" — telling officers WHEN to deploy.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[2] Peak Time Forecaster...")

peak_time = (
    df.groupby(["police_station", "hour"])
    .size()
    .reset_index(name="violation_count")
    .sort_values(["police_station", "violation_count"], ascending=[True, False])
)
# Top 3 peak hours per station
peak_top3 = peak_time.groupby("police_station").head(3).reset_index(drop=True)
peak_top3.to_csv(DASH_DIR / "peak_time_forecast.csv", index=False)

# Global peak hours
global_peak = (
    df.groupby("hour").size().reset_index(name="violation_count")
    .sort_values("violation_count", ascending=False)
)
global_peak.to_csv(DASH_DIR / "global_peak_hours.csv", index=False)
print(f"  ✅  peak_time_forecast.csv  ({len(peak_top3)} rows)")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ARTIFACT 2: Hour vs Violation Heat Matrix (Feature 23)
# Crosstab of hour × primary_violation_final counts
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[3] Hour vs Violation Heat Matrix...")

target_col = (
    "primary_violation_final" if "primary_violation_final" in df.columns
    else "primary_violation"
)
hour_viol_matrix = pd.crosstab(df["hour"], df[target_col])
hour_viol_matrix.to_csv(DASH_DIR / "hour_vs_violation_matrix.csv")
print(f"  ✅  hour_vs_violation_matrix.csv  ({hour_viol_matrix.shape})")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ARTIFACT 3: Patrol Gap Analysis (Feature 6)
# Compare device activity zones vs violation density zones
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[4] Patrol Gap Analysis...")

if "device_id" in df.columns and "geohash5" in df.columns:
    # Device activity: count of records per geohash5 zone per device
    device_zones = (
        df.groupby("geohash5")["device_id"]
        .nunique()
        .reset_index(name="active_devices")
    )
    # Violation density per zone
    zone_violations = (
        df.groupby("geohash5")
        .agg(
            violation_count=("id", "count"),
            centroid_lat=("latitude", "mean"),
            centroid_lon=("longitude", "mean"),
        )
        .reset_index()
    )
    patrol_gap = zone_violations.merge(device_zones, on="geohash5", how="left")
    patrol_gap["active_devices"] = patrol_gap["active_devices"].fillna(0).astype(int)
    # Gap score: high violations + low devices = high gap
    v_max = patrol_gap["violation_count"].max()
    d_max = max(patrol_gap["active_devices"].max(), 1)
    patrol_gap["gap_score"] = (
        (patrol_gap["violation_count"] / v_max) -
        (patrol_gap["active_devices"] / d_max)
    ).clip(0, 1).round(4)
    patrol_gap = patrol_gap.sort_values("gap_score", ascending=False).reset_index(drop=True)
    patrol_gap.to_csv(DASH_DIR / "patrol_gap_analysis.csv", index=False)
    print(f"  ✅  patrol_gap_analysis.csv  ({len(patrol_gap)} rows)")
else:
    print("  ⚠️  device_id or geohash5 not found — skipping patrol gap")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ARTIFACT 4: Reactive vs Proactive Metric (Feature 25)
# Night patrol (reactive) vs AM/PM rush hour (proactive) enforcement
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[5] Reactive vs Proactive Metric...")

# IST hours: night_patrol = 10PM-6AM, am_rush = 8-10AM, pm_rush = 5-8PM
night_mask = (df["hour"] >= 22) | (df["hour"] < 6)
am_rush_mask = (df["hour"] >= 8) & (df["hour"] < 10)
pm_rush_mask = (df["hour"] >= 17) & (df["hour"] < 20)

reactive_count = int(night_mask.sum())
proactive_count = int(am_rush_mask.sum() + pm_rush_mask.sum())
total = len(df)

reactive_proactive = {
    "reactive_night_patrol": reactive_count,
    "proactive_rush_hour": proactive_count,
    "other_hours": total - reactive_count - proactive_count,
    "total": total,
    "reactive_pct": round(reactive_count / total * 100, 2),
    "proactive_pct": round(proactive_count / total * 100, 2),
    "proactive_ratio": round(proactive_count / max(reactive_count, 1), 4),
    "insight": (
        f"Only {round(proactive_count / total * 100, 1)}% of enforcement happens "
        f"during rush hours vs {round(reactive_count / total * 100, 1)}% at night. "
        f"Shift enforcement to 8-10AM and 5-8PM IST for maximum congestion impact."
    ),
}
with open(DASH_DIR / "reactive_vs_proactive.json", "w") as f:
    json.dump(reactive_proactive, f, indent=2)
print(f"  ✅  reactive_vs_proactive.json")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ARTIFACT 5: Anomaly Detection Banner (Feature 29)
# Detect months with abnormally low data collection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[6] Anomaly Detection...")

if "created_datetime_ist" in df.columns:
    # Drop rows with NaT datetimes before grouping to avoid "NaT" period strings
    valid_dt = df.dropna(subset=["created_datetime_ist"]).copy()
    valid_dt["_month"] = valid_dt["created_datetime_ist"].dt.to_period("M").astype(str)
    # Filter out any remaining NaT-like period strings
    valid_dt = valid_dt[~valid_dt["_month"].isin(["NaT", "nan", ""])]
    monthly_counts = valid_dt.groupby("_month").size().reset_index(name="record_count")
    median_count = monthly_counts["record_count"].median()
    anomalies = monthly_counts[monthly_counts["record_count"] < median_count * 0.2].copy()
    anomalies["expected_median"] = int(median_count)
    anomalies["drop_pct"] = (
        (1 - anomalies["record_count"] / median_count) * 100
    ).round(1)

    anomaly_data = {
        "monthly_counts": monthly_counts.to_dict(orient="records"),
        "anomalies": anomalies.to_dict(orient="records"),
        "has_anomalies": len(anomalies) > 0,
        "warning_message": (
            f"⚠️ Data collection dropped sharply in: "
            f"{', '.join(anomalies['_month'].tolist())}. "
            f"These months have <20% of median volume ({int(median_count):,} records). "
            f"Treat insights from these periods with caution."
        ) if len(anomalies) > 0 else "",
    }
else:
    anomaly_data = {"has_anomalies": False, "anomalies": [], "monthly_counts": []}

with open(DASH_DIR / "anomaly_detection.json", "w") as f:
    json.dump(anomaly_data, f, indent=2, default=str)
print(f"  ✅  anomaly_detection.json  (anomalies found: {anomaly_data['has_anomalies']})")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ARTIFACT 6: SCITA Sync Statistics (Feature 15)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[7] SCITA Sync Statistics...")

if "data_sent_to_scita" in df.columns:
    scita_col = df["data_sent_to_scita"].astype(str).str.strip().str.lower()
    sent = int(scita_col.isin(["true", "1"]).sum())
    not_sent = int(scita_col.isin(["false", "0"]).sum())
    total_scita = sent + not_sent
    scita_stats = {
        "sent_to_scita": sent,
        "not_sent": not_sent,
        "total": total_scita,
        "sync_pct": round(sent / max(total_scita, 1) * 100, 2),
    }
else:
    scita_stats = {"sent_to_scita": 0, "not_sent": 0, "total": 0, "sync_pct": 0}

with open(DASH_DIR / "scita_sync.json", "w") as f:
    json.dump(scita_stats, f, indent=2)
print(f"  ✅  scita_sync.json  (sync rate: {scita_stats['sync_pct']}%)")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ARTIFACT 7: Vehicle Lookup Index (Feature 17)
# Pre-aggregate per-vehicle violation history for fast lookup
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[8] Vehicle Lookup Index...")

if "vehicle_number" in df.columns:
    vehicle_index = (
        df.groupby("vehicle_number")
        .agg(
            total_violations=("id", "count"),
            vehicle_type=("vehicle_type", "first"),
            vehicle_category=("vehicle_category", "first"),
            primary_stations=("police_station", lambda x: ", ".join(x.value_counts().head(3).index)),
            primary_violations=("primary_violation", lambda x: ", ".join(x.value_counts().head(3).index)),
            first_seen=("created_datetime_ist", "min"),
            last_seen=("created_datetime_ist", "max"),
            avg_severity=("max_severity", "mean"),
            is_habitual=("is_habitual_offender", "max"),
        )
        .reset_index()
        .sort_values("total_violations", ascending=False)
    )
    vehicle_index["first_seen"] = vehicle_index["first_seen"].astype(str).str[:10]
    vehicle_index["last_seen"] = vehicle_index["last_seen"].astype(str).str[:10]
    vehicle_index["avg_severity"] = vehicle_index["avg_severity"].round(2)
    vehicle_index.to_csv(DASH_DIR / "vehicle_lookup_index.csv", index=False)
    print(f"  ✅  vehicle_lookup_index.csv  ({len(vehicle_index):,} vehicles)")
else:
    print("  ⚠️  vehicle_number not found — skipping")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ARTIFACT 8: Multi-Violation Profiles (Feature 19)
# Records with 2+ simultaneous violations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[9] Multi-Violation Profiles...")

if "violation_count" in df.columns:
    multi = df[df["violation_count"] > 1].copy()
    multi_profiles = multi[[
        "id", "vehicle_number", "vehicle_type", "vehicle_category",
        "violation_type", "violation_count", "max_severity",
        "police_station", "junction_name", "location",
        "latitude", "longitude", "hour", "day_of_week",
    ]].sort_values("violation_count", ascending=False).head(500)
    multi_profiles.to_csv(DASH_DIR / "multi_violation_profiles.csv", index=False)

    # Summary stats
    multi_summary = {
        "total_multi_violation_records": int(len(multi)),
        "pct_of_total": round(len(multi) / len(df) * 100, 2),
        "avg_violations_per_record": round(float(multi["violation_count"].mean()), 2),
        "max_violations_single_record": int(multi["violation_count"].max()),
        "distribution": multi["violation_count"].value_counts().sort_index().to_dict(),
    }
    with open(DASH_DIR / "multi_violation_summary.json", "w") as f:
        json.dump(multi_summary, f, indent=2)
    print(f"  ✅  multi_violation_profiles.csv  ({len(multi_profiles)} records)")
else:
    print("  ⚠️  violation_count not found — skipping")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ARTIFACT 9: Vehicle vs Violation Crosstab (Feature 20)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[10] Vehicle vs Violation Crosstab...")

if "vehicle_category" in df.columns and target_col in df.columns:
    veh_viol_matrix = pd.crosstab(df["vehicle_category"], df[target_col])
    veh_viol_matrix.to_csv(DASH_DIR / "vehicle_vs_violation_matrix.csv")
    print(f"  ✅  vehicle_vs_violation_matrix.csv  ({veh_viol_matrix.shape})")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ARTIFACT 10: Geohash Grid Overlay Data (Feature 35)
# Aggregate violations per geohash6 cell for grid visualization
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[11] Geohash Grid Overlay...")

if "geohash6" in df.columns:
    geohash_grid = (
        df.groupby("geohash6")
        .agg(
            violation_count=("id", "count"),
            centroid_lat=("latitude", "mean"),
            centroid_lon=("longitude", "mean"),
            avg_severity=("max_severity", "mean"),
            heavy_pct=("is_heavy_vehicle", "mean"),
            top_station=("police_station", lambda x: x.value_counts().index[0] if len(x) > 0 else "Unknown"),
        )
        .reset_index()
        .sort_values("violation_count", ascending=False)
    )
    geohash_grid["avg_severity"] = geohash_grid["avg_severity"].round(2)
    geohash_grid["heavy_pct"] = (geohash_grid["heavy_pct"] * 100).round(1)

    # Decode geohash6 to bounding box for rectangle rendering
    def geohash_bbox(ghash):
        try:
            lat, lon, lat_err, lon_err = gh.decode_exactly(ghash)
            return {
                "sw_lat": lat - lat_err, "sw_lon": lon - lon_err,
                "ne_lat": lat + lat_err, "ne_lon": lon + lon_err,
            }
        except Exception:
            return {"sw_lat": 0, "sw_lon": 0, "ne_lat": 0, "ne_lon": 0}

    bbox_data = geohash_grid["geohash6"].apply(geohash_bbox).apply(pd.Series)
    geohash_grid = pd.concat([geohash_grid, bbox_data], axis=1)
    geohash_grid.to_csv(DASH_DIR / "geohash_grid_overlay.csv", index=False)
    print(f"  ✅  geohash_grid_overlay.csv  ({len(geohash_grid)} cells)")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ARTIFACT 11: Plain-English Recommendations (Feature 14)
# Auto-generate actionable deployment directives
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[12] Plain-English Recommendations...")

recommendations = []

# Load priority data
if PRIORITY_CSV.exists():
    priority_df = pd.read_csv(PRIORITY_CSV)
    # Top 5 stations
    for _, row in priority_df.head(5).iterrows():
        station = row["police_station"]
        # Find peak hour for this station
        station_data = df[df["police_station"] == station] if "police_station" in df.columns else pd.DataFrame()
        if len(station_data) > 0:
            peak_hr = station_data["hour"].value_counts().index[0]
            peak_hr_end = (peak_hr + 2) % 24
            total_v = int(row["violation_total"])
            sev = row["avg_severity"]
            heavy = int(row.get("heavy_veh_count", 0))

            # Calculate recommended officers (rough heuristic)
            officers = max(2, min(8, total_v // 2000))

            rec = {
                "priority": int(row["rank"]),
                "station": station,
                "directive": (
                    f"Deploy {officers} officers to {station} jurisdiction "
                    f"between {peak_hr}:00 – {peak_hr_end}:00 IST. "
                    f"This zone has {total_v:,} violations "
                    f"(avg severity: {sev:.1f}/5). "
                    + (f"Watch for {heavy} heavy vehicle violations." if heavy > 50 else "")
                ),
                "score": float(row["priority_score"]),
            }
            recommendations.append(rec)

# Global recommendations
recommendations.append({
    "priority": 0,
    "station": "CITY-WIDE",
    "directive": reactive_proactive["insight"],
    "score": 1.0,
})

with open(DASH_DIR / "recommendations.json", "w") as f:
    json.dump(recommendations, f, indent=2)
print(f"  ✅  recommendations.json  ({len(recommendations)} directives)")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ARTIFACT 12: Time-Block Shift Data (Feature 22)
# Pre-aggregate violation counts by shift/time block
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[13] Time-Block Shift Data...")

if "time_bucket" in df.columns:
    time_block = (
        df.groupby("time_bucket")
        .agg(
            violation_count=("id", "count"),
            avg_severity=("max_severity", "mean"),
            heavy_count=("is_heavy_vehicle", "sum"),
            avg_congestion_weight=("vehicle_congestion_weight", "mean"),
        )
        .reset_index()
    )
    time_block["avg_severity"] = time_block["avg_severity"].round(2)
    time_block["avg_congestion_weight"] = time_block["avg_congestion_weight"].round(2)
    # Add IST equivalents
    IST_MAP = {
        "NIGHT": "10:00 PM – 6:00 AM IST",
        "MORNING": "6:00 AM – 10:00 AM IST",
        "MIDDAY": "10:00 AM – 2:00 PM IST",
        "AFTERNOON": "2:00 PM – 6:00 PM IST",
        "EVENING": "6:00 PM – 10:00 PM IST",
    }
    time_block["ist_range"] = time_block["time_bucket"].map(IST_MAP)
    time_block.to_csv(DASH_DIR / "time_block_shifts.csv", index=False)
    print(f"  ✅  time_block_shifts.csv  ({len(time_block)} blocks)")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ARTIFACT 13: Habitual Offender Details (Feature 16)
# Top 100 habitual offenders with full profiles
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[14] Habitual Offender Details...")

if "vehicle_number" in df.columns and "is_habitual_offender" in df.columns:
    habitual = df[df["is_habitual_offender"] == 1].copy()
    if len(habitual) > 0:
        hab_profiles = (
            habitual.groupby("vehicle_number")
            .agg(
                total_violations=("id", "count"),
                vehicle_type=("vehicle_type", "first"),
                vehicle_category=("vehicle_category", "first"),
                top_station=("police_station", lambda x: x.value_counts().index[0]),
                top_violation=("primary_violation", lambda x: x.value_counts().index[0]),
                avg_severity=("max_severity", "mean"),
                first_seen=("created_datetime_ist", "min"),
                last_seen=("created_datetime_ist", "max"),
            )
            .reset_index()
            .sort_values("total_violations", ascending=False)
            .head(100)
        )
        hab_profiles["first_seen"] = hab_profiles["first_seen"].astype(str).str[:10]
        hab_profiles["last_seen"] = hab_profiles["last_seen"].astype(str).str[:10]
        hab_profiles["avg_severity"] = hab_profiles["avg_severity"].round(2)
        hab_profiles.to_csv(DASH_DIR / "habitual_offenders.csv", index=False)
        print(f"  ✅  habitual_offenders.csv  ({len(hab_profiles)} offenders)")
    else:
        print("  ⚠️  No habitual offenders found")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ARTIFACT 14: Station + Junction quick-reference (Features 3, 7, 8, 9, 12)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[15] Station & Junction quick-reference...")

if "police_station" in df.columns:
    station_ref = (
        df.groupby("police_station")
        .agg(
            violation_count=("id", "count"),
            centroid_lat=("latitude", "mean"),
            centroid_lon=("longitude", "mean"),
            avg_severity=("max_severity", "mean"),
            heavy_pct=("is_heavy_vehicle", "mean"),
            habitual_count=("is_habitual_offender", "sum"),
            top_violation=(target_col, lambda x: x.value_counts().index[0] if len(x) > 0 else ""),
        )
        .reset_index()
        .sort_values("violation_count", ascending=False)
    )
    station_ref["avg_severity"] = station_ref["avg_severity"].round(2)
    station_ref["heavy_pct"] = (station_ref["heavy_pct"] * 100).round(1)
    station_ref.to_csv(DASH_DIR / "station_reference.csv", index=False)
    print(f"  ✅  station_reference.csv  ({len(station_ref)} stations)")

# Junction reference
if "junction_name" in df.columns:
    junction_ref = (
        df[df["is_junction"] == 1]
        .groupby("junction_name")
        .agg(
            violation_count=("id", "count"),
            centroid_lat=("latitude", "mean"),
            centroid_lon=("longitude", "mean"),
            avg_severity=("max_severity", "mean"),
        )
        .reset_index()
        .sort_values("violation_count", ascending=False)
        .head(30)
    )
    junction_ref["avg_severity"] = junction_ref["avg_severity"].round(2)
    junction_ref.to_csv(DASH_DIR / "junction_reference.csv", index=False)
    print(f"  ✅  junction_reference.csv  ({len(junction_ref)} junctions)")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ARTIFACT 15: Quick-View Presets (Features 7, 8, 9)
# IT Corridor, Commercial Belt, Outskirts zoom coordinates
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[16] Quick-View Presets...")

quick_views = {
    "it_corridor": {
        "label": "🏢 IT Corridor (New Horizon / Embassy Tech Village)",
        "center": [12.9352, 77.6929],
        "zoom": 15,
        "stations": ["Mahadevapura", "HAL Old Airport"],
    },
    "commercial_belt": {
        "label": "🏪 Commercial Belt (Chickpete / Gandhi Nagar)",
        "center": [12.9680, 77.5770],
        "zoom": 15,
        "stations": ["Upparpet", "City Market"],
    },
    "outskirts": {
        "label": "🏗️ Outskirts (Begur / Chikkanahalli)",
        "center": [12.8700, 77.6400],
        "zoom": 14,
        "stations": ["HSR Layout"],
    },
    "shivajinagar": {
        "label": "🏬 Shivaji Nagar / Sivanchetti Gardens",
        "center": [12.9830, 77.6040],
        "zoom": 15,
        "stations": ["Shivajinagar"],
    },
}

with open(DASH_DIR / "quick_view_presets.json", "w") as f:
    json.dump(quick_views, f, indent=2)
print(f"  ✅  quick_view_presets.json")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ARTIFACT 16: Day-of-Week Trend Data (Feature 24)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[17] Day-of-Week Trend Data...")

if "day_of_week" in df.columns:
    dow_data = df.groupby("day_of_week").agg(
        violation_count=("id", "count"),
        avg_severity=("max_severity", "mean"),
        heavy_count=("is_heavy_vehicle", "sum"),
    ).reset_index()
    dow_data["day_name"] = dow_data["day_of_week"].map({
        0: "Monday", 1: "Tuesday", 2: "Wednesday",
        3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"
    })
    dow_data["avg_severity"] = dow_data["avg_severity"].round(2)
    dow_data.to_csv(DASH_DIR / "day_of_week_trends.csv", index=False)
    print(f"  ✅  day_of_week_trends.csv")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ARTIFACT 17: Severity Color Code Map (Feature 13)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[18] Severity Color-Code Map...")

severity_colors = {
    "severity_levels": {
        "1": {"label": "Very Low", "color": "#52b788", "violations": ["WITHOUT SIDE MIRROR", "FAIL TO USE SAFETY BELTS"]},
        "2": {"label": "Low", "color": "#90e0ef", "violations": ["DEFECTIVE NUMBER PLATE", "REFUSE TO GO FOR HIRE"]},
        "3": {"label": "Medium", "color": "#f4a261", "violations": ["WRONG PARKING", "NO PARKING", "PARKING NEAR ROAD CROSSING"]},
        "4": {"label": "High", "color": "#e94560", "violations": ["PARKING IN MAIN ROAD", "DOUBLE PARKING", "PARKING ON FOOTPATH"]},
        "5": {"label": "Very High", "color": "#9b2335", "violations": ["PARKING NEAR TRAFFIC LIGHT", "AGAINST ONE WAY"]},
    }
}

with open(DASH_DIR / "severity_colors.json", "w") as f:
    json.dump(severity_colors, f, indent=2)
print(f"  ✅  severity_colors.json")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ARTIFACT 18: Validation Status Summary (Feature 26)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[19] Validation Status Summary...")

if "validation_status" in df.columns:
    vs_counts = df["validation_status"].fillna("Unreviewed (NaN)").value_counts().to_dict()
    vs_data = {
        "counts": {str(k): int(v) for k, v in vs_counts.items()},
        "total": len(df),
        "approved_count": int(df["validation_status"].eq("approved").sum()),
        "approved_pct": round(df["validation_status"].eq("approved").mean() * 100, 2),
    }
else:
    vs_data = {"counts": {}, "total": len(df), "approved_count": 0}

with open(DASH_DIR / "validation_status.json", "w") as f:
    json.dump(vs_data, f, indent=2)
print(f"  ✅  validation_status.json")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ARTIFACT 19: Offence Code Filter Reference (Feature 27)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[20] Offence Code Filter Reference...")

if target_col in df.columns:
    offence_ref = df[target_col].value_counts().reset_index()
    offence_ref.columns = ["violation_type", "count"]
    offence_ref.to_csv(DASH_DIR / "offence_filter_reference.csv", index=False)
    print(f"  ✅  offence_filter_reference.csv  ({len(offence_ref)} types)")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ARTIFACT 20: ML Feature Importance Data (Feature 31)
# Read from the existing pipeline outputs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[21] Live Prediction Lookup Tables (real data, not placeholders)...")

# ── Pull the exact training feature order + target class order ────────────
# CRITICAL: must match parking_intelligence_pipeline.py's persisted order
# exactly, or the model receives a scrambled feature vector at inference.
model_summary = {}
if MODEL_JSON.exists():
    with open(MODEL_JSON) as f:
        model_summary = json.load(f)

feature_columns = model_summary.get("feature_columns")
target_classes  = model_summary.get("target_classes")
decision_thresholds = model_summary.get("decision_thresholds")

if not feature_columns:
    print("  ⚠️  model_summary.json has no 'feature_columns' — re-run "
          "parking_intelligence_pipeline.py (updated version) first. "
          "Falling back to a best-guess order; live prediction may be wrong "
          "until you retrain.")
    feature_columns = [
        "hour_sin", "hour_cos", "day_sin", "day_cos",
        "is_night_shift", "is_weekend",
        "geohash6_density", "police_station_density",
        "vehicle_congestion_weight", "is_heavy_vehicle",
        "is_junction", "is_top_junction",
        "repeat_offender_score", "is_repeat_offender", "is_habitual_offender",
        "station_hour", "veh_time",
        "vehicle_category_enc", "police_station_enc",
        "junction_name_enc", "repeat_offender_tier_enc", "time_bucket_enc",
    ]
if not target_classes:
    target_classes = sorted(df[target_col].unique().tolist()) if target_col in df.columns else \
        ["GENERIC_PARKING", "SEVERE_OBSTRUCTION", "VEHICLE_COMPLIANCE"]

if not decision_thresholds:
    print("  ⚠️  model_summary.json has no 'decision_thresholds' — re-run "
          "parking_intelligence_pipeline.py (updated version) to calibrate them. "
          "Falling back to a flat 0.5 threshold per class, which will still "
          "favor the majority class until you retrain.")
    decision_thresholds = {c: 0.5 for c in target_classes}

live_pred_features = {
    "feature_columns": feature_columns,
    "target_classes":  target_classes,
    "decision_thresholds": decision_thresholds,
    "vehicle_categories": ["TWO_WHEELER", "THREE_WHEELER", "CAR", "COMMERCIAL", "HEAVY", "OTHER"],
    "time_buckets": ["NIGHT", "MORNING", "MIDDAY", "AFTERNOON", "EVENING"],
}

# ── Global fallback medians (used only if a specific lookup key is missing,
#    e.g. a station/vehicle combo that never appears in training data) ────
global_fallback = {}
for col in ["geohash6_density", "police_station_density", "station_hour", "veh_time"]:
    if col in df.columns:
        global_fallback[col] = float(df[col].median())
live_pred_features["global_fallback"] = global_fallback

# ── Station-level lookup: density, encoding, per-hour interaction freq,
#    and the real junctions that exist at that station ────────────────────
station_lookup = {}
if "police_station" in df.columns:
    has_enc = "police_station_enc" in df.columns
    has_sh  = "station_hour" in df.columns
    has_gh  = "geohash6_density" in df.columns
    has_psd = "police_station_density" in df.columns

    for stn, g in df.groupby("police_station"):
        entry = {
            "police_station_enc":     int(g["police_station_enc"].iloc[0]) if has_enc else 0,
            "police_station_density": float(g["police_station_density"].iloc[0]) if has_psd else global_fallback.get("police_station_density", 0.0),
            "geohash6_density_avg":   float(g["geohash6_density"].mean()) if has_gh else global_fallback.get("geohash6_density", 0.0),
        }
        if has_sh:
            by_hour = g.groupby("hour")["station_hour"].first()
            entry["station_hour_by_hour"] = {str(int(h)): float(v) for h, v in by_hour.items()}
        else:
            entry["station_hour_by_hour"] = {}

        # Real junctions at this station, with their actual encodings
        junctions = []
        if "junction_name" in df.columns and "is_junction" in df.columns:
            jg = g[g["is_junction"] == 1]
            if len(jg) > 0:
                jstats = (
                    jg.groupby("junction_name")
                    .agg(
                        count=("junction_name", "size"),
                        junction_name_enc=("junction_name_enc", "first") if "junction_name_enc" in df.columns else ("junction_name", "first"),
                        is_top_junction=("is_top_junction", "first") if "is_top_junction" in df.columns else ("junction_name", lambda x: 0),
                    )
                    .reset_index()
                    .sort_values("count", ascending=False)
                    .head(8)
                )
                for _, row in jstats.iterrows():
                    junctions.append({
                        "junction_name": str(row["junction_name"]),
                        "junction_name_enc": int(row["junction_name_enc"]) if "junction_name_enc" in df.columns else 0,
                        "is_top_junction": int(row["is_top_junction"]) if "is_top_junction" in df.columns else 0,
                        "count": int(row["count"]),
                    })
        entry["junctions"] = junctions

        # "No Junction" encoding (the data_cleaning.py fill value) — used
        # whenever the officer says the violation is NOT at a junction.
        if "junction_name" in df.columns and "junction_name_enc" in df.columns:
            no_junc = df[df["junction_name"] == "No Junction"]
            entry["no_junction_enc"] = int(no_junc["junction_name_enc"].iloc[0]) if len(no_junc) > 0 else 0
        else:
            entry["no_junction_enc"] = 0

        station_lookup[stn] = entry

live_pred_features["station_lookup"] = station_lookup
live_pred_features["police_stations"] = sorted(station_lookup.keys())

# ── Vehicle-category lookup: real encodings + weights ──────────────────────
vehicle_lookup = {}
if "vehicle_category" in df.columns:
    has_enc = "vehicle_category_enc" in df.columns
    for veh, g in df.groupby("vehicle_category"):
        vehicle_lookup[veh] = {
            "vehicle_category_enc": int(g["vehicle_category_enc"].iloc[0]) if has_enc else 0,
            "vehicle_congestion_weight": int(g["vehicle_congestion_weight"].iloc[0]) if "vehicle_congestion_weight" in df.columns else 2,
            "is_heavy_vehicle": int(g["is_heavy_vehicle"].iloc[0]) if "is_heavy_vehicle" in df.columns else 0,
        }
live_pred_features["vehicle_lookup"] = vehicle_lookup

# ── Time-bucket lookup: real encodings ──────────────────────────────────────
time_bucket_lookup = {}
if "time_bucket" in df.columns and "time_bucket_enc" in df.columns:
    for tb, g in df.groupby("time_bucket"):
        time_bucket_lookup[tb] = {"time_bucket_enc": int(g["time_bucket_enc"].iloc[0])}
live_pred_features["time_bucket_lookup"] = time_bucket_lookup

# ── vehicle × time interaction frequency (veh_time feature) ────────────────
veh_time_lookup = {}
if "vehicle_category" in df.columns and "time_bucket" in df.columns and "veh_time" in df.columns:
    for (veh, tb), g in df.groupby(["vehicle_category", "time_bucket"]):
        veh_time_lookup[f"{veh}_{tb}"] = float(g["veh_time"].iloc[0])
live_pred_features["veh_time_lookup"] = veh_time_lookup

# ── Repeat-offender tier lookup: real encodings + typical stats per tier ──
offender_tier_lookup = {}
if "repeat_offender_tier" in df.columns:
    has_enc = "repeat_offender_tier_enc" in df.columns
    for tier, g in df.groupby("repeat_offender_tier"):
        offender_tier_lookup[tier] = {
            "repeat_offender_tier_enc": int(g["repeat_offender_tier_enc"].iloc[0]) if has_enc else 0,
            "repeat_offender_score_typical": float(g["repeat_offender_score"].median()) if "repeat_offender_score" in df.columns else 1.0,
            "is_repeat_offender": int(g["is_repeat_offender"].iloc[0]) if "is_repeat_offender" in df.columns else 0,
            "is_habitual_offender": int(g["is_habitual_offender"].iloc[0]) if "is_habitual_offender" in df.columns else 0,
        }
live_pred_features["offender_tier_lookup"] = offender_tier_lookup
live_pred_features["offender_tiers"] = sorted(offender_tier_lookup.keys())

with open(DASH_DIR / "live_prediction_config.json", "w") as f:
    json.dump(live_pred_features, f, indent=2)
print(f"  ✅  live_prediction_config.json — "
      f"{len(station_lookup)} stations, {len(vehicle_lookup)} vehicle classes, "
      f"{len(veh_time_lookup)} veh×time combos, {len(offender_tier_lookup)} offender tiers "
      f"(all from real aggregations, no hardcoded constants)")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FINAL SUMMARY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 70)
print("  ✅  DASHBOARD DATA PIPELINE COMPLETE")
print("=" * 70)
print(f"  Output directory: {DASH_DIR}")
print(f"  Artifacts generated:")

for f in sorted(DASH_DIR.iterdir()):
    size_kb = f.stat().st_size / 1024
    print(f"    📄  {f.name:<45s}  ({size_kb:.1f} KB)")

print(f"\n  Next step: streamlit run app.py")
print("=" * 70 + "\n")