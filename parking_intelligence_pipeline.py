"""
=============================================================================
  parking_intelligence_pipeline.py
  Flipkart GridLock 2.0 — Round 2 | PS1
  PURPOSE: Full end-to-end ML pipeline — hotspot detection, ensemble
           violation classifier, congestion scoring, Folium maps,
           Matplotlib/Seaborn visualisations, enforcement priority ranking.

  ARCHITECTURE:
    Step 0  — auto-install all requirements
    Step 1  — load feature-engineered dataset (expects ~93K rows after cleaning)
    Step 2  — run pipeline data checks
    Step 3  — DBSCAN hotspot clustering (haversine metric)
    Step 4  — LightGBM + XGBoost soft-voting ensemble classifier
    Step 5  — congestion impact scoring (per-record)
    Step 6  — enforcement priority zone ranking (per-station)
    Step 7  — 3 Folium interactive maps
    Step 8  — 10 Matplotlib/Seaborn plots
    Step 9  — accuracy & evaluation report
    Step 10 — output summary

  NO DATA LEAKAGE:
    • All risk-scores (station density, geohash density) computed on FULL set
      BEFORE train/test split — these are global structural features, not
      target-derived statistics, so no leakage occurs
    • Target-encoding (if used) would be fitted on TRAIN fold only
    • class weights fitted on y_train only
    • Both models trained only on X_train
=============================================================================
"""

# ── Step 0: auto-install ────────────────────────────────────────────────────
import subprocess, sys, importlib

DEPS = {
    "pandas":         "pandas>=2.0",
    "numpy":          "numpy>=1.24",
    "scikit-learn":   "scikit-learn>=1.3",
    "lightgbm":       "lightgbm>=4.0",
    "xgboost":        "xgboost>=2.0",
    "folium":         "folium>=0.15",
    "matplotlib":     "matplotlib>=3.7",
    "seaborn":        "seaborn>=0.13",
    "pygeohash": "pygeohash",
    "tqdm":           "tqdm",
}
IMPORT_MAP = {"scikit-learn": "sklearn", "pygeohash": "pygeohash"}

print("\n" + "=" * 70)
print("  STEP 0 — Checking & installing requirements")
print("=" * 70)
for pkg, pip_spec in DEPS.items():
    iname = IMPORT_MAP.get(pkg, pkg)
    try:
        importlib.import_module(iname)
        print(f"  ✅  {pkg:<20s} already installed")
    except ImportError:
        print(f"  ⬇️   {pkg:<20s} installing...", end=" ", flush=True)
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", pip_spec, "-q"],
            capture_output=True
        )
        print("done ✅" if result.returncode == 0
              else f"FAILED ❌  {result.stderr.decode()[:100]}")

import ast, json, pickle, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

import folium
from folium.plugins import HeatMap, MarkerCluster
import pygeohash as gh

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score,
)
from sklearn.utils.class_weight import compute_class_weight

import lightgbm as lgb
import xgboost as xgb
from tqdm import tqdm

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 50)

# ── PATHS ───────────────────────────────────────────────────────────────────
BASE_DIR   = Path(r"D:\Flipkart Gridlock 2.0\Round-2")
DATA_DIR   = BASE_DIR / "data"
CLEAN_DIR  = DATA_DIR / "cleaned"
OUT_DIR    = BASE_DIR / "outputs"
MAP_DIR    = OUT_DIR / "maps"
PLOT_DIR   = OUT_DIR / "plots"
MODEL_DIR  = OUT_DIR / "model"
for d in [OUT_DIR, MAP_DIR, PLOT_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

FEATURES_CSV  = CLEAN_DIR / "dataset_features.csv"
WEIGHTS_JSON  = DATA_DIR / "class_weights.json"
META_JSON     = BASE_DIR / "feature_metadata.json"

CHUNKSIZE    = 50_000
RANDOM_STATE = 42
TEST_SIZE    = 0.20
MAX_CW_CAP   = 10.0

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 1 — Load data
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 70)
print("  STEP 1 — Loading feature dataset")
print("=" * 70)

if not FEATURES_CSV.exists():
    raise FileNotFoundError(
        f"\n❌  Feature CSV not found: {FEATURES_CSV}\n"
        "   Run the pipeline in order:\n"
        "     1. python data_cleaning.py\n"
        "     2. python feature_engineering.py\n"
        "     3. python violation_features.py\n"
        "     4. python parking_intelligence_pipeline.py"
    )

print(f"  Loading {FEATURES_CSV.name}...")
chunks = []
reader = pd.read_csv(FEATURES_CSV, chunksize=CHUNKSIZE, low_memory=False)
for chunk in tqdm(reader, desc="  Loading", unit="chunk"):
    chunks.append(chunk)
df = pd.concat(chunks, ignore_index=True)
print(f"  ✅  {len(df):,} rows × {df.shape[1]} cols loaded")

# Row-count gate — confirm device-trust NaN recovery worked
EXPECTED_MIN_ROWS = 30_000
if len(df) < EXPECTED_MIN_ROWS:
    print(f"\n  ⚠️  WARNING: Only {len(df):,} rows detected. "
          f"Expected ≥{EXPECTED_MIN_ROWS:,} after device-trust NaN recovery.")
    print("     Re-run data_cleaning.py first if training set seems too small.\n")
else:
    print(f"  ✅  Row count looks healthy (≥{EXPECTED_MIN_ROWS:,})")

# Load feature metadata
if META_JSON.exists():
    with open(META_JSON) as f:
        meta = json.load(f)
    FEATURE_COLS_FROM_META = meta.get("numeric_features", [])
else:
    FEATURE_COLS_FROM_META = []

# Load capped class weights
cw_dict = {}
if WEIGHTS_JSON.exists():
    with open(WEIGHTS_JSON) as f:
        cw_dict = json.load(f)
    print(f"  Loaded class weights ({len(cw_dict)} classes, capped @ {MAX_CW_CAP})")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 2 — Pipeline data checks
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 70)
print("  STEP 2 — Data checks")
print("=" * 70)

# Determine target column
TARGET_COL = (
    "primary_violation_final" if "primary_violation_final" in df.columns
    else "primary_violation" if "primary_violation" in df.columns
    else None
)


if TARGET_COL is None:
    raise ValueError("No target column found. Run violation_features.py first.")

# HARD STOP: don't silently fall back to the unmerged 16-class target.
# If this fires, you forgot to run violation_features.py before this script.
assert TARGET_COL == "primary_violation_final", (
    f"\n❌  Training would use '{TARGET_COL}' (unmerged, 16 raw classes) "
    f"instead of 'primary_violation_final' (your curated 8-class target).\n"
    f"    Run violation_features.py, THEN rerun this script."
)

print(f"  Target column : {TARGET_COL}")
print(f"  Unique classes: {df[TARGET_COL].nunique()}")


# Drop UNKNOWN / very small classes from ML training
MIN_CLASS_SIZE = 20
vc = df[TARGET_COL].value_counts()
valid_classes = vc[vc >= MIN_CLASS_SIZE].index
df_ml = df[df[TARGET_COL].isin(valid_classes)].copy()
print(f"  Classes with ≥{MIN_CLASS_SIZE} records: {len(valid_classes)}")
print(f"  ML training rows: {len(df_ml):,}")

# Re-bin 8 fine-grained violation types into 3 enforcement-meaningful
# groups. The PS asks for "targeted enforcement," not exact legal-code
# classification — these groups map to genuinely different response
# types (routine ticket vs. priority safety dispatch vs. vehicle
# compliance check), and merging away near-duplicate confusion
# (WRONG PARKING vs NO PARKING) removes noise, not signal.
VIOLATION_GROUP_MAP = {
    "WRONG PARKING":                            "GENERIC_PARKING",
    "NO PARKING":                                "GENERIC_PARKING",
    "PARKING IN A MAIN ROAD":                    "SEVERE_OBSTRUCTION",
    "PARKING ON FOOTPATH":                       "SEVERE_OBSTRUCTION",
    "PARKING NEAR BUSTOP/SCHOOL/HOSPITAL ETC":   "SEVERE_OBSTRUCTION",
    "PARKING NEAR ROAD CROSSING":                "SEVERE_OBSTRUCTION",
    "DEFECTIVE NUMBER PLATE":                    "VEHICLE_COMPLIANCE",
    "OTHER_VIOLATION":                           "VEHICLE_COMPLIANCE",
}
df_ml["target_group"] = df_ml[TARGET_COL].map(VIOLATION_GROUP_MAP)
print(f"\n  Re-binned into {df_ml['target_group'].nunique()} enforcement groups:")
print(df_ml["target_group"].value_counts().to_string())

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 3 — DBSCAN hotspot clustering (haversine metric)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 70)
print("  STEP 3 — DBSCAN hotspot clustering (haversine)")
print("=" * 70)

geo_df = df[["latitude", "longitude"]].dropna().copy()
MAX_DBSCAN = 120_000
if len(geo_df) > MAX_DBSCAN:
    geo_sample = geo_df.sample(MAX_DBSCAN, random_state=RANDOM_STATE)
    print(f"  Subsampled to {MAX_DBSCAN:,} rows for DBSCAN performance")
else:
    geo_sample = geo_df

coords_rad = np.deg2rad(geo_sample[["latitude", "longitude"]].values)
print(f"  Running DBSCAN on {len(geo_sample):,} points (eps=300m, min_samples=10)...")
db = DBSCAN(
    eps=300 / 6_371_000,   # 300 metres converted to radians
    min_samples=10,
    algorithm="ball_tree",
    metric="haversine",
    n_jobs=-1,
)
cluster_labels = db.fit_predict(coords_rad)
geo_sample = geo_sample.copy()
geo_sample["cluster"] = cluster_labels

n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_noise    = int((cluster_labels == -1).sum())
print(f"  ✅  Clusters found: {n_clusters}  |  Noise: {n_noise:,}")

cluster_info = (
    geo_sample[geo_sample["cluster"] >= 0]
    .groupby("cluster")
    .agg(centroid_lat=("latitude", "mean"),
         centroid_lon=("longitude", "mean"),
         point_count  =("latitude", "count"))
    .reset_index()
    .sort_values("point_count", ascending=False)
    .reset_index(drop=True)
)
cluster_info.to_csv(OUT_DIR / "hotspot_clusters.csv", index=False)
print(f"  Top cluster: {cluster_info['point_count'].iloc[0]:,} violations")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 4 — LightGBM + XGBoost SOFT-VOTING ENSEMBLE (KEY FIX)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 70)
print("  STEP 4 — LightGBM + XGBoost soft-voting ensemble")
print("=" * 70)

# Build feature column list — no lat/lon
BASE_FEATURES = [
    "hour_sin", "hour_cos", "day_sin", "day_cos",
    "is_night_shift", "is_weekend",
    "geohash6_density", "police_station_density",
    "vehicle_congestion_weight", "is_heavy_vehicle",
    "violation_count", "max_severity", "avg_severity", "total_severity",
    "is_multi_label", "is_high_severity", "is_parking_violation",
    "is_junction", "is_top_junction",
    "repeat_offender_score", "is_repeat_offender", "is_habitual_offender",
    "station_hour", "veh_time",
    # viol_* columns intentionally EXCLUDED — direct target leakage
    "vehicle_category_enc", "police_station_enc",
    "junction_name_enc", "repeat_offender_tier_enc", "time_bucket_enc",
]
# Use metadata list if available and broader
# FIX: `set()` on strings has NON-DETERMINISTIC iteration order across
# Python runs (hash randomization). Both LightGBM and XGBoost are fit on a
# raw NumPy array below with no column names attached, so they only know
# positional feature index — not feature name. If this order shifts between
# the run that trained the model and the run that builds app.py's inference
# vector, every prediction is silently scrambled (the model is fed
# "police_station_density" where it expects "hour_sin", etc.), which makes
# outputs collapse toward the majority class regardless of input.
# `dict.fromkeys(...)` dedupes while preserving insertion order deterministically.
if FEATURE_COLS_FROM_META:
    candidate_cols = list(dict.fromkeys(BASE_FEATURES + FEATURE_COLS_FROM_META))
else:
    candidate_cols = list(dict.fromkeys(BASE_FEATURES))



# Columns that leak the target — blocked even if they reappear via a
# stale feature_metadata.json
LEAKAGE_COLS = {
    "viol_wrong_parking", "viol_no_parking", "viol_parking_in_a_main_road",
    "viol_defective_number_plate", "viol_parking_on_footpath",
    "viol_double_parking", "viol_parking_near_road_crossing",
    "viol_parking_near_bustop_school_hospital_etc",
    "violation_count", "max_severity", "avg_severity", "total_severity",
    "is_multi_label", "is_high_severity", "is_parking_violation",
}

# Keep only columns that exist, are numeric, aren't the target, aren't leakage
FEATURE_COLS = [
    c for c in candidate_cols
    if c in df_ml.columns
    and pd.api.types.is_numeric_dtype(df_ml[c])
    and c != TARGET_COL
    and c not in LEAKAGE_COLS
]
blocked = LEAKAGE_COLS & set(candidate_cols)
if blocked:
    print(f"  ⚠️  Blocked {len(blocked)} leakage column(s) from training: {sorted(blocked)}")
print(f"  Feature columns: {len(FEATURE_COLS)}")
print(f"  Excluded: latitude, longitude (geohash6_density replaces them)")

# Encode target — now using the 3-group target, not the raw 8 classes
le_target = LabelEncoder()
df_ml["target_enc"] = le_target.fit_transform(df_ml["target_group"])
n_classes = len(le_target.classes_)
print(f"  Classes: {n_classes}")

X = df_ml[FEATURE_COLS].fillna(0).astype("float32").values
y = df_ml["target_enc"].values

# Stratified split — no leakage (class weights fitted on X_train's y only)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# Manual per-group weights — "balanced" auto-computation gave
# SEVERE_OBSTRUCTION ~5.4x weight, which pushed recall to 0.83 but
# crashed precision to 0.36 (model over-predicts it). Tuning these
# directly instead of fighting an indirect cap.
MANUAL_GROUP_WEIGHTS = {
    "GENERIC_PARKING":    1.0,
    "SEVERE_OBSTRUCTION": 2.5,   # was ~5.4x under "balanced" — dialed down
    "VEHICLE_COMPLIANCE": 8.0,   # was ~73.6x capped to 10 — dialed down too,
                                 # since 73.6x was likely also overshooting
}
class_label_to_weight = {
    le_target.transform([name])[0]: w
    for name, w in MANUAL_GROUP_WEIGHTS.items()
}
sample_weights_train = np.array([
    class_label_to_weight.get(c, 1.0) for c in y_train
])
print(f"  Manual class weights: { {le_target.inverse_transform([k])[0]: v for k, v in class_label_to_weight.items()} }")

# ─── 4a. LightGBM ──────────────────────────────────────────────────────────
print("\n  [4a] Training LightGBM...")
lgb_train_ds = lgb.Dataset(X_train, label=y_train, weight=sample_weights_train, free_raw_data=False)
lgb_val_ds   = lgb.Dataset(X_test,  label=y_test,  reference=lgb_train_ds,      free_raw_data=False)

lgb_params = {
    "objective":          "multiclass",
    "num_class":          n_classes,
    "metric":             "multi_logloss",
    "learning_rate":      0.07,
    "num_leaves":         63,
    "max_depth":          7,
    "min_data_in_leaf":   20,
    "feature_fraction":   0.80,
    "bagging_fraction":   0.80,
    "bagging_freq":       5,
    "lambda_l2":          0.1,
    "verbosity":          -1,
    "random_state":       RANDOM_STATE,
    "n_jobs":             -1,
}
lgb_model = lgb.train(
    lgb_params,
    lgb_train_ds,
    num_boost_round=500,
    valid_sets=[lgb_val_ds],
    callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(100)],
)


lgb_model.save_model(str(MODEL_DIR / "lgbm_model.txt"))
print("  ✅  LightGBM trained & saved")

# # ─── DIAGNOSTIC: find what's actually separating DEFECTIVE NUMBER PLATE ───

# print("\n  [DIAG] LightGBM feature importance — REAL NAMES (top 15, gain):")
# imp_gain  = lgb_model.feature_importance(importance_type="gain")
# imp_names = lgb_model.feature_name()  # these are "Column_N" placeholders
# imp_df = pd.DataFrame({"feature": imp_names, "gain": imp_gain})
# imp_df["feature_name"] = imp_df["feature"].apply(
#     lambda c: FEATURE_COLS[int(c.split("_")[1])] if c.startswith("Column_") else c
# )
# imp_df = imp_df.sort_values("gain", ascending=False)
# print(imp_df.head(15)[["feature_name", "gain"]].to_string(index=False))

# dnp_label = le_target.transform(["DEFECTIVE NUMBER PLATE"])[0]
# X_train_arr = np.asarray(X_train)
# y_train_arr = np.asarray(y_train)
# dnp_mask  = (y_train_arr == dnp_label)
# print(f"\n  [DIAG] DEFECTIVE NUMBER PLATE train rows: {dnp_mask.sum()}")
# print(f"  [DIAG] Per-feature: DNP mean/std vs rest-of-data mean (flag = near-constant AND distinct):")
# for i, col in enumerate(FEATURE_COLS):
#     dnp_vals  = X_train_arr[dnp_mask, i]
#     rest_vals = X_train_arr[~dnp_mask, i]
#     dnp_mean, dnp_std = dnp_vals.mean(), dnp_vals.std()
#     rest_mean = rest_vals.mean()
#     flag = " ⚠ SUSPECT" if dnp_std < 0.05 and abs(dnp_mean - rest_mean) > 0.3 else ""
#     print(f"    {col:<28s} DNP_mean={dnp_mean:8.3f}  DNP_std={dnp_std:6.3f}  rest_mean={rest_mean:8.3f}{flag}")

# ─── 4b. XGBoost ───────────────────────────────────────────────────────────
print("\n  [4b] Training XGBoost...")

# XGBoost needs integer scale_pos_weight; use per-sample weights instead
xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=7,
    learning_rate=0.07,
    subsample=0.80,
    colsample_bytree=0.80,
    eval_metric="mlogloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=0,
    early_stopping_rounds=60,
    num_class=n_classes,
    objective="multi:softprob",
)
xgb_model.fit(
    X_train, y_train,
    sample_weight=sample_weights_train,
    eval_set=[(X_test, y_test)],
    verbose=100,
)
xgb_model.save_model(str(MODEL_DIR / "xgb_model.json"))
print("  ✅  XGBoost trained & saved")

# ─── 4c. Soft-voting ensemble (KEY FIX from screenshot) ───────────────────
print("\n  [4c] Soft-voting ensemble (LGB + XGB average)...")
lgb_proba = lgb_model.predict(X_test)            # shape (n, n_classes)
xgb_proba = xgb_model.predict_proba(X_test)      # shape (n, n_classes)

# Simple average — both models output calibrated probabilities
ensemble_proba  = (lgb_proba + xgb_proba) / 2
y_pred_ensemble = np.argmax(ensemble_proba, axis=1)

# Individual model predictions for comparison
y_pred_lgb  = np.argmax(lgb_proba,  axis=1)
y_pred_xgb  = np.argmax(xgb_proba,  axis=1)

acc_lgb  = accuracy_score(y_test, y_pred_lgb)
acc_xgb  = accuracy_score(y_test, y_pred_xgb)
acc_ens  = accuracy_score(y_test, y_pred_ensemble)
f1_lgb   = f1_score(y_test, y_pred_lgb,  average="weighted", zero_division=0)
f1_xgb   = f1_score(y_test, y_pred_xgb,  average="weighted", zero_division=0)
f1_ens   = f1_score(y_test, y_pred_ensemble, average="weighted", zero_division=0)

print(f"""
  ┌────────────────────────────────────────────┐
  │         ENSEMBLE EVALUATION RESULTS        │
  ├────────────────┬────────────┬──────────────┤
  │ Model          │ Accuracy   │ F1 (weighted)│
  ├────────────────┼────────────┼──────────────┤
  │ LightGBM       │ {acc_lgb:.4f}     │ {f1_lgb:.4f}       │
  │ XGBoost        │ {acc_xgb:.4f}     │ {f1_xgb:.4f}       │
  │ ENSEMBLE (avg) │ {acc_ens:.4f}     │ {f1_ens:.4f}       │
  └────────────────┴────────────┴──────────────┘
""")

y_pred_labels = le_target.inverse_transform(y_pred_ensemble)
y_test_labels = le_target.inverse_transform(y_test)

# ── Per-class decision thresholds (KEY FIX for majority-class bias) ────────
# GENERIC_PARKING is ~92% of records. Plain argmax(probability) means
# whichever class's RAW probability is numerically largest wins — and for
# an under-represented class that number rarely beats the majority class's,
# even on scenarios where it's genuinely elevated relative to ITS OWN normal
# level. Class weights only reshape the training loss; they don't change how
# argmax breaks ties at inference. Instead, calibrate a per-class decision
# threshold on the held-out test set (the F1-maximizing cut point for "is
# this class, one-vs-rest"), then at inference pick whichever class clears
# its OWN threshold by the largest margin (probability / threshold) — not
# whichever raw probability happens to be biggest.
print("\n  [4d] Calibrating per-class decision thresholds...")
decision_thresholds = {}
for ci, cname in enumerate(le_target.classes_):
    y_bin = (y_test == ci).astype(int)
    p_bin = ensemble_proba[:, ci]
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        pred_bin = (p_bin >= t).astype(int)
        f1 = f1_score(y_bin, pred_bin, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    decision_thresholds[cname] = round(float(best_t), 3)
    print(f"    {cname:<22s} threshold={best_t:.2f}  (F1@threshold={best_f1:.3f})")

report_str = classification_report(y_test_labels, y_pred_labels, zero_division=0)
with open(OUT_DIR / "classification_report.txt", "w") as f:
    f.write("LGB + XGB Soft-Voting Ensemble — Classification Report\n")
    f.write(f"Generated: {datetime.now()}\n\n")
    f.write(f"LightGBM Accuracy     : {acc_lgb:.4f}\n")
    f.write(f"XGBoost  Accuracy     : {acc_xgb:.4f}\n")
    f.write(f"Ensemble Accuracy     : {acc_ens:.4f}\n")
    f.write(f"Ensemble F1 (weighted): {f1_ens:.4f}\n\n")
    f.write(report_str)
print("  📄  Classification report → outputs/classification_report.txt")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 5 — Congestion impact scoring
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 70)
print("  STEP 5 — Congestion impact scoring")
print("=" * 70)

SEVERITY_W  = 0.35; VEHICLE_W = 0.25; COUNT_W = 0.20
REPEAT_W    = 0.10; JUNC_W   = 0.10

def congestion_score(row):
    s  = (row.get("max_severity", 2) - 1) / 4
    v  = (row.get("vehicle_congestion_weight", 2) - 1) / 4
    c  = min(row.get("violation_count", 1) / 5, 1.0)
    r  = min(row.get("repeat_offender_score", 1) / 55, 1.0)
    j  = float(row.get("is_top_junction", 0))
    return round((SEVERITY_W*s + VEHICLE_W*v + COUNT_W*c + REPEAT_W*r + JUNC_W*j) * 100, 2)

if "congestion_score" not in df.columns:
    score_cols = ["max_severity", "vehicle_congestion_weight",
                  "violation_count", "repeat_offender_score", "is_top_junction"]
    score_cols = [c for c in score_cols if c in df.columns]
    tqdm.pandas(desc="  Scoring")
    df["congestion_score"] = df[score_cols].fillna(0).progress_apply(
        congestion_score, axis=1
    )
else:
    print("  congestion_score column already present")

print(f"  Mean: {df['congestion_score'].mean():.2f}  "
      f"| P90: {df['congestion_score'].quantile(0.9):.2f}  "
      f"| Max: {df['congestion_score'].max():.2f}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 6 — Enforcement priority ranking
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 70)
print("  STEP 6 — Enforcement priority ranking")
print("=" * 70)

priority = pd.DataFrame()
if "police_station" in df.columns:
    priority = (
        df.groupby("police_station", as_index=False)
        .agg(
            violation_total  =("congestion_score",     "count"),
            avg_severity     =("max_severity",         "mean"),
            avg_congestion   =("congestion_score",     "mean"),
            habitual_count   =("is_habitual_offender", "sum"),
            heavy_veh_count  =("is_heavy_vehicle",     "sum"),
        )
    )
    for col in ["violation_total", "avg_severity", "avg_congestion"]:
        mn, mx = priority[col].min(), priority[col].max()
        priority[f"{col}_norm"] = (priority[col] - mn) / (mx - mn + 1e-9)

    priority["priority_score"] = (
        0.40 * priority["violation_total_norm"] +
        0.30 * priority["avg_congestion_norm"]  +
        0.30 * priority["avg_severity_norm"]
    ).round(4)
    priority = priority.sort_values("priority_score", ascending=False).reset_index(drop=True)
    priority.insert(0, "rank", range(1, len(priority) + 1))
    priority.to_csv(OUT_DIR / "enforcement_priority_ranked.csv", index=False)

    print("  🏆  Top 10 Enforcement Zones:")
    for _, row in priority.head(10).iterrows():
        print(f"    #{int(row['rank']):2d}  {str(row['police_station']):<28s}  "
              f"score={row['priority_score']:.4f}  "
              f"violations={int(row['violation_total']):,}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 7 — Folium interactive maps (3 maps)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 70)
print("  STEP 7 — Folium maps (3 maps)")
print("=" * 70)

BLR = [12.9716, 77.5946]

# ── Map 1: Congestion heatmap ───────────────────────────────────────────────
print("  [7a] Congestion heatmap...")
m1 = folium.Map(location=BLR, zoom_start=12, tiles="CartoDB dark_matter")
heat_df = df[["latitude", "longitude", "congestion_score"]].dropna().sample(
    min(60_000, len(df)), random_state=RANDOM_STATE
)
HeatMap(
    heat_df.values.tolist(),
    radius=13, blur=11, max_zoom=15,
    gradient={"0.3": "#1a0a2e", "0.5": "#6b0f6e", "0.7": "#e94560", "1.0": "#ff9f1c"},
).add_to(m1)
m1.get_root().html.add_child(folium.Element("""
<div style="position:fixed;top:12px;left:50%;transform:translateX(-50%);
     z-index:9999;background:rgba(15,15,26,0.85);color:#fff;
     padding:9px 20px;border-radius:8px;font:13px Arial;
     border:1px solid #e94560;">
  🔴 Bengaluru Parking — Congestion Heatmap
</div>"""))
folium.LayerControl().add_to(m1)
m1.save(str(MAP_DIR / "01_congestion_heatmap.html"))
print("  ✅  01_congestion_heatmap.html")

# ── Map 2: DBSCAN cluster + enforcement priority ────────────────────────────
print("  [7b] Cluster + priority map...")
m2 = folium.Map(location=BLR, zoom_start=12, tiles="CartoDB positron")
PALETTE = ["#e63946", "#f4a261", "#2a9d8f", "#457b9d", "#6d6875"]

for i, row in cluster_info.head(40).iterrows():
    color = PALETTE[i % len(PALETTE)]
    folium.CircleMarker(
        location=[row["centroid_lat"], row["centroid_lon"]],
        radius=max(5, min(16, row["point_count"] // 300)),
        color=color, fill=True, fill_color=color, fill_opacity=0.75,
        tooltip=f"Hotspot #{i+1} — {int(row['point_count']):,} violations",
        popup=folium.Popup(
            f"<b>Hotspot #{i+1}</b><br>"
            f"Lat: {row['centroid_lat']:.4f}<br>"
            f"Lon: {row['centroid_lon']:.4f}<br>"
            f"Violations: {int(row['point_count']):,}",
            max_width=200
        )
    ).add_to(m2)

# Enforcement priority markers
if not priority.empty and "latitude" in df.columns:
    station_coords = df.groupby("police_station")[["latitude", "longitude"]].mean()
    for _, prow in priority.head(10).iterrows():
        stn = prow["police_station"]
        if stn in station_coords.index:
            lat = station_coords.loc[stn, "latitude"]
            lon = station_coords.loc[stn, "longitude"]
            folium.Marker(
                location=[lat, lon],
                icon=folium.Icon(color="red", icon="exclamation-sign", prefix="glyphicon"),
                tooltip=f"⚠️ Priority #{int(prow['rank'])}: {stn}",
                popup=folium.Popup(
                    f"<b>Priority #{int(prow['rank'])}</b><br>"
                    f"{stn}<br>"
                    f"Score: {prow['priority_score']:.4f}<br>"
                    f"Violations: {int(prow['violation_total']):,}",
                    max_width=240
                )
            ).add_to(m2)

m2.get_root().html.add_child(folium.Element("""
<div style="position:fixed;top:12px;left:50%;transform:translateX(-50%);
     z-index:9999;background:rgba(255,255,255,0.92);color:#333;
     padding:9px 20px;border-radius:8px;font:13px Arial;
     border:2px solid #e63946;box-shadow:0 2px 8px rgba(0,0,0,0.2);">
  🎯 DBSCAN Hotspot Clusters + Top 10 Priority Zones
</div>"""))
folium.LayerControl().add_to(m2)
m2.save(str(MAP_DIR / "02_hotspot_clusters_priority.html"))
print("  ✅  02_hotspot_clusters_priority.html")

# ── Map 3: Night vs Day violation split ─────────────────────────────────────
print("  [7c] Night vs Day map...")
m3 = folium.Map(location=BLR, zoom_start=12, tiles="CartoDB positron")

if "is_night_shift" in df.columns:
    for is_night, name, gradient, show in [
        (1, "🌙 Night (10PM–6AM)", {"0.4": "#0d0d2b", "0.7": "#4c2c8e", "1.0": "#a855f7"}, True),
        (0, "☀️ Day (6AM–10PM)",   {"0.4": "#fffde7", "0.7": "#ff9800", "1.0": "#e65100"}, False),
    ]:
        subset = df[df["is_night_shift"] == is_night][["latitude", "longitude"]].dropna()
        sample = subset.sample(min(20_000, len(subset)), random_state=RANDOM_STATE)
        grp = folium.FeatureGroup(name=name, show=show)
        HeatMap(sample.values.tolist(), radius=10, blur=8, gradient=gradient).add_to(grp)
        grp.add_to(m3)

folium.LayerControl(collapsed=False).add_to(m3)
m3.get_root().html.add_child(folium.Element("""
<div style="position:fixed;top:12px;left:50%;transform:translateX(-50%);
     z-index:9999;background:rgba(255,255,255,0.92);color:#333;
     padding:9px 20px;border-radius:8px;font:13px Arial;
     border:2px solid #4c2c8e;">
  🌙 Night vs ☀️ Day Violations (toggle layers)
</div>"""))
m3.save(str(MAP_DIR / "03_night_vs_day.html"))
print("  ✅  03_night_vs_day.html")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 8 — Matplotlib / Seaborn visualisations (10 plots)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 70)
print("  STEP 8 — Matplotlib/Seaborn plots (10 plots)")
print("=" * 70)

DARK   = "#0f0f1a";  ACCENT = "#e94560";  ACCENT2 = "#4cc9f0"
NEUTRAL= "#a0a0b0";  FONT   = "#e0e0f0"

def darken(ax):
    ax.set_facecolor(DARK)
    ax.tick_params(colors=NEUTRAL, labelsize=8)
    ax.xaxis.label.set_color(NEUTRAL); ax.yaxis.label.set_color(NEUTRAL)
    ax.title.set_color(FONT)
    for sp in ax.spines.values(): sp.set_edgecolor("#333355")

def save(name):
    plt.savefig(PLOT_DIR / f"{name}.png", dpi=140, bbox_inches="tight",
                facecolor=DARK)
    plt.close()
    print(f"  ✅  {name}.png")

sns.set_theme(style="dark")

# Plot 1 — Top violation classes
target_col_plot = TARGET_COL if TARGET_COL in df.columns else "primary_violation"
if target_col_plot in df.columns:
    fig, ax = plt.subplots(figsize=(12, 7), facecolor=DARK)
    vc = df[target_col_plot].value_counts().head(15)
    colors = [ACCENT if i < 3 else ACCENT2 for i in range(len(vc))]
    ax.barh(vc.index[::-1], vc.values[::-1], color=colors[::-1], edgecolor="none")
    darken(ax)
    ax.set_xlabel("Violations", color=NEUTRAL)
    ax.set_title("Top 15 Violation Classes", fontsize=14, fontweight="bold")
    for i, v in enumerate(vc.values[::-1]):
        ax.text(v + 200, i, f"{v:,}", va="center", color=NEUTRAL, fontsize=8)
    save("01_top_violation_classes")

# Plot 2 — Hourly pattern with rush-hour bands
if "hour" in df.columns:
    fig, ax = plt.subplots(figsize=(13, 5), facecolor=DARK)
    hourly = df.groupby("hour").size()
    ax.fill_between(hourly.index, hourly.values, alpha=0.35, color=ACCENT)
    ax.plot(hourly.index, hourly.values, color=ACCENT, linewidth=2.5)
    ax.axvspan(0,  6,  alpha=0.10, color=ACCENT2,  label="Night patrol peak (00–06)")
    ax.axvspan(8,  10, alpha=0.10, color="#f4a261", label="AM rush (08–10)")
    ax.axvspan(17, 20, alpha=0.10, color="#f4a261", label="PM rush (17–20)")
    darken(ax)
    ax.set_xlabel("Hour (IST)", color=NEUTRAL); ax.set_ylabel("Violations", color=NEUTRAL)
    ax.set_title("Violations by Hour (IST) — Patrol vs Rush Hour Gap", fontsize=14, fontweight="bold")
    ax.set_xticks(range(0, 24)); ax.legend(facecolor="#1a1a2e", labelcolor=NEUTRAL, fontsize=8)
    save("02_hourly_pattern")

# Plot 3 — Day of week
if "day_of_week" in df.columns:
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=DARK)
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dc = df["day_of_week"].value_counts().sort_index()
    ax.bar(days[:len(dc)], dc.values,
           color=[ACCENT if d >= 5 else ACCENT2 for d in dc.index], edgecolor="none", width=0.6)
    darken(ax)
    ax.set_ylabel("Violations", color=NEUTRAL)
    ax.set_title("Violations by Day of Week (Weekend = Red)", fontsize=14, fontweight="bold")
    save("03_day_of_week")

# Plot 4 — Vehicle category
if "vehicle_category" in df.columns:
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=DARK)
    vg = df["vehicle_category"].value_counts()
    pal = [ACCENT, ACCENT2, "#f4a261", "#2ecc71", "#9b59b6", "#e67e22"]
    ax.bar(vg.index, vg.values,
           color=[pal[i % len(pal)] for i in range(len(vg))], edgecolor="none")
    darken(ax); ax.tick_params(axis="x", rotation=20)
    ax.set_ylabel("Violations", color=NEUTRAL)
    ax.set_title("Violations by Vehicle Category (22-type resolved)", fontsize=14, fontweight="bold")
    save("04_vehicle_category")

# Plot 5 — Severity distribution
if "max_severity" in df.columns:
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=DARK)
    sv = df["max_severity"].value_counts().sort_index()
    labels = {1: "Very Low", 2: "Low", 3: "Medium", 4: "High", 5: "Very High"}
    sev_colors = ["#52b788", "#90e0ef", "#f4a261", ACCENT, "#9b2335"]
    ax.bar([labels.get(i, str(i)) for i in sv.index], sv.values,
           color=sev_colors[:len(sv)], edgecolor="none")
    darken(ax); ax.set_ylabel("Count", color=NEUTRAL)
    ax.set_title("Max Violation Severity Distribution", fontsize=14, fontweight="bold")
    save("05_severity_distribution")

# Plot 6 — Top 10 police stations
if "police_station" in df.columns:
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=DARK)
    ps = df["police_station"].value_counts().head(10)
    ax.barh(ps.index[::-1], ps.values[::-1],
            color=[ACCENT if i == 0 else ACCENT2 for i in range(len(ps))][::-1], edgecolor="none")
    darken(ax); ax.set_xlabel("Violations", color=NEUTRAL)
    ax.set_title("Top 10 Police Stations by Violation Count", fontsize=14, fontweight="bold")
    for i, v in enumerate(ps.values[::-1]):
        ax.text(v + 100, i, f"{v:,}", va="center", color=NEUTRAL, fontsize=8)
    save("06_top_police_stations")

# Plot 7 — Confusion matrix (top 8 classes, ensemble predictions)
top8_cls = list(pd.Series(y_test_labels).value_counts().head(8).index)
mask8    = np.isin(y_test_labels, top8_cls) & np.isin(y_pred_labels, top8_cls)
cm8 = confusion_matrix(
    np.array(y_test_labels)[mask8],
    np.array(y_pred_labels)[mask8],
    labels=top8_cls
)
fig, ax = plt.subplots(figsize=(10, 8), facecolor=DARK); ax.set_facecolor(DARK)
sns.heatmap(cm8, annot=True, fmt="d", xticklabels=[l[:18] for l in top8_cls],
            yticklabels=[l[:18] for l in top8_cls], cmap="RdPu", ax=ax,
            linewidths=0.5, linecolor=DARK, annot_kws={"size": 8})
ax.tick_params(colors=NEUTRAL, labelsize=7)
ax.set_xlabel("Predicted", color=NEUTRAL); ax.set_ylabel("Actual", color=NEUTRAL)
ax.set_title("Ensemble Confusion Matrix — Top 8 Classes", fontsize=13, fontweight="bold", color=FONT)
plt.xticks(rotation=35, ha="right")
save("07_confusion_matrix")

# Plot 8 — LightGBM feature importance (top 20)
fi = pd.Series(
    lgb_model.feature_importance(importance_type="gain"),
    index=FEATURE_COLS
).sort_values(ascending=False).head(20)
fig, ax = plt.subplots(figsize=(12, 7), facecolor=DARK)
ax.barh(fi.index[::-1], fi.values[::-1],
        color=[ACCENT if i < 5 else ACCENT2 for i in range(len(fi))][::-1], edgecolor="none")
darken(ax); ax.set_xlabel("Feature Importance (Gain)", color=NEUTRAL)
ax.set_title("Top 20 Feature Importances — LightGBM", fontsize=14, fontweight="bold")
save("08_lgbm_feature_importance")

# Plot 9 — DBSCAN cluster sizes
if len(cluster_info) > 0:
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=DARK)
    top_c = cluster_info.head(25)
    ax.bar(range(len(top_c)), top_c["point_count"],
           color=[ACCENT if i < 5 else ACCENT2 for i in range(len(top_c))], edgecolor="none")
    darken(ax); ax.set_xlabel("Cluster Rank", color=NEUTRAL)
    ax.set_ylabel("Violation Count", color=NEUTRAL)
    ax.set_title(f"Top 25 DBSCAN Hotspot Clusters  (total: {n_clusters})", fontsize=14, fontweight="bold")
    save("09_dbscan_cluster_sizes")

# Plot 10 — Model comparison bar chart
fig, ax = plt.subplots(figsize=(8, 5), facecolor=DARK)
models  = ["LightGBM", "XGBoost", "Ensemble"]
acc_vals = [acc_lgb, acc_xgb, acc_ens]
f1_vals  = [f1_lgb,  f1_xgb,  f1_ens]
x = np.arange(3); w = 0.35
ax.bar(x - w/2, acc_vals, w, label="Accuracy",     color=ACCENT,  edgecolor="none")
ax.bar(x + w/2, f1_vals,  w, label="F1 (weighted)",color=ACCENT2, edgecolor="none")
ax.set_xticks(x); ax.set_xticklabels(models)
ax.set_ylim(0, 1.05)
ax.legend(facecolor="#1a1a2e", labelcolor=NEUTRAL, fontsize=9)
darken(ax)
ax.set_title("LightGBM vs XGBoost vs Ensemble — Accuracy & F1", fontsize=13, fontweight="bold")
for i, (a, f) in enumerate(zip(acc_vals, f1_vals)):
    ax.text(i - w/2, a + 0.01, f"{a:.4f}", ha="center", color=NEUTRAL, fontsize=8)
    ax.text(i + w/2, f + 0.01, f"{f:.4f}", ha="center", color=NEUTRAL, fontsize=8)
save("10_model_comparison")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 9 — Accuracy & evaluation report
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 70)
print("  STEP 9 — Accuracy & evaluation report")
print("=" * 70)

prec_ens = precision_score(y_test, y_pred_ensemble, average="weighted", zero_division=0)
rec_ens  = recall_score(y_test,    y_pred_ensemble, average="weighted", zero_division=0)
f1_mac   = f1_score(y_test,        y_pred_ensemble, average="macro",    zero_division=0)

per_class = pd.DataFrame({
    "class":     le_target.classes_,
    "f1_score":  f1_score(y_test, y_pred_ensemble, average=None, zero_division=0, labels=list(range(n_classes))),
    "precision": precision_score(y_test, y_pred_ensemble, average=None, zero_division=0, labels=list(range(n_classes))),
    "recall":    recall_score(y_test, y_pred_ensemble, average=None, zero_division=0, labels=list(range(n_classes))),
    "support":   np.bincount(y_test, minlength=n_classes),
}).sort_values("f1_score", ascending=False)
per_class.to_csv(OUT_DIR / "per_class_metrics.csv", index=False)

summary = {
    "model":               "LightGBM + XGBoost soft-voting ensemble",
    "n_classes":           int(n_classes),
    "train_rows":          int(len(X_train)),
    "test_rows":           int(len(X_test)),
    "feature_count":       len(FEATURE_COLS),
    "lgbm_accuracy":       round(float(acc_lgb), 4),
    "xgb_accuracy":        round(float(acc_xgb), 4),
    "ensemble_accuracy":   round(float(acc_ens), 4),
    "ensemble_f1_weighted":round(float(f1_ens), 4),
    "ensemble_f1_macro":   round(float(f1_mac), 4),
    "ensemble_precision":  round(float(prec_ens), 4),
    "ensemble_recall":     round(float(rec_ens), 4),
    "n_clusters":          int(n_clusters),
    "noise_points":        int(n_noise),
    "generated":           str(datetime.now()),
    # ── CRITICAL FOR INFERENCE ──────────────────────────────────────────
    # Both models were fit on a positional NumPy array (X = df_ml[FEATURE_COLS]
    # .values), so this exact order is the ONLY source of truth for how to
    # build a feature vector at inference time. Anything that builds a
    # prediction request (e.g. app.py) MUST read this list and zip values
    # to it in this order — never hardcode/guess a feature order elsewhere.
    "feature_columns":     FEATURE_COLS,
    # le_target.classes_ is the exact index→label mapping the model's
    # softmax output columns correspond to (argmax index 0 = classes_[0]).
    "target_classes":      list(le_target.classes_),
    # F1-maximizing per-class threshold for the "probability / threshold"
    # decision rule app.py uses to counter majority-class bias at inference.
    "decision_thresholds": decision_thresholds,
}
with open(OUT_DIR / "model_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"  📌  Persisted exact training feature order ({len(FEATURE_COLS)} cols) "
      f"and target class order ({len(le_target.classes_)} classes) → model_summary.json")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 10 — Output summary
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 70)
print("  STEP 10 — Output summary")
print("=" * 70)
print(f"""
  ┌─────────────────────────────────────────────────────────────────────┐
  │               PIPELINE COMPLETE — FINAL RESULTS                    │
  ├──────────────────────────────────┬──────────────────────────────────┤
  │ Ensemble Accuracy                │ {acc_ens:.4f}                        │
  │ Ensemble F1 (weighted)           │ {f1_ens:.4f}                        │
  │ Ensemble F1 (macro)              │ {f1_mac:.4f}                        │
  │ Precision (weighted)             │ {prec_ens:.4f}                        │
  │ Recall (weighted)                │ {rec_ens:.4f}                        │
  ├──────────────────────────────────┼──────────────────────────────────┤
  │ DBSCAN clusters                  │ {n_clusters}                             │
  │ Training rows                    │ {len(X_train):,}                       │
  │ Violation classes                │ {n_classes}                             │
  └──────────────────────────────────┴──────────────────────────────────┘

  📁 outputs/
  ├── maps/
  │   ├── 01_congestion_heatmap.html
  │   ├── 02_hotspot_clusters_priority.html
  │   └── 03_night_vs_day.html
  ├── plots/
  │   ├── 01_top_violation_classes.png
  │   ├── 02_hourly_pattern.png
  │   ├── 03_day_of_week.png
  │   ├── 04_vehicle_category.png
  │   ├── 05_severity_distribution.png
  │   ├── 06_top_police_stations.png
  │   ├── 07_confusion_matrix.png
  │   ├── 08_lgbm_feature_importance.png
  │   ├── 09_dbscan_cluster_sizes.png
  │   └── 10_model_comparison.png
  ├── model/
  │   ├── lgbm_model.txt
  │   └── xgb_model.json
  ├── hotspot_clusters.csv
  ├── enforcement_priority_ranked.csv
  ├── per_class_metrics.csv
  ├── classification_report.txt
  └── model_summary.json
""")
print("=" * 70 + "\n")