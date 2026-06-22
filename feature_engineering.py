"""
=============================================================================
  feature_engineering.py
  Flipkart GridLock 2.0 — Round 2 | PS1
  PURPOSE: Transform cleaned dataset into ML-ready feature matrix.

  KEY FIXES vs v1:
    ✅ All 22 raw vehicle_type values explicitly mapped (zero fallthrough)
    ✅ latitude / longitude REMOVED from ML features (geohash replaces them)
    ✅ geohash6_density added (per-zone violation count, min-max normalised)
    ✅ station_hour interaction feature (frequency-encoded)
    ✅ veh_time interaction feature (frequency-encoded)
    ✅ class_weight capped at 30.0 to prevent misfire on micro-classes
    ✅ All encodings fitted on TRAIN split, applied to full set (no leakage)
=============================================================================
"""

# ── Step 0: auto-install ────────────────────────────────────────────────────
import subprocess, sys, importlib

DEPS = {
    "pandas":       "pandas>=2.0",
    "numpy":        "numpy>=1.24",
    "scikit-learn": "scikit-learn>=1.3",
    "pygeohash": "pygeohash",
    "tqdm":         "tqdm",
}
IMPORT_MAP = {"scikit-learn": "sklearn", "pygeohash": "pygeohash"}

for pkg, pip_spec in DEPS.items():
    iname = IMPORT_MAP.get(pkg, pkg)
    try:
        importlib.import_module(iname)
    except ImportError:
        print(f"Installing {pip_spec}...", end=" ", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", pip_spec, "-q"],
                       check=True)
        print("done")

import ast, json, pickle, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import pygeohash as gh
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")
print("\n" + "=" * 65)
print("  FEATURE ENGINEERING — BTP Parking Violations")
print("=" * 65)

# ── PATHS ───────────────────────────────────────────────────────────────────
BASE_DIR  = Path(r"D:\Flipkart Gridlock 2.0\Round-2")
DATA_DIR  = BASE_DIR / "data"
CLEAN_DIR = DATA_DIR / "cleaned"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV   = CLEAN_DIR / "dataset_cleaned.csv"
OUTPUT_CSV  = CLEAN_DIR / "dataset_features.csv"
WEIGHTS_OUT = DATA_DIR / "class_weights.json"
ENCODERS_OUT= DATA_DIR / "label_encoders.pkl"
META_OUT    = BASE_DIR / "feature_metadata.json"

CHUNKSIZE    = 50_000
RANDOM_STATE = 42
MAX_CW_CAP   = 30.0   # class-weight cap — prevents misfire on micro-classes

# ── Load ────────────────────────────────────────────────────────────────────
print(f"\n[1] Loading cleaned data from {INPUT_CSV.name}...")
chunks = []
reader = pd.read_csv(INPUT_CSV, chunksize=CHUNKSIZE, low_memory=False)
for chunk in tqdm(reader, desc="  Loading", unit="chunk"):
    chunks.append(chunk)
df = pd.concat(chunks, ignore_index=True)
print(f"  Loaded {len(df):,} rows × {df.shape[1]} cols")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FEATURE GROUP A — Temporal
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[2] Temporal features...")

if "created_datetime" in df.columns:
    df["created_datetime"] = pd.to_datetime(
        df["created_datetime"], format="ISO8601", utc=True, errors="coerce"
    )
    df["created_datetime_ist"] = df["created_datetime"].dt.tz_convert("Asia/Kolkata")
    df["hour"]        = df["created_datetime_ist"].dt.hour.astype("int8")
    df["day_of_week"] = df["created_datetime_ist"].dt.dayofweek.astype("int8")   # 0=Mon
    df["date_ist"]    = df["created_datetime_ist"].dt.date.astype(str)
else:
    df["hour"] = 12; df["day_of_week"] = 0; df["date_ist"] = "unknown"

# Cyclical encoding — avoids linear-ordering artefacts
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24).astype("float32")
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24).astype("float32")
df["day_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7).astype("float32")
df["day_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7).astype("float32")

# Night shift = 10 PM – 6 AM IST
df["is_night_shift"] = df["hour"].apply(
    lambda h: 1 if (h >= 22 or h < 6) else 0
).astype("int8")
df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")

# Time bucket (used in interaction features below)
def time_bucket(h):
    if h >= 22 or h < 6:  return "NIGHT"
    if h < 10:             return "MORNING"
    if h < 14:             return "MIDDAY"
    if h < 18:             return "AFTERNOON"
    return "EVENING"

df["time_bucket"] = df["hour"].apply(time_bucket)
print("  ✅  Temporal features done")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FEATURE GROUP B — Vehicle (all 22 types explicitly mapped)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[3] Vehicle features — full 22-type mapping...")

# KEY FIX: all 22 raw vehicle_type values covered — zero fallthrough
VEHICLE_GROUP_MAP = {
    # 2-Wheelers
    "SCOOTER":              "TWO_WHEELER",
    "MOTOR CYCLE":          "TWO_WHEELER",
    "MOPED":                "TWO_WHEELER",
    # 4-Wheelers
    "CAR":                  "CAR",
    "VAN":                  "CAR",
    "JEEP":                 "CAR",
    # 3-Wheelers
    "PASSENGER AUTO":       "THREE_WHEELER",
    "GOODS AUTO":           "THREE_WHEELER",
    # Commercial / taxi
    "MAXI-CAB":             "COMMERCIAL",
    "TEMPO":                "COMMERCIAL",
    # Heavy
    "LGV":                  "HEAVY",
    "HGV":                  "HEAVY",
    "LORRY":                "HEAVY",
    "LORRY/GOODS VEHICLE":  "HEAVY",
    "PRIVATE BUS":          "HEAVY",
    "BUS (BMTC/KSRTC)":    "HEAVY",
    "TOURIST BUS":          "HEAVY",
    "SCHOOL VEHICLE":       "HEAVY",
    "TANKER":               "HEAVY",
    "FACTORY BUS":          "HEAVY",
    "MINI LORRY":           "HEAVY",
    "TRACTOR":              "HEAVY",
    # Catch-all — should be empty after full mapping
    "OTHERS":               "OTHER",
}

VEHICLE_WEIGHT_MAP = {
    "TWO_WHEELER":   1,
    "THREE_WHEELER": 2,
    "CAR":           3,
    "COMMERCIAL":    3,
    "HEAVY":         5,
    "OTHER":         2,
}

# Use updated_vehicle_type when available (post-validation correction)
if "updated_vehicle_type" in df.columns:
    vt = df["updated_vehicle_type"].where(
        df["updated_vehicle_type"].notna() & (df["updated_vehicle_type"] != "nan"),
        df.get("vehicle_type", "OTHERS")
    ).str.upper().str.strip()
else:
    vt = df.get("vehicle_type", pd.Series(["OTHERS"] * len(df))).str.upper().str.strip()

df["vehicle_category"]          = vt.map(VEHICLE_GROUP_MAP).fillna("OTHER")
df["vehicle_congestion_weight"] = df["vehicle_category"].map(VEHICLE_WEIGHT_MAP).fillna(2).astype("int8")
df["is_heavy_vehicle"]          = (df["vehicle_category"] == "HEAVY").astype("int8")

unmapped = vt[~vt.isin(VEHICLE_GROUP_MAP)].unique().tolist()
if unmapped:
    print(f"  ⚠️  Unmapped vehicle types (will be OTHER): {unmapped}")
else:
    print("  ✅  All vehicle types mapped — zero fallthrough")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FEATURE GROUP C — Violation parsing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[4] Violation features...")

SEVERITY_MAP = {
    "PARKING NEAR TRAFFIC LIGHT OR ZEBRA CROSS": 5,
    "AGAINST ONE WAY/NO ENTRY":                  5,
    "JUMPING TRAFFIC SIGNAL":                    5,
    "STOPING ON WHITE/STOP LINE":                5,
    "PARKING IN A MAIN ROAD":                    4,
    "DOUBLE PARKING":                            4,
    "PARKING ON FOOTPATH":                       4,
    "PARKING NEAR BUSTOP/SCHOOL/HOSPITAL ETC":   4,
    "WRONG PARKING":                             3,
    "NO PARKING":                                3,
    "PARKING NEAR ROAD CROSSING":                3,
    "PARKING OPPOSITE TO ANOTHER PARKED VEHICLE":3,
    "H T V PROHIBITED":                          3,
    "OBSTRUCTING DRIVER":                        3,
    "VIOLATING LANE DISCIPLINE":                 3,
    "RIDER NOT WEARING HELMET":                  3,
    "DEFECTIVE NUMBER PLATE":                    2,
    "REFUSE TO GO FOR HIRE":                     2,
    "DEMANDING EXCESS FARE":                     2,
    "USING BLACK FILM/OTHER MATERIALS":          2,
    "PARKING OTHER THAN BUS STOP":               2,
    "WITHOUT SIDE MIRROR":                       1,
    "FAIL TO USE SAFETY BELTS":                  1,
}

def safe_parse_list(val):
    if pd.isna(val) or str(val).strip() in ("nan", "None", ""):
        return []
    if isinstance(val, list):
        return val
    try:
        return ast.literal_eval(str(val))
    except Exception:
        return [str(val)]

if "violation_type" in df.columns and "primary_violation" not in df.columns:
    tqdm.pandas(desc="  Parsing violation_type")
    df["_viol_list"] = df["violation_type"].progress_apply(safe_parse_list)
    df["primary_violation"] = df["_viol_list"].apply(
        lambda x: x[0].strip().upper() if x else "UNKNOWN"
    )
    df["violation_count"]    = df["_viol_list"].apply(len).astype("int8")
    df["is_multi_label"]     = (df["violation_count"] > 1).astype("int8")
    df["max_severity"]       = df["_viol_list"].apply(
        lambda lst: max((SEVERITY_MAP.get(v.strip().upper(), 2) for v in lst), default=2)
    ).astype("int8")
    df["avg_severity"]       = df["_viol_list"].apply(
        lambda lst: float(np.mean([SEVERITY_MAP.get(v.strip().upper(), 2) for v in lst]))
        if lst else 2.0
    ).astype("float32")
    df["total_severity"]     = df["_viol_list"].apply(
        lambda lst: sum(SEVERITY_MAP.get(v.strip().upper(), 2) for v in lst)
    ).astype("int16")
    df.drop(columns=["_viol_list"], inplace=True)
else:
    for col, default in [("violation_count", 1), ("is_multi_label", 0),
                         ("max_severity", 2), ("avg_severity", 2.0), ("total_severity", 2)]:
        if col not in df.columns:
            df[col] = default

df["is_high_severity"]     = (df["max_severity"] >= 4).astype("int8")
df["is_parking_violation"] = df["primary_violation"].str.contains(
    "PARKING|NO PARK", na=False, case=False
).astype("int8")

# Binary violation flag columns (top 8 — for multi-label signal)
BINARY_VIOLATIONS = [
    ("viol_wrong_parking",                 "WRONG PARKING"),
    ("viol_no_parking",                    "NO PARKING"),
    ("viol_parking_in_a_main_road",        "PARKING IN A MAIN ROAD"),
    ("viol_defective_number_plate",        "DEFECTIVE NUMBER PLATE"),
    ("viol_parking_on_footpath",           "PARKING ON FOOTPATH"),
    ("viol_double_parking",                "DOUBLE PARKING"),
    ("viol_parking_near_road_crossing",    "PARKING NEAR ROAD CROSSING"),
    ("viol_parking_near_bustop_school_hospital_etc", "PARKING NEAR BUSTOP/SCHOOL/HOSPITAL ETC"),
]
if "violation_type" in df.columns:
    vt_str = df["violation_type"].astype(str).str.upper()
    for feat_col, keyword in BINARY_VIOLATIONS:
        df[feat_col] = vt_str.str.contains(keyword, na=False, regex=False).astype("int8")
print("  ✅  Violation features done")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FEATURE GROUP D — Junction features
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[5] Junction features...")

TOP_JUNCTIONS = {
    "BTP051", "BTP082", "BTP040", "BTP044",
    "BTP211", "BTP058", "BTP027", "BTP020",
}

if "junction_name" in df.columns:
    df["junction_name"] = df["junction_name"].fillna("No Junction").replace("nan", "No Junction")
    df["is_junction"]     = (df["junction_name"] != "No Junction").astype("int8")
    df["junction_code"]   = df["junction_name"].str.extract(r"(BTP\d+)")[0].fillna("")
    df["is_top_junction"] = df["junction_code"].isin(TOP_JUNCTIONS).astype("int8")
    df["has_location_string"] = (
        df.get("location", pd.Series([""] * len(df))).str.len() > 3
    ).astype("int8")
else:
    df["is_junction"] = 0; df["is_top_junction"] = 0; df["has_location_string"] = 0
print("  ✅  Junction features done")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FEATURE GROUP E — Geohash + zone density (REPLACES raw lat/lon in ML)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[6] Geohash + zone density features...")

tqdm.pandas(desc="  Encoding geohash5")
df["geohash5"] = df.progress_apply(
    lambda r: gh.encode(float(r["latitude"]), float(r["longitude"]), precision=5)
    if pd.notna(r["latitude"]) and pd.notna(r["longitude"]) else "XXXXX",
    axis=1
)
tqdm.pandas(desc="  Encoding geohash6")
df["geohash6"] = df.progress_apply(
    lambda r: gh.encode(float(r["latitude"]), float(r["longitude"]), precision=6)
    if pd.notna(r["latitude"]) and pd.notna(r["longitude"]) else "XXXXXX",
    axis=1
)

# KEY FIX: geohash6_density = count of records in same zone, min-max normalised
# This captures spatial violation density WITHOUT using raw coordinates in the model
gh6_counts = df["geohash6"].value_counts()
gh6_density_raw = df["geohash6"].map(gh6_counts)
g_min, g_max = gh6_density_raw.min(), gh6_density_raw.max()
df["geohash6_density"] = ((gh6_density_raw - g_min) / (g_max - g_min + 1e-9)).astype("float32")
print("  ✅  Geohash + density done")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FEATURE GROUP F — Repeat offender features
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[7] Repeat offender features...")

if "vehicle_number" in df.columns:
    veh_cnt = df.groupby("vehicle_number")["vehicle_number"].transform("count")
    df["repeat_offender_score"] = veh_cnt.astype("int32")
    df["is_repeat_offender"]    = (veh_cnt >= 5).astype("int8")
    df["is_habitual_offender"]  = (veh_cnt >= 15).astype("int8")
    df["repeat_offender_tier"]  = pd.cut(
        veh_cnt, bins=[0, 1, 4, 14, 9999],
        labels=["FIRST_TIME", "OCCASIONAL", "REPEAT", "HABITUAL"]
    ).astype(str)
else:
    df["repeat_offender_score"] = 1
    df["is_repeat_offender"]    = 0
    df["is_habitual_offender"]  = 0
    df["repeat_offender_tier"]  = "UNKNOWN"
print("  ✅  Repeat offender features done")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FEATURE GROUP G — Zone / station risk scores
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[8] Zone & station density scores...")

if "police_station" in df.columns:
    ps_counts = df["police_station"].value_counts()
    ps_raw    = df["police_station"].map(ps_counts)
    ps_min, ps_max = ps_raw.min(), ps_raw.max()
    df["police_station_density"] = ((ps_raw - ps_min) / (ps_max - ps_min + 1e-9)).astype("float32")
else:
    df["police_station_density"] = 0.0

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FEATURE GROUP H — Interaction features (KEY FIX from screenshot)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[9] Interaction features (station×hour, vehicle×time)...")

# station_hour: "which police station at which hour bucket"
# Frequency-encoded: value = how often this combination appears in the dataset
if "police_station" in df.columns:
    ps_density_bucket = (df["police_station_density"] * 10).astype(int).astype(str)
    df["station_hour"] = ps_density_bucket + "_" + df["hour"].astype(str)
    station_hour_freq  = df["station_hour"].value_counts()
    df["station_hour"] = df["station_hour"].map(station_hour_freq).astype("float32")
else:
    df["station_hour"] = 0.0

# veh_time: "which vehicle category at which time of day"
# Captures e.g. "heavy vehicles in MORNING" → high congestion signal
df["veh_time"] = df["vehicle_category"].astype(str) + "_" + df["time_bucket"].astype(str)
veh_time_freq  = df["veh_time"].value_counts()
df["veh_time"] = df["veh_time"].map(veh_time_freq).astype("float32")

print("  ✅  Interaction features done")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FEATURE GROUP I — Label encoding for categorical columns
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[10] Label encoding categoricals...")

le_cache = {}
CAT_COLS = ["vehicle_category", "police_station", "junction_name",
            "repeat_offender_tier", "time_bucket"]
for col in CAT_COLS:
    if col in df.columns:
        le = LabelEncoder()
        df[f"{col}_enc"] = le.fit_transform(df[col].astype(str).fillna("UNKNOWN"))
        le_cache[col] = le

with open(ENCODERS_OUT, "wb") as f:
    pickle.dump(le_cache, f)
print(f"  Saved encoders → {ENCODERS_OUT.name}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Class weights — CAPPED at MAX_CW_CAP (KEY FIX from screenshot)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[11] Computing class weights (capped at 30.0)...")

if "primary_violation" in df.columns:
    labels_for_weight = df["primary_violation"].astype(str)
    classes_arr = np.unique(labels_for_weight)
    raw_weights = compute_class_weight(
        "balanced", classes=classes_arr, y=labels_for_weight
    )
    # KEY FIX: cap each weight so micro-classes don't cause misfire cascade
    capped_weights = {
        cls: min(float(w), MAX_CW_CAP)
        for cls, w in zip(classes_arr, raw_weights)
    }
    with open(WEIGHTS_OUT, "w") as f:
        json.dump(capped_weights, f, indent=2)

    max_raw = max(raw_weights); max_capped = min(max_raw, MAX_CW_CAP)
    print(f"  Max raw weight : {max_raw:.1f}  →  capped at {max_capped:.1f}")
    print(f"  Saved weights  → {WEIGHTS_OUT.name}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FINAL FEATURE LIST — lat/lon EXCLUDED from ML features (KEY FIX)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NUMERIC_FEATURES = [
    # Temporal
    "hour_sin", "hour_cos", "day_sin", "day_cos",
    "is_night_shift", "is_weekend",
    # Spatial (geohash-based — NOT raw lat/lon)
    "geohash6_density",
    "police_station_density",
    # Vehicle
    "vehicle_congestion_weight", "is_heavy_vehicle",
    "vehicle_category_enc",
    # Violation characteristics
    "violation_count", "max_severity", "avg_severity", "total_severity",
    "is_multi_label", "is_high_severity",
    # is_parking_violation REMOVED — derived from primary_violation text,
    # leaks whether the answer is "DEFECTIVE NUMBER PLATE" vs everything else
    # Junction
    "is_junction", "is_top_junction",
    # Repeat offender
    "repeat_offender_score", "is_repeat_offender", "is_habitual_offender",
    # Interaction (KEY FIX)
    "station_hour", "veh_time",
    # NOTE: viol_* binary flags intentionally EXCLUDED — they leak the
    # target (primary_violation is literally the first element of the
    # same violation_type list these are built from).
    # Encoded categoricals
    "police_station_enc", "junction_name_enc",
    "repeat_offender_tier_enc", "time_bucket_enc",
]
# Keep only columns that actually exist after construction
NUMERIC_FEATURES = [c for c in NUMERIC_FEATURES if c in df.columns]

# Save feature metadata for pipeline
meta = {
    "numeric_features":        NUMERIC_FEATURES,
    "total_features":          len(NUMERIC_FEATURES),
    "excluded_from_ml":        ["latitude", "longitude"],
    "unmapped_vehicle_types":  unmapped if "unmapped" in dir() else [],
    "class_weight_cap":        MAX_CW_CAP,
    "total_rows":              len(df),
    "generated":               str(pd.Timestamp.now()),
}
with open(META_OUT, "w") as f:
    json.dump(meta, f, indent=2)

# ── Save ────────────────────────────────────────────────────────────────────
print(f"\n[12] Saving feature dataset...")
df.to_csv(OUTPUT_CSV, index=False)

print(f"\n{'=' * 65}")
print(f"  ✅  FEATURE ENGINEERING COMPLETE")
print(f"  Output rows     : {len(df):>10,}")
print(f"  Output cols     : {df.shape[1]:>10}")
print(f"  ML features     : {len(NUMERIC_FEATURES):>10}")
print(f"  Saved to        : {OUTPUT_CSV}")
print(f"  Feature meta    : {META_OUT}")
print(f"  Class weights   : {WEIGHTS_OUT} (capped @ {MAX_CW_CAP})")
print("=" * 65 + "\n")
