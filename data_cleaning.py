"""
=============================================================================
  data_cleaning.py
  Flipkart GridLock 2.0 — Round 2 | PS1
  PURPOSE: Clean raw BTP data and recover NaN validation records via
           device-trust scoring, growing training pool from ~8K → ~93K rows.

  PIPELINE ORDER (leakage-free):
    Step 0  — auto-install requirements
    Step 1  — load raw CSV in memory-safe chunks
    Step 2  — drop dead columns & deduplicate
    Step 3  — device-trust scoring (NaN recovery) ← KEY FIX
    Step 4  — filter to high-quality records only
    Step 5  — basic type casting & null-fill
    Step 6  — save cleaned CSV
=============================================================================
"""

# ── Step 0: auto-install ────────────────────────────────────────────────────
import subprocess, sys, importlib

DEPS = {
    "pandas": "pandas>=2.0",
    "numpy":  "numpy>=1.24",
    "tqdm":   "tqdm",
}
for import_name, pip_spec in DEPS.items():
    try:
        importlib.import_module(import_name)
    except ImportError:
        print(f"Installing {pip_spec}...", end=" ", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", pip_spec, "-q"],
                       check=True)
        print("done")

import ast, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")
print("\n" + "=" * 65)
print("  DATA CLEANING PIPELINE — BTP Parking Violations")
print("=" * 65)

# ── PATHS — adjust BASE_DIR if needed ──────────────────────────────────────
BASE_DIR  = Path(r"D:\Flipkart Gridlock 2.0\Round-2")
DATA_DIR  = BASE_DIR / "data"
CLEAN_DIR = DATA_DIR / "cleaned"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

RAW_CSV     = DATA_DIR / "jan to may police violation_anonymized791b166.csv"
OUTPUT_CSV  = CLEAN_DIR / "dataset_cleaned.csv"
AUDIT_FILE  = CLEAN_DIR / "cleaning_audit.json"

# ── CONFIG ──────────────────────────────────────────────────────────────────
CHUNKSIZE          = 50_000
DEVICE_TRUST_FLOOR = 0.80   # NaN records from devices with ≥80% approval rate kept
RANDOM_STATE       = 42

# ── Step 1: Load raw CSV ────────────────────────────────────────────────────
print("\n[Step 1] Loading raw CSV...")

DEAD_COLS = {"description", "closed_datetime", "action_taken_timestamp"}
DTYPE_MAP = {
    "latitude":    "float32",
    "longitude":   "float32",
    "center_code": "float32",
}

def load_raw(path: Path) -> pd.DataFrame:
    total_rows = sum(1 for _ in open(path, encoding="utf-8")) - 1
    print(f"  Detected {total_rows:,} rows in raw file")
    chunks = []
    reader = pd.read_csv(
        path, chunksize=CHUNKSIZE, low_memory=False,
        dtype=DTYPE_MAP, encoding="utf-8"
    )
    for chunk in tqdm(reader, desc="  Loading", unit="chunk",
                      total=(total_rows // CHUNKSIZE + 1)):
        drop = [c for c in DEAD_COLS if c in chunk.columns]
        chunk.drop(columns=drop, inplace=True, errors="ignore")
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    print(f"  Loaded: {len(df):,} rows × {df.shape[1]} cols")
    return df

df_all = load_raw(RAW_CSV)
audit  = {"raw_rows": len(df_all)}

# ── Step 2: Drop exact duplicates ───────────────────────────────────────────
print("\n[Step 2] Deduplication...")
before = len(df_all)
# Mark duplicate violation IDs (keep first)
if "id" in df_all.columns:
    df_all = df_all.drop_duplicates(subset=["id"], keep="first")
# Also drop validation_status == "duplicate"
if "validation_status" in df_all.columns:
    df_all = df_all[df_all["validation_status"] != "duplicate"]
after = len(df_all)
print(f"  Removed {before - after:,} duplicates  →  {after:,} rows remain")
audit["after_dedup"] = after

# ── Step 3: Device-trust NaN recovery ───────────────────────────────────────
#   LEAKAGE-FREE DESIGN:
#   - Approval rates computed ONLY from records whose validation_status is
#     explicitly 'approved' or 'rejected'  (never from NaN or 'created1' rows)
#   - NaN records are passive recipients of the score; they never influence it
print("\n[Step 3] Device-trust NaN recovery...")

def step3_device_trust_scoring(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each device_id, compute approval_rate = approved / (approved + rejected).
    NaN-validation records from devices with approval_rate > DEVICE_TRUST_FLOOR
    are promoted to the training pool. All others are dropped.

    This grows the usable dataset from ~8–11K (approved only) to ~75–93K rows.
    """
    vs = df["validation_status"].fillna("__NAN__")

    approved_mask = vs == "approved"
    rejected_mask = vs == "rejected"
    nan_mask      = vs == "__NAN__"

    # Approval rate per device — computed ONLY on labelled records
    labelled = df[approved_mask | rejected_mask].copy()
    dev_appr = labelled[labelled["validation_status"] == "approved"]["device_id"].value_counts()
    dev_tot  = labelled["device_id"].value_counts()
    approval_rate = (dev_appr / dev_tot).fillna(0.0)

    # Tag every row with its device's approval rate
    df = df.copy()
    df["device_approval_rate"] = df["device_id"].map(approval_rate).fillna(0.0)

    # Trusted NaN records = NaN validation AND device approval rate ≥ floor
    high_trust_nan = nan_mask & (df["device_approval_rate"] >= DEVICE_TRUST_FLOOR)

    # Final pool:
    #   - all approved records (ground truth)
    #   - high-trust NaN records (recovered)
    #   Drop: rejected, processing, created1, duplicate, low-trust NaN
    kept = df[approved_mask | high_trust_nan].copy()

    n_approved   = approved_mask.sum()
    n_recovered  = high_trust_nan.sum()
    n_low_trust  = nan_mask.sum() - n_recovered
    n_rejected   = rejected_mask.sum()
    print(f"  Approved records        : {n_approved:>8,}")
    print(f"  NaN recovered (trusted) : {n_recovered:>8,}  (device rate ≥ {DEVICE_TRUST_FLOOR})")
    print(f"  NaN dropped (low trust) : {n_low_trust:>8,}")
    print(f"  Rejected / other dropped: {n_rejected:>8,}")
    print(f"  ─────────────────────────────────────")
    print(f"  Final training pool     : {len(kept):>8,}  rows")

    return kept

df_filtered = step3_device_trust_scoring(df_all)
audit["after_trust_filter"] = len(df_filtered)

# ── Step 4: Remove residual bad-quality records ──────────────────────────────
print("\n[Step 4] Final quality filters...")

before = len(df_filtered)

# Drop rows missing critical geo data
df_filtered = df_filtered.dropna(subset=["latitude", "longitude"])

# Drop rows with impossible lat/lon (outside Bengaluru bounding box ± margin)
LAT_MIN, LAT_MAX = 12.70, 13.40
LON_MIN, LON_MAX = 77.35, 77.85
geo_mask = (
    df_filtered["latitude"].between(LAT_MIN, LAT_MAX) &
    df_filtered["longitude"].between(LON_MIN, LON_MAX)
)
df_filtered = df_filtered[geo_mask]

# Drop rows missing created_datetime
if "created_datetime" in df_filtered.columns:
    df_filtered = df_filtered.dropna(subset=["created_datetime"])

print(f"  Dropped {before - len(df_filtered):,} geo/datetime-invalid rows")
print(f"  Remaining: {len(df_filtered):,}")
audit["after_quality_filter"] = len(df_filtered)

# ── Step 5: Type casting & null-fill ────────────────────────────────────────
print("\n[Step 5] Type casting and null-fill...")

# Parse datetimes
if "created_datetime" in df_filtered.columns:
    df_filtered["created_datetime"] = pd.to_datetime(
        df_filtered["created_datetime"], format="ISO8601", utc=True, errors="coerce"
    )

if "modified_datetime" in df_filtered.columns:
    df_filtered["modified_datetime"] = pd.to_datetime(
        df_filtered["modified_datetime"], format="ISO8601", utc=True, errors="coerce"
    )

# String columns — strip whitespace
str_cols = df_filtered.select_dtypes(include="object").columns
for col in str_cols:
    df_filtered[col] = df_filtered[col].astype(str).str.strip()

# Fill common nulls
fill_map = {
    "junction_name":  "No Junction",
    "police_station": "Unknown",
    "location":       "",
    "vehicle_type":   "OTHERS",
}
for col, val in fill_map.items():
    if col in df_filtered.columns:
        df_filtered[col] = df_filtered[col].replace({"nan": val, "None": val}).fillna(val)

# center_code: fill with per-station median
if "center_code" in df_filtered.columns and "police_station" in df_filtered.columns:
    station_median = df_filtered.groupby("police_station")["center_code"].transform("median")
    df_filtered["center_code"] = df_filtered["center_code"].fillna(station_median)

print("  ✅  Type casting complete")

# ── Step 6: Save ─────────────────────────────────────────────────────────────
print(f"\n[Step 6] Saving cleaned data → {OUTPUT_CSV.name}")
df_filtered.to_csv(OUTPUT_CSV, index=False)

audit["final_rows"]    = len(df_filtered)
audit["final_cols"]    = int(df_filtered.shape[1])
audit["device_floor"]  = DEVICE_TRUST_FLOOR
audit["geo_bounds"]    = {"lat": [LAT_MIN, LAT_MAX], "lon": [LON_MIN, LON_MAX]}

with open(AUDIT_FILE, "w") as f:
    json.dump(audit, f, indent=2, default=str)

print(f"\n{'=' * 65}")
print(f"  ✅  CLEANING COMPLETE")
print(f"  Raw rows       : {audit['raw_rows']:>10,}")
print(f"  After dedup    : {audit['after_dedup']:>10,}")
print(f"  After trust    : {audit['after_trust_filter']:>10,}  ← device-trust NaN recovery")
print(f"  Final clean    : {audit['final_rows']:>10,}")
print(f"  Saved to       : {OUTPUT_CSV}")
print(f"  Audit log      : {AUDIT_FILE}")
print("=" * 65 + "\n")
