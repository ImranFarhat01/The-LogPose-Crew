"""
=============================================================================
  violation_features.py
  Flipkart GridLock 2.0 — Round 2 | PS1
  PURPOSE: Build the final violation-label column used as the ML target.

  KEY FIX vs v1:
    ✅ Rare classes merged via EXPLICIT hardcoded set — NOT threshold < 100
       This keeps DEFECTIVE NUMBER PLATE, PARKING NEAR BUSTOP, and
       PARKING NEAR ROAD CROSSING as standalone classes (operationally distinct)
       while only merging 9 truly-rare OR behaviorally-redundant classes.
    ✅ Outputs primary_violation_final + violation_category used by pipeline
    ✅ Also produces a multi-label binary matrix for optional multi-output models
=============================================================================
"""

# ── Step 0: auto-install ────────────────────────────────────────────────────
import subprocess, sys, importlib

DEPS = {"pandas": "pandas>=2.0", "numpy": "numpy>=1.24", "tqdm": "tqdm"}
for pkg, pip_spec in DEPS.items():
    try:
        importlib.import_module(pkg)
    except ImportError:
        print(f"Installing {pip_spec}...", end=" ", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", pip_spec, "-q"], check=True)
        print("done")

import ast, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")
print("\n" + "=" * 65)
print("  VIOLATION FEATURE ENGINEERING")
print("=" * 65)

# ── PATHS ───────────────────────────────────────────────────────────────────
BASE_DIR   = Path(r"D:\Flipkart Gridlock 2.0\Round-2")
CLEAN_DIR  = BASE_DIR / "data" / "cleaned"
INPUT_CSV  = CLEAN_DIR / "dataset_features.csv"
OUTPUT_CSV = CLEAN_DIR / "dataset_features.csv"   # update in-place

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EXPLICIT RARE-CLASS MERGE SET (KEY FIX from screenshot)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# These 9 classes are merged into "OTHER_VIOLATION" because they are either:
#   (a) extremely rare (<50 records each), OR
#   (b) enforcement-context violations, not parking-congestion violations
#
# Classes KEPT as standalone (NOT in this set):
#   DEFECTIVE NUMBER PLATE        — 7,848 records, operational enforcement signal
#   PARKING NEAR BUSTOP/SCHOOL/HOSPITAL ETC — 2,403 records, high-severity zone
#   PARKING NEAR ROAD CROSSING    — 1,687 records, distinct from generic parking
#   WRONG PARKING, NO PARKING, PARKING IN A MAIN ROAD, PARKING ON FOOTPATH,
#   DOUBLE PARKING, PARKING NEAR TRAFFIC LIGHT OR ZEBRA CROSS, etc.
RARE_CLASSES_TO_MERGE = {
    "DOUBLE PARKING",                        # 2,037 — merge: operationally same as WRONG PARKING at junction
    "PARKING NEAR TRAFFIC LIGHT OR ZEBRA CROSS",  # 525 — merge: too few for stable class
    "PARKING OTHER THAN BUS STOP",           # 242 — merge
    "PARKING OPPOSITE TO ANOTHER PARKED VEHICLE",  # 486 — merge
    "H T V PROHIBITED",                      # 31  — merge
    "REFUSE TO GO FOR HIRE",                 # 887 — merge: taxi regulation, not parking
    "OBSTRUCTING DRIVER",                    # 16  — merge
    "DEMANDING EXCESS FARE",                 # 240 — merge: taxi regulation, not parking
    "FAIL TO USE SAFETY BELTS",              # 8   — merge
}

# ── Load ────────────────────────────────────────────────────────────────────
print(f"\n[1] Loading from {INPUT_CSV.name}...")
df = pd.read_csv(INPUT_CSV, low_memory=False)
print(f"  {len(df):,} rows × {df.shape[1]} cols")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Build primary_violation column if not already present
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def safe_parse_list(val):
    if pd.isna(val) or str(val).strip() in ("nan", "None", ""):
        return []
    if isinstance(val, list):
        return val
    try:
        return ast.literal_eval(str(val))
    except Exception:
        return [str(val)]

if "primary_violation" not in df.columns and "violation_type" in df.columns:
    print("[2] Parsing violation_type to extract primary_violation...")
    tqdm.pandas(desc="  Parsing")
    viol_lists = df["violation_type"].progress_apply(safe_parse_list)
    df["primary_violation"] = viol_lists.apply(
        lambda x: x[0].strip().upper() if x else "UNKNOWN"
    )
else:
    df["primary_violation"] = df["primary_violation"].astype(str).str.upper().str.strip()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# engineer_violation_features()  — apply rare-class merge
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[3] Applying explicit rare-class merge...")

def engineer_violation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    KEY FIX: Uses explicit RARE_CLASSES_TO_MERGE set instead of
    rare_violations = violation_counts[violation_counts < 100].index.tolist()
    
    Threshold logic merges DEFECTIVE NUMBER PLATE (7,848), PARKING NEAR BUSTOP
    (2,403), and PARKING NEAR ROAD CROSSING (1,687) into OTHER — all of which
    are operationally distinct enforcement signals the model should learn.
    The explicit set merges only 9 specific classes.
    """
    df = df.copy()

    before_classes = df["primary_violation"].nunique()
    print(f"  Classes before merge : {before_classes}")

    df["primary_violation_final"] = df["primary_violation"].apply(
        lambda v: "OTHER_VIOLATION" if v in RARE_CLASSES_TO_MERGE else v
    )

    after_classes = df["primary_violation_final"].nunique()
    print(f"  Classes after merge  : {after_classes}  (merged {before_classes - after_classes} into OTHER_VIOLATION)")

    # Show class distribution
    vc = df["primary_violation_final"].value_counts()
    print("\n  Final class distribution:")
    for cls, cnt in vc.items():
        bar = "█" * min(30, cnt // 2000)
        kept_tag = "  ← standalone (kept)" if cls not in RARE_CLASSES_TO_MERGE else ""
        print(f"    {cls:50s} {cnt:>7,}  {bar}{kept_tag}")

    # Violation category (broader grouping for secondary signal)
    CATEGORY_MAP = {
        "WRONG PARKING":              "ILLEGAL_PARKING",
        "NO PARKING":                 "ILLEGAL_PARKING",
        "PARKING IN A MAIN ROAD":     "ROAD_OBSTRUCTION",
        "PARKING ON FOOTPATH":        "FOOTPATH_OBSTRUCTION",
        "DEFECTIVE NUMBER PLATE":     "VEHICLE_COMPLIANCE",
        "PARKING NEAR ROAD CROSSING": "JUNCTION_HAZARD",
        "PARKING NEAR BUSTOP/SCHOOL/HOSPITAL ETC": "PROTECTED_ZONE",
        "PARKING NEAR TRAFFIC LIGHT OR ZEBRA CROSS": "JUNCTION_HAZARD",
        "OTHER_VIOLATION":            "OTHER",
        "UNKNOWN":                    "OTHER",
    }
    df["violation_category"] = df["primary_violation_final"].map(
        CATEGORY_MAP
    ).fillna("ILLEGAL_PARKING")

    return df

df = engineer_violation_features(df)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Multi-label binary matrix — optional secondary model signal
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[4] Building multi-label binary matrix...")

STANDALONE_CLASSES = [
    "WRONG PARKING",
    "NO PARKING",
    "PARKING IN A MAIN ROAD",
    "DEFECTIVE NUMBER PLATE",
    "PARKING ON FOOTPATH",
    "PARKING NEAR BUSTOP/SCHOOL/HOSPITAL ETC",
    "PARKING NEAR ROAD CROSSING",
    "OTHER_VIOLATION",
]

if "violation_type" in df.columns:
    vt_str = df["violation_type"].astype(str).str.upper()
    for cls in STANDALONE_CLASSES:
        col_name = "ml_" + cls.lower().replace(" ", "_").replace("/", "_").replace(",", "")
        df[col_name] = vt_str.str.contains(cls, na=False, regex=False).astype("int8")
    print(f"  Built {len(STANDALONE_CLASSES)} binary label columns (ml_*)")

# ── Save ────────────────────────────────────────────────────────────────────
print(f"\n[5] Saving → {OUTPUT_CSV.name}")
df.to_csv(OUTPUT_CSV, index=False)

print(f"\n{'=' * 65}")
print(f"  ✅  VIOLATION FEATURES COMPLETE")
print(f"  Rows saved          : {len(df):,}")
print(f"  Target column       : primary_violation_final")
print(f"  Category column     : violation_category")
print(f"  Merged into OTHER   : {len(RARE_CLASSES_TO_MERGE)} classes")
print(f"  Kept standalone     : {df['primary_violation_final'].nunique()} classes")
print(f"  Saved to            : {OUTPUT_CSV}")
print("=" * 65 + "\n")
