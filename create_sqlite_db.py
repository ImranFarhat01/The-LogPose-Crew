import sqlite3
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
import os

BASE_DIR = Path(__file__).parent
CLEAN_DIR = BASE_DIR / "data" / "cleaned"
DASH_DIR = BASE_DIR / "outputs" / "dashboard_data"
OUT_DIR = BASE_DIR / "outputs"
DB_PATH = BASE_DIR / "btp_database.db"

def build_sqlite_db():
    print("Creating local SQLite database...")
    
    # Remove old DB if exists
    if DB_PATH.exists():
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create table for JSON key-value store
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS json_store (
            key TEXT PRIMARY KEY,
            data TEXT
        )
    ''')

    files_to_push = {
        "dataset_features": CLEAN_DIR / "dataset_features.csv",
        "model_summary": OUT_DIR / "model_summary.json",
        "enforcement_priority_ranked": OUT_DIR / "enforcement_priority_ranked.csv",
        "hotspot_clusters": OUT_DIR / "hotspot_clusters.csv",
        "peak_time_forecast": DASH_DIR / "peak_time_forecast.csv",
        "global_peak_hours": DASH_DIR / "global_peak_hours.csv",
        "hour_vs_violation_matrix": DASH_DIR / "hour_vs_violation_matrix.csv",
        "patrol_gap_analysis": DASH_DIR / "patrol_gap_analysis.csv",
        "reactive_vs_proactive": DASH_DIR / "reactive_vs_proactive.json",
        "anomaly_detection": DASH_DIR / "anomaly_detection.json",
        "scita_sync": DASH_DIR / "scita_sync.json",
        "vehicle_lookup_index": DASH_DIR / "vehicle_lookup_index.csv",
        "multi_violation_profiles": DASH_DIR / "multi_violation_profiles.csv",
        "multi_violation_summary": DASH_DIR / "multi_violation_summary.json",
        "vehicle_vs_violation_matrix": DASH_DIR / "vehicle_vs_violation_matrix.csv",
        "geohash_grid_overlay": DASH_DIR / "geohash_grid_overlay.csv",
        "recommendations": DASH_DIR / "recommendations.json",
        "time_block_shifts": DASH_DIR / "time_block_shifts.csv",
        "habitual_offenders": DASH_DIR / "habitual_offenders.csv",
        "station_reference": DASH_DIR / "station_reference.csv",
        "junction_reference": DASH_DIR / "junction_reference.csv",
        "quick_view_presets": DASH_DIR / "quick_view_presets.json",
        "day_of_week_trends": DASH_DIR / "day_of_week_trends.csv",
        "validation_status": DASH_DIR / "validation_status.json",
        "offence_filter_reference": DASH_DIR / "offence_filter_reference.csv",
        "live_prediction_config": DASH_DIR / "live_prediction_config.json"
    }

    for coll_name, filepath in tqdm(files_to_push.items(), desc="Building Database"):
        if not filepath.exists():
            tqdm.write(f"Skipping {coll_name} - not found.")
            continue

        if filepath.suffix == '.csv':
            try:
                df = pd.read_csv(filepath, low_memory=False)
                df.to_sql(coll_name, conn, if_exists='replace', index=False)
                tqdm.write(f"OK: Added table: {coll_name}")
            except Exception as e:
                tqdm.write(f"Error with CSV {coll_name}: {e}")

        elif filepath.suffix == '.json':
            try:
                with open(filepath, 'r') as f:
                    data_str = f.read()
                cursor.execute("INSERT OR REPLACE INTO json_store (key, data) VALUES (?, ?)", (coll_name, data_str))
                tqdm.write(f"OK: Added JSON: {coll_name}")
            except Exception as e:
                tqdm.write(f"Error with JSON {coll_name}: {e}")

    conn.commit()
    conn.close()
    
    db_size_mb = os.path.getsize(DB_PATH) / (1024 * 1024)
    print(f"\n🚀 Successfully created SQLite Database: btp_database.db ({db_size_mb:.2f} MB)")

if __name__ == "__main__":
    build_sqlite_db()
