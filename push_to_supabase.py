import os
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine
from tqdm import tqdm

# Load .env variables
load_dotenv(override=True)
SUPABASE_URI = os.getenv("SUPABASE_URI", "")

BASE_DIR = Path(__file__).parent
CLEAN_DIR = BASE_DIR / "data" / "cleaned"
DASH_DIR = BASE_DIR / "outputs" / "dashboard_data"
OUT_DIR = BASE_DIR / "outputs"

def push_to_supabase():
    if not SUPABASE_URI or "user:password" in SUPABASE_URI:
        print("ERROR: Please set your actual SUPABASE_URI in the .env file before running this script.")
        print("Example: postgresql://postgres:YOUR_PASSWORD@db.YOUR_ID.supabase.co:5432/postgres")
        return

    print(f"Connecting to Supabase (PostgreSQL)...")
    try:
        engine = create_engine(SUPABASE_URI)
        connection = engine.connect()
        print("Connected successfully!")
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    # Create table for JSON key-value store if not exists
    try:
        connection.execute(engine.execute("CREATE TABLE IF NOT EXISTS json_store (key TEXT PRIMARY KEY, data TEXT)"))
    except Exception:
        # SQLAlchemy 2.0 style
        from sqlalchemy import text
        connection.execute(text("CREATE TABLE IF NOT EXISTS json_store (key TEXT PRIMARY KEY, data TEXT)"))
        connection.commit()

    # Define the mapping of collection names to local file paths
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

    total_files = len(files_to_push)
    print(f"\nStarting upload of {total_files} tables...\n")

    for coll_name, filepath in tqdm(files_to_push.items(), desc="Overall Progress"):
        if not filepath.exists():
            tqdm.write(f"Skipping {coll_name} - File not found: {filepath.name}")
            continue

        if filepath.suffix == '.csv':
            try:
                # Count lines for progress bar
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    total_rows = sum(1 for _ in f) - 1
                
                chunk_size = 10000
                total_chunks = (total_rows // chunk_size) + 1
                
                # Replace table first, then append chunks
                first_chunk = True
                
                for chunk in tqdm(pd.read_csv(filepath, chunksize=chunk_size, low_memory=False), 
                                  total=total_chunks, desc=f"Pushing {coll_name}", leave=False):
                    
                    if first_chunk:
                        chunk.to_sql(coll_name, engine, if_exists='replace', index=False)
                        first_chunk = False
                    else:
                        chunk.to_sql(coll_name, engine, if_exists='append', index=False)
                        
                tqdm.write(f"OK: Inserted {total_rows} rows into {coll_name}")
            except Exception as e:
                tqdm.write(f"Error pushing CSV {coll_name}: {e}")

        elif filepath.suffix == '.json':
            try:
                with open(filepath, 'r') as f:
                    data_str = f.read()
                
                from sqlalchemy import text
                connection.execute(
                    text("INSERT INTO json_store (key, data) VALUES (:key, :data) ON CONFLICT (key) DO UPDATE SET data = EXCLUDED.data"),
                    {"key": coll_name, "data": data_str}
                )
                connection.commit()
                tqdm.write(f"OK: Inserted JSON into {coll_name}")
            except Exception as e:
                tqdm.write(f"Error pushing JSON {coll_name}: {e}")

    connection.close()
    print("\nAll available data has been pushed to Supabase successfully!")

if __name__ == "__main__":
    push_to_supabase()
