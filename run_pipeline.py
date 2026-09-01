import os
import sys
import pandas as pd
from src.data_loader import load_all_json_data
from src.preprocessing import preprocess_and_validate

def run():
    print("=" * 60)
    print("   CRICSHEET T20 DATA INGESTION & PREPROCESSING PIPELINE   ")
    print("=" * 60)

    raw_dir = os.path.join("data", "raw", "t20s_male_json")
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    print(f"\n1. Ingesting raw JSON data from: {raw_dir}")
    matches_list, deliveries_list, parse_report = load_all_json_data(raw_dir)

    print("\n2. Cleaning, filtering, and validating datasets...")
    df_matches, df_deliveries, metrics = preprocess_and_validate(matches_list, deliveries_list, parse_report)

    # Export datasets
    matches_csv_path = os.path.join(processed_dir, "matches.csv")
    deliveries_csv_path = os.path.join(processed_dir, "deliveries.csv")

    df_matches.to_csv(matches_csv_path, index=False)
    df_deliveries.to_csv(deliveries_csv_path, index=False)

    print(f"\nSaved processed datasets:")
    print(f"  - Matches CSV:    {matches_csv_path} ({len(df_matches)} rows)")
    print(f"  - Deliveries CSV: {deliveries_csv_path} ({len(df_deliveries)} rows)")

    # Print Summary Report
    print("\n" + "=" * 60)
    print("                PIPELINE SUMMARY REPORT                ")
    print("=" * 60)
    print(f"JSON files found:                  {metrics['json_files_found']}")
    print(f"Successfully parsed files:         {metrics['successfully_parsed']}")
    print(f"Failed files:                      {metrics['failed_files_count']}")
    if metrics['failed_files_count'] > 0:
        print(f"  Failed files list: {metrics['failed_files_list']}")
    print(f"Number of completed matches:       {metrics['num_completed_matches']}")
    print(f"Number of deliveries:              {metrics['num_deliveries']}")
    print(f"Number of unique teams:            {metrics['num_unique_teams']}")
    print(f"Date range:                        {metrics['date_range']}")
    print(f"Number of no-result/abandoned:     {metrics['num_no_result_abandoned']}")
    print(f"Number of ties:                    {metrics['num_ties']}")
    print(f"Number of super overs:             {metrics['num_super_overs']}")
    print(f"Number of DLS/revised-target:      {metrics['num_dls_matches']}")

    print("\nTop 20 Teams by Match Count:")
    for rank, (team, count) in enumerate(metrics['top_20_teams'].items(), 1):
        print(f"  {rank:2d}. {team:<25} : {count} matches")

    print("\nMissing-Value Summary (matches.csv):")
    for col, null_count in metrics['matches_missing_summary'].items():
        print(f"  - {col:<20} : {null_count} missing")

    print("\nMissing-Value Summary (deliveries.csv):")
    for col, null_count in metrics['deliveries_missing_summary'].items():
        print(f"  - {col:<20} : {null_count} missing")

    print("\n" + "=" * 60)
    print("     ALL VALIDATION CHECKS PASSED SUCCESSFULLY!        ")
    print("=" * 60)

if __name__ == "__main__":
    run()
