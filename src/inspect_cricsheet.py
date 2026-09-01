import os
import json
import glob
from collections import Counter, defaultdict
from pathlib import Path

def inspect_dataset(data_dir):
    json_files = glob.glob(os.path.join(data_dir, "**", "*.json"), recursive=True)
    print(f"--- CRICSHEET JSON INSPECTION REPORT ---")
    print(f"Total JSON files found: {len(json_files)}")

    match_types = Counter()
    outcome_results = Counter()
    has_winner = 0
    has_method_dls = 0
    has_super_over_flag = 0
    total_innings_types = Counter()
    extras_keys_found = Counter()
    wicket_kinds_found = Counter()
    missing_fields = defaultdict(int)
    total_deliveries = 0

    parsed_count = 0
    failed_files = []

    for filepath in json_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            parsed_count += 1
        except Exception as e:
            failed_files.append((filepath, str(e)))
            continue

        info = data.get("info", {})
        
        # Match type
        mtype = info.get("match_type", "UNKNOWN")
        match_types[mtype] += 1

        # Check required info fields
        for field in ["dates", "teams", "venue", "toss"]:
            if field not in info:
                missing_fields[field] += 1

        # City check
        if "city" not in info:
            missing_fields["city"] += 1

        # Outcome checks
        outcome = info.get("outcome", {})
        if "winner" in outcome:
            has_winner += 1
        
        result_str = outcome.get("result", "completed")
        outcome_results[result_str] += 1

        if "method" in outcome or outcome.get("method") in ["D/L", "DLS"]:
            has_method_dls += 1

        if "eliminator" in outcome or outcome.get("result") == "tie" or "bowl out" in outcome.get("method", "").lower():
            has_super_over_flag += 1

        # Innings inspection
        innings_list = data.get("innings", [])
        for idx, inn in enumerate(innings_list, start=1):
            is_super_over = inn.get("super_over", False)
            if is_super_over:
                total_innings_types["super_over"] += 1
            else:
                total_innings_types[f"innings_{idx}"] += 1

            for over_obj in inn.get("overs", []):
                for deliv in over_obj.get("deliveries", []):
                    total_deliveries += 1
                    extras_dict = deliv.get("extras", {})
                    for k in extras_dict:
                        extras_keys_found[k] += 1

                    wickets_list = deliv.get("wickets", [])
                    for w in wickets_list:
                        wicket_kinds_found[w.get("kind", "unknown")] += 1

    print(f"Successfully parsed files: {parsed_count}")
    print(f"Failed files count: {len(failed_files)}")
    if failed_files:
        print(f"Failed files list: {failed_files[:5]}")

    print("\nMatch Types Distribution:")
    for mt, count in match_types.most_common():
        print(f"  {mt}: {count}")

    print("\nOutcome Results Distribution:")
    for res, count in outcome_results.most_common():
        print(f"  {res}: {count}")

    print(f"\nMatches with winner: {has_winner}")
    print(f"Matches with DLS/method: {has_method_dls}")
    print(f"Super Over / Eliminator flags in outcome: {has_super_over_flag}")

    print("\nMissing Fields Summary in Info:")
    for k, count in missing_fields.items():
        print(f"  Missing '{k}': {count} files")

    print(f"\nTotal Innings Parsed: {dict(total_innings_types)}")
    print(f"Total Deliveries Parsed: {total_deliveries}")

    print("\nExtras Keys Found across Deliveries:")
    for k, count in extras_keys_found.most_common():
        print(f"  {k}: {count}")

    print("\nWicket Kinds Found across Deliveries:")
    for k, count in wicket_kinds_found.most_common():
        print(f"  {k}: {count}")

if __name__ == "__main__":
    raw_path = os.path.join("data", "raw", "t20s_male_json")
    inspect_dataset(raw_path)
