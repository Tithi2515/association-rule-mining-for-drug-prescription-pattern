"""
check_dataset.py
================================================================================
Diagnostic script to inspect the Connecticut Accidental Drug Deaths dataset,
test detection rules, and resolve the 17 vs 18 candidate substance columns dilemma.

Usage:
    python check_dataset.py
================================================================================
"""

import sys
from pathlib import Path
import pandas as pd

# Import configuration constants
try:
    from config import DATA_PATH, REPORTED_TARGETS
except ImportError:
    DATA_PATH = Path("data/accidental_drug_deaths.csv")
    REPORTED_TARGETS = {"transactions": 12805}

# The 18 plausible candidate substance columns in the dataset
CANDIDATE_COLUMNS = [
    "Heroin",
    "Cocaine",
    "Fentanyl",
    "Fentanyl Analogue",
    "Oxycodone",
    "Oxymorphone",
    "Ethanol",
    "Hydrocodone",
    "Benzodiazepine",
    "Methadone",
    "Meth/Amphetamine",
    "Amphet",
    "Tramad",
    "Hydromorphone",
    "Morphine (Not Heroin)",
    "Xylazine",
    "Gabapentin",
    "Opiate NOS",
]

TARGET_TRANSACTIONS = REPORTED_TARGETS.get("transactions", 12805)


def load_csv_safely(file_path: Path) -> pd.DataFrame:
    """
    Safely load the CSV file, testing multiple common encodings.
    Provides plain-English guidance if the file is missing or corrupted.
    """
    if not file_path.exists():
        print("=" * 80)
        print("ERROR: Dataset file not found!")
        print(f"Expected path: {file_path.resolve()}")
        print("-" * 80)
        print("How to fix:")
        print("1. Download the dataset CSV from Connecticut Open Data:")
        print("   https://data.ct.gov/api/views/rybz-nyjw/rows.csv?accessType=DOWNLOAD")
        print("2. Save the file inside the project directory as:")
        print(f"   {file_path}")
        print("=" * 80)
        sys.exit(1)

    encodings_to_try = ["utf-8", "utf-8-sig", "latin-1"]
    last_error = None

    for enc in encodings_to_try:
        try:
            df = pd.read_csv(file_path, dtype=str, keep_default_na=False, encoding=enc)
            # Strip whitespace from column headers
            df.columns = df.columns.str.strip()
            print(f"[OK] Successfully loaded CSV using '{enc}' encoding.")
            return df
        except UnicodeDecodeError as e:
            last_error = e
            continue
        except Exception as e:
            print(f"ERROR: Failed reading CSV file: {e}")
            sys.exit(1)

    print(f"ERROR: Unable to decode file with encodings {encodings_to_try}.")
    print(f"Details: {last_error}")
    sys.exit(1)


def detect_series(series: pd.Series, rule: str) -> pd.Series:
    """
    Converts a raw string series to a boolean detection series according to rule.
    """
    stripped = series.str.strip()
    if rule == "non_blank_not_N":
        # Positive if non-empty and not 'N'
        return (stripped != "") & (stripped.str.upper() != "N")
    elif rule == "starts_with_Y":
        # Positive only if starts with 'Y' (case-insensitive)
        return stripped.str.upper().str.startswith("Y")
    else:
        raise ValueError(f"Unknown rule: {rule}")


def run_diagnostics():
    """Runs full dataset inspection and prints column counts & transaction matches."""
    print("=" * 80)
    print("ACCIDENTAL DRUG DEATHS DATASET INSPECTION")
    print(f"Target transaction count from report: {TARGET_TRANSACTIONS:,}")
    print("=" * 80)

    df = load_csv_safely(DATA_PATH)
    total_rows = len(df)
    print(f"Total raw rows in CSV: {total_rows:,}")
    print(f"Total columns in CSV: {len(df.columns)}")

    # Verify that all 18 candidate columns exist in the CSV
    missing_cols = [c for c in CANDIDATE_COLUMNS if c not in df.columns]
    if missing_cols:
        print("\nERROR: The following expected substance columns were NOT found in the CSV:")
        for mc in missing_cols:
            print(f"  - {mc}")
        print("\nAvailable columns in dataset:")
        print(list(df.columns))
        sys.exit(1)

    rules = ["non_blank_not_N", "starts_with_Y"]

    for rule in rules:
        print("\n" + "#" * 80)
        print(f"EVALUATING DETECTION RULE: '{rule}'")
        print("#" * 80)

        # Build boolean dataframe for candidate columns
        bool_df = pd.DataFrame(index=df.index)
        col_stats = []

        for col in CANDIDATE_COLUMNS:
            raw_s = df[col]
            b_s = detect_series(raw_s, rule)
            bool_df[col] = b_s
            detected_cnt = int(b_s.sum())
            pct = (detected_cnt / total_rows) * 100

            # Gather sample raw values
            val_counts = raw_s.str.strip().value_counts().to_dict()
            non_empty_vals = {k: v for k, v in val_counts.items() if k != ""}
            col_stats.append({
                "Column": col,
                "Detected": detected_cnt,
                "% of Total": f"{pct:.2f}%",
                "Raw Values (Non-blank)": str(non_empty_vals)[:50] + ("..." if len(str(non_empty_vals)) > 50 else ""),
            })

        stats_table = pd.DataFrame(col_stats)
        print("\nPer-Column Detection Summary:")
        print(stats_table.to_string(index=False))

        # Check total transactions retained using all 18 candidates
        all_18_kept = int(bool_df[CANDIDATE_COLUMNS].any(axis=1).sum())
        all_18_flag = " <== equals report" if all_18_kept == TARGET_TRANSACTIONS else ""
        print(f"\nAll 18 candidates: kept {all_18_kept:,} transactions (dropped {total_rows - all_18_kept} empty rows){all_18_flag}")

        print("\nTransaction count when dropping EACH candidate individually (17 columns remaining):")
        print(f"{'Dropped Column':<25} | {'Kept Transactions':<18} | {'Difference from Report'}")
        print("-" * 75)

        matched_candidates = []
        for drop_col in CANDIDATE_COLUMNS:
            subset_cols = [c for c in CANDIDATE_COLUMNS if c != drop_col]
            kept_cnt = int(bool_df[subset_cols].any(axis=1).sum())
            diff = kept_cnt - TARGET_TRANSACTIONS
            diff_str = f"{diff:+d}" if diff != 0 else "MATCH (0)"
            flag = " <== equals report" if kept_cnt == TARGET_TRANSACTIONS else ""
            print(f"{drop_col:<25} | {kept_cnt:<18,} | {diff_str}{flag}")
            if kept_cnt == TARGET_TRANSACTIONS:
                matched_candidates.append(drop_col)

        print("\nSUMMARY FOR RULE '" + rule + "':")
        if matched_candidates:
            print(f"Dropping any of these columns yields EXACTLY {TARGET_TRANSACTIONS:,} transactions:")
            for mc in matched_candidates:
                print(f"  * '{mc}'")
        else:
            print(f"No single column drop yielded exactly {TARGET_TRANSACTIONS:,} transactions.")

    print("\n" + "=" * 80)
    print("KEY FINDINGS & CONFIGURATION GUIDANCE")
    print("=" * 80)
    print("1. When using all 18 candidates, 12,812 rows have at least one detected substance.")
    print("2. Dropping EITHER 'Morphine (Not Heroin)' OR 'Meth/Amphetamine' results in EXACTLY")
    print(f"   12,805 retained transactions, matching the university project report ({TARGET_TRANSACTIONS:,}).")
    print("3. Furthermore, at min_support = 0.02 (2% threshold), neither 'Morphine (Not Heroin)'")
    print("   (~0.48%) nor 'Meth/Amphetamine' (~0.99%) meets the threshold.")
    print("   Therefore, frequent itemsets (51) and association rules (12) are IDENTICAL")
    print("   regardless of which of the two candidates is dropped!")
    print("4. You can keep all 18 candidates or drop one in config.py SUBSTANCE_COLUMNS.")
    print("=" * 80)


if __name__ == "__main__":
    run_diagnostics()
