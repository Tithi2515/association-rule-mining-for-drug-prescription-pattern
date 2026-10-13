"""
preprocessing.py
================================================================================
Data preprocessing pipeline for the Drug Association Rule Mining Project.
Selects target substance columns, converts values to binary/boolean detection
flags using configurable rules, and filters out non-substance transactions.
================================================================================
"""

from typing import List, Tuple, Dict, Any
import pandas as pd


def validate_and_select_columns(
    df: pd.DataFrame, substance_columns: List[str]
) -> pd.DataFrame:
    """
    Validates that all requested substance columns exist in the DataFrame
    and extracts only those columns.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset DataFrame.
    substance_columns : List[str]
        List of substance column names to retain.

    Returns
    -------
    pd.DataFrame
        DataFrame containing only the requested substance columns.

    Raises
    ------
    KeyError
        If any requested columns are missing from the input DataFrame.
    """
    missing = [c for c in substance_columns if c not in df.columns]
    if missing:
        available = list(df.columns)
        raise KeyError(
            f"The following requested substance columns were not found in the dataset:\n"
            f"  {missing}\n"
            f"Available columns in CSV:\n"
            f"  {available}"
        )
    return df[substance_columns].copy()


def convert_to_binary(
    df_substances: pd.DataFrame, rule: str = "non_blank_not_N"
) -> pd.DataFrame:
    """
    Converts string substance columns into boolean values (True = detected, False = not detected).

    Parameters
    ----------
    df_substances : pd.DataFrame
        DataFrame with substance columns as strings.
    rule : str, default="non_blank_not_N"
        Detection rule to apply:
        - "non_blank_not_N": Detected if cell after stripping is non-empty and != 'N'.
        - "starts_with_Y": Detected if cell after stripping and uppercasing starts with 'Y'.

    Returns
    -------
    pd.DataFrame
        DataFrame with boolean dtype (True/False) for each substance.
    """
    bool_df = pd.DataFrame(index=df_substances.index)

    for col in df_substances.columns:
        s = df_substances[col].astype(str).str.strip()

        if rule == "non_blank_not_N":
            # Handles 'Y', 'Y POPS', 'Y (PTCH)', 'P', 'y', notes, and ignores '', 'N', 'nan'
            is_detected = (s != "") & (s.str.upper() != "N") & (s.str.lower() != "nan")
        elif rule == "starts_with_Y":
            # Only considers values starting with 'Y' (handles 'Y', 'y', 'Y POPS', etc.)
            is_detected = s.str.upper().str.startswith("Y")
        else:
            raise ValueError(
                f"Unknown detection rule '{rule}'. Expected 'non_blank_not_N' or 'starts_with_Y'."
            )

        bool_df[col] = is_detected

    return bool_df


def filter_empty_transactions(
    bool_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, int, int, int]:
    """
    Drops rows where NO selected substance was detected (all False).

    In market basket / association rule mining, each transaction must contain
    at least one positive item; an empty transaction cannot produce co-occurrences.

    Parameters
    ----------
    bool_df : pd.DataFrame
        Boolean DataFrame of substances.

    Returns
    -------
    Tuple[pd.DataFrame, int, int, int]
        (filtered_df, initial_row_count, retained_row_count, dropped_row_count)
    """
    initial_rows = len(bool_df)
    has_any_substance = bool_df.any(axis=1)
    filtered_df = bool_df[has_any_substance].copy()
    retained_rows = len(filtered_df)
    dropped_rows = initial_rows - retained_rows

    return filtered_df, initial_rows, retained_rows, dropped_rows


def preprocess_data(
    raw_df: pd.DataFrame,
    substance_columns: List[str],
    rule: str = "non_blank_not_N",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Complete preprocessing pipeline.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Raw dataset DataFrame.
    substance_columns : List[str]
        List of target substance column names.
    rule : str
        Detection rule to apply.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        - Cleaned boolean DataFrame (ready for Apriori / FP-Growth)
        - Preprocessing metadata dictionary with row and column counts
    """
    df_sub = validate_and_select_columns(raw_df, substance_columns)
    df_bin = convert_to_binary(df_sub, rule=rule)
    cleaned_df, init_rows, kept_rows, dropped_rows = filter_empty_transactions(df_bin)

    meta = {
        "raw_rows": init_rows,
        "retained_transactions": kept_rows,
        "dropped_empty_rows": dropped_rows,
        "substance_count": len(substance_columns),
        "rule_used": rule,
        "substance_columns": list(substance_columns),
    }

    return cleaned_df, meta
