"""
data_loader.py
================================================================================
Robust data loading utilities for the Drug Association Rule Mining Project.
Handles file verification, multiple encodings, whitespace trimming, and string types.
================================================================================
"""

import sys
from pathlib import Path
from typing import Union
import pandas as pd


def load_raw_data(filepath: Union[Path, str]) -> pd.DataFrame:
    """
    Loads the Accidental Drug Related Deaths dataset from CSV.

    Parameters
    ----------
    filepath : Path or str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame with string dtype and trimmed whitespace.

    Raises
    ------
    FileNotFoundError
        If the file cannot be located at the provided path.
    RuntimeError
        If decoding fails across all tested encodings.
    """
    path = Path(filepath)

    if not path.exists():
        url = "https://data.ct.gov/api/views/rybz-nyjw/rows.csv?accessType=DOWNLOAD"
        try:
            import urllib.request
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] Dataset not found at {path}. Automatically downloading from Connecticut Open Data...")
            urllib.request.urlretrieve(url, path)
            print(f"[SUCCESS] Downloaded dataset to {path}")
        except Exception as dl_err:
            err_msg = (
                f"\n[ERROR] Dataset not found at: {path.resolve()}\n"
                f"Automatic download attempt failed: {dl_err}\n"
                f"Please ensure the CSV file is placed at:\n"
                f"  {path}\n"
                f"You can download it manually from Connecticut Open Data:\n"
                f"  {url}"
            )
            raise FileNotFoundError(err_msg) from dl_err

    # Try common encodings to avoid decode failures on various Windows locales
    encodings = ["utf-8", "utf-8-sig", "latin-1"]
    last_err = None

    for enc in encodings:
        try:
            # We load with dtype=str and keep_default_na=False so that empty cells
            # remain as empty strings "" rather than being converted to NaN floats.
            df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc)

            # Strip whitespace from column names
            df.columns = df.columns.str.strip()

            # Strip leading and trailing whitespace from string values
            # Using vectorized applymap/map for robustness across pandas versions
            if hasattr(df, "map"):
                df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
            else:
                df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

            return df
        except UnicodeDecodeError as err:
            last_err = err
            continue

    raise RuntimeError(
        f"Failed to read CSV at {path} with encodings {encodings}. Last error: {last_err}"
    )
