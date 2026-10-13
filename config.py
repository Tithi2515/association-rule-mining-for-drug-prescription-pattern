"""
config.py
================================================================================
Configuration parameters for the Drug Association Rule Mining Project.

This file defines dataset paths, candidate substance columns, preprocessing rules,
mining thresholds, and benchmark targets from the university project report.
================================================================================
"""

from pathlib import Path

# ------------------------------------------------------------------------------
# 1. PATH CONFIGURATION
# ------------------------------------------------------------------------------
# Base directory of the project (relative to this file)
BASE_DIR = Path(__file__).resolve().parent

# Path to the Connecticut Accidental Drug Related Deaths CSV dataset
DATA_PATH = BASE_DIR / "data" / "accidental_drug_deaths.csv"

# Directory where generated tables, reports, and charts are saved
OUTPUTS_DIR = BASE_DIR / "outputs"


# ------------------------------------------------------------------------------
# 2. SUBSTANCE COLUMNS SELECTION
# ------------------------------------------------------------------------------
# The project report specifies exactly 17 substance items, but does not name
# which 17 columns were used from the 18 plausible candidate columns in the CSV.
#
# Plausible candidates in the CSV:
#   Heroin, Cocaine, Fentanyl, Fentanyl Analogue, Oxycodone, Oxymorphone,
#   Ethanol, Hydrocodone, Benzodiazepine, Methadone, Meth/Amphetamine,
#   Amphet, Tramad, Hydromorphone, Morphine (Not Heroin), Xylazine,
#   Gabapentin, Opiate NOS.
#
# Always EXCLUDED columns (non-item flags, death certificate notes, free text):
#   - Any Opioid
#   - Heroin/Morph/Codeine
#   - Heroin death certificate (DC)
#   - Other Opioid
#   - Other
#   - All demographic, geographic, and date columns
#
# NOTE: Edit this list to exactly 17 columns once verified.
# (Running check_dataset.py shows that dropping either 'Morphine (Not Heroin)'
# or 'Meth/Amphetamine' yields exactly 12,805 transactions matching the report).
SUBSTANCE_COLUMNS = [
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


# ------------------------------------------------------------------------------
# 3. DETECTION RULE
# ------------------------------------------------------------------------------
# Rule used to convert raw CSV text values into binary (detected=1, not detected=0):
#
# Options:
#   - "non_blank_not_N":
#       A substance is detected if the stripped cell value is non-blank AND not "N".
#       This gracefully handles:
#         * "Y" / "y" (standard positive detection)
#         * "Y POPS" / "Y (PTCH)" (Fentanyl delivery forms)
#         * "P" (Ethanol positive detection)
#         * Medical text notes (e.g. in Morphine column)
#         * Explicit "N" as NOT detected
#         * Empty string "" as NOT detected
#
#   - "starts_with_Y":
#       A substance is detected only if the stripped uppercase value begins with "Y".
#       (Treats "P" or free-text notes as not detected).
#
DETECTED_RULE = "non_blank_not_N"


# ------------------------------------------------------------------------------
# 4. ASSOCIATION RULE MINING PARAMETERS
# ------------------------------------------------------------------------------
# Minimum support threshold: a frequent itemset must appear in at least 2% of deaths
MIN_SUPPORT = 0.02

# Minimum confidence threshold: rules must have at least 50% conditional probability
MIN_CONFIDENCE = 0.50

# Minimum lift threshold: rules must show positive association (lift >= 1.00)
MIN_LIFT = 1.00

# Number of repetitions for timing performance benchmark
TIMING_RUNS = 10

# Support levels tested in the runtime-vs-support experiment (report Figure 3)
SUPPORT_EXPERIMENT_VALS = [0.01, 0.02, 0.04, 0.08]


# ------------------------------------------------------------------------------
# 5. REPORTED TARGET CONFIGURATION (Comparison Only)
# ------------------------------------------------------------------------------
# These numbers were reported in the student's project report/presentation.
# HARD RULE: They are defined here strictly for a final "calculated vs reported"
# comparison table. They are NEVER hardcoded as program output or assumed to match.
REPORTED_TARGETS = {
    "transactions": 12805,
    "items": 17,
    "min_support": 0.02,
    "min_confidence": 0.50,
    "min_lift": 1.00,
    "frequent_itemsets": 51,
    "rules": 12,
    "apriori_time_sec": 0.0063,
    "fpgrowth_time_sec": 1.4581,
    "strongest_rule": {
        "antecedents": "Fentanyl Analogue + Heroin",
        "consequents": "Fentanyl",
        "support": 0.028,
        "confidence": 1.000,
        "lift": 1.453,
    },
}
