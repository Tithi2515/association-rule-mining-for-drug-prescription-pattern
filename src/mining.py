"""
mining.py
================================================================================
Frequent itemset mining and association rule extraction algorithms.
Implements:
- Apriori algorithm (level-wise candidate generation and join/prune)
- FP-Growth algorithm (frequent pattern tree without candidate generation)
- Association rule generation with support, confidence, and lift metrics
- Rigorous mathematical comparison between Apriori and FP-Growth outputs
- Multi-run benchmark timing with time.perf_counter()
- Parameter sensitivity benchmarks across support thresholds
================================================================================
"""

import time
import inspect
import statistics
from typing import Dict, Any, List, Tuple
import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules


def mine_apriori(df_bool: pd.DataFrame, min_support: float = 0.02) -> pd.DataFrame:
    """
    Discovers frequent itemsets using the Apriori algorithm.

    Parameters
    ----------
    df_bool : pd.DataFrame
        One-hot/boolean encoded transaction DataFrame.
    min_support : float, default=0.02
        Minimum support threshold.

    Returns
    -------
    pd.DataFrame
        Frequent itemsets with columns ['support', 'itemsets', 'length'].
    """
    fis = apriori(df_bool, min_support=min_support, use_colnames=True)
    if not fis.empty:
        fis["length"] = fis["itemsets"].apply(len)
        fis = fis.sort_values(by=["length", "support"], ascending=[True, False]).reset_index(drop=True)
    return fis


def mine_fpgrowth(df_bool: pd.DataFrame, min_support: float = 0.02) -> pd.DataFrame:
    """
    Discovers frequent itemsets using the FP-Growth algorithm.

    Parameters
    ----------
    df_bool : pd.DataFrame
        One-hot/boolean encoded transaction DataFrame.
    min_support : float, default=0.02
        Minimum support threshold.

    Returns
    -------
    pd.DataFrame
        Frequent itemsets with columns ['support', 'itemsets', 'length'].
    """
    fis = fpgrowth(df_bool, min_support=min_support, use_colnames=True)
    if not fis.empty:
        fis["length"] = fis["itemsets"].apply(len)
        fis = fis.sort_values(by=["length", "support"], ascending=[True, False]).reset_index(drop=True)
    return fis


def extract_rules(
    frequent_itemsets: pd.DataFrame,
    min_confidence: float = 0.50,
    min_lift: float = 1.00,
) -> pd.DataFrame:
    """
    Generates association rules from frequent itemsets filtered by confidence and lift.
    Gracefully inspects and adapts to differences across mlxtend versions.

    Parameters
    ----------
    frequent_itemsets : pd.DataFrame
        Frequent itemsets discovered by Apriori or FP-Growth.
    min_confidence : float, default=0.50
        Minimum rule confidence threshold.
    min_lift : float, default=1.00
        Minimum rule lift threshold.

    Returns
    -------
    pd.DataFrame
        DataFrame of association rules, sorted by lift descending then confidence descending.
    """
    if frequent_itemsets.empty:
        return pd.DataFrame(
            columns=[
                "antecedents", "consequents", "antecedent support",
                "consequent support", "support", "confidence", "lift",
                "antecedents_str", "consequents_str", "rule_str",
            ]
        )

    # In mlxtend, some versions have num_itemsets parameter while others do not
    sig = inspect.signature(association_rules)
    has_num_itemsets = "num_itemsets" in sig.parameters

    try:
        if has_num_itemsets:
            rules = association_rules(
                frequent_itemsets,
                num_itemsets=len(frequent_itemsets),
                metric="confidence",
                min_threshold=min_confidence,
            )
        else:
            rules = association_rules(
                frequent_itemsets,
                metric="confidence",
                min_threshold=min_confidence,
            )
    except TypeError:
        # Fallback for unexpected keyword signature variations
        rules = association_rules(
            frequent_itemsets,
            metric="confidence",
            min_threshold=min_confidence,
        )

    if rules.empty:
        return rules

    # Filter by minimum lift threshold
    rules = rules[rules["lift"] >= min_lift].copy()

    # Create clean human-readable string representations
    rules["antecedents_str"] = rules["antecedents"].apply(lambda s: " + ".join(sorted(list(s))))
    rules["consequents_str"] = rules["consequents"].apply(lambda s: " + ".join(sorted(list(s))))
    rules["rule_str"] = rules["antecedents_str"] + " => " + rules["consequents_str"]

    # Sort rules primarily by lift (descending) and secondarily by confidence (descending)
    rules = rules.sort_values(by=["lift", "confidence"], ascending=[False, False]).reset_index(drop=True)
    return rules


def compare_mining_results(
    fis_apriori: pd.DataFrame,
    fis_fpgrowth: pd.DataFrame,
    rules_apriori: pd.DataFrame,
    rules_fpgrowth: pd.DataFrame,
    tolerance: float = 1e-6,
) -> Dict[str, Any]:
    """
    Rigorously verifies whether Apriori and FP-Growth produced identical itemsets
    and association rules on the same dataset.

    Parameters
    ----------
    fis_apriori : pd.DataFrame
        Itemsets from Apriori.
    fis_fpgrowth : pd.DataFrame
        Itemsets from FP-Growth.
    rules_apriori : pd.DataFrame
        Rules from Apriori.
    rules_fpgrowth : pd.DataFrame
        Rules from FP-Growth.
    tolerance : float, default=1e-6
        Floating-point comparison tolerance for metric values.

    Returns
    -------
    Dict[str, Any]
        Structured comparison report.
    """
    # 1. Itemset counts
    cnt_ap = len(fis_apriori)
    cnt_fp = len(fis_fpgrowth)
    counts_match = (cnt_ap == cnt_fp)

    # 2. Check itemset sets and support values
    # Map frozenset -> support
    map_ap = {row["itemsets"]: row["support"] for _, row in fis_apriori.iterrows()}
    map_fp = {row["itemsets"]: row["support"] for _, row in fis_fpgrowth.iterrows()}

    itemset_keys_match = (set(map_ap.keys()) == set(map_fp.keys()))

    itemset_support_match = True
    support_discrepancies = []
    if itemset_keys_match:
        for k in map_ap:
            diff = abs(map_ap[k] - map_fp[k])
            if diff > tolerance:
                itemset_support_match = False
                support_discrepancies.append((k, map_ap[k], map_fp[k], diff))
    else:
        itemset_support_match = False

    # 3. Check rule sets and metrics
    # Key: (frozenset(antecedents), frozenset(consequents)) -> (support, confidence, lift)
    rules_map_ap = {
        (row["antecedents"], row["consequents"]): (row["support"], row["confidence"], row["lift"])
        for _, row in rules_apriori.iterrows()
    }
    rules_map_fp = {
        (row["antecedents"], row["consequents"]): (row["support"], row["confidence"], row["lift"])
        for _, row in rules_fpgrowth.iterrows()
    }

    rule_keys_match = (set(rules_map_ap.keys()) == set(rules_map_fp.keys()))

    rule_metrics_match = True
    rule_discrepancies = []
    if rule_keys_match:
        for k in rules_map_ap:
            m_ap = rules_map_ap[k]
            m_fp = rules_map_fp[k]
            if any(abs(a - b) > tolerance for a, b in zip(m_ap, m_fp)):
                rule_metrics_match = False
                rule_discrepancies.append((k, m_ap, m_fp))
    else:
        rule_metrics_match = False

    return {
        "apriori_itemset_count": cnt_ap,
        "fpgrowth_itemset_count": cnt_fp,
        "itemset_counts_match": counts_match,
        "itemset_sets_identical": itemset_keys_match,
        "itemset_support_values_match": itemset_support_match,
        "apriori_rule_count": len(rules_apriori),
        "fpgrowth_rule_count": len(rules_fpgrowth),
        "rule_counts_match": (len(rules_apriori) == len(rules_fpgrowth)),
        "rules_identical": rule_keys_match,
        "rule_metrics_match": rule_metrics_match,
        "all_match": (
            counts_match and itemset_keys_match and itemset_support_match
            and rule_keys_match and rule_metrics_match
        ),
        "support_discrepancies": support_discrepancies,
        "rule_discrepancies": rule_discrepancies,
    }


def benchmark_algorithms(
    df_bool: pd.DataFrame,
    min_support: float = 0.02,
    runs: int = 10,
) -> Dict[str, Any]:
    """
    Rigorously benchmarks execution time of Apriori vs FP-Growth using time.perf_counter().

    TIMING SCOPE:
    This function times strictly the frequent itemset mining step (candidate generation
    and itemset discovery) on identical data. It does NOT include rule generation or
    data loading.

    Parameters
    ----------
    df_bool : pd.DataFrame
        Boolean transaction dataset.
    min_support : float, default=0.02
        Minimum support threshold.
    runs : int, default=10
        Number of consecutive benchmarking runs.

    Returns
    -------
    Dict[str, Any]
        Detailed timing statistics including first run, median, min, max, and all runs.
    """
    apriori_times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = apriori(df_bool, min_support=min_support, use_colnames=True)
        t1 = time.perf_counter()
        apriori_times.append(t1 - t0)

    fpgrowth_times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = fpgrowth(df_bool, min_support=min_support, use_colnames=True)
        t1 = time.perf_counter()
        fpgrowth_times.append(t1 - t0)

    return {
        "runs": runs,
        "min_support": min_support,
        "apriori": {
            "first_run_sec": apriori_times[0],
            "median_sec": statistics.median(apriori_times),
            "mean_sec": statistics.mean(apriori_times),
            "min_sec": min(apriori_times),
            "max_sec": max(apriori_times),
            "all_runs_sec": apriori_times,
        },
        "fpgrowth": {
            "first_run_sec": fpgrowth_times[0],
            "median_sec": statistics.median(fpgrowth_times),
            "mean_sec": statistics.mean(fpgrowth_times),
            "min_sec": min(fpgrowth_times),
            "max_sec": max(fpgrowth_times),
            "all_runs_sec": fpgrowth_times,
        },
    }


def benchmark_support_levels(
    df_bool: pd.DataFrame,
    support_levels: List[float] = [0.01, 0.02, 0.04, 0.08],
) -> pd.DataFrame:
    """
    Evaluates execution time and frequent itemset counts as min_support varies
    (replicating the experiment in the student report's Figure 3).

    Parameters
    ----------
    df_bool : pd.DataFrame
        Boolean transaction dataset.
    support_levels : List[float]
        List of support thresholds to test.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['min_support', 'apriori_time_sec', 'fpgrowth_time_sec', 'itemset_count'].
    """
    records = []
    for supp in support_levels:
        # Time Apriori (median of 3 runs for stability)
        ap_times = []
        for _ in range(3):
            t0 = time.perf_counter()
            fis_ap = apriori(df_bool, min_support=supp, use_colnames=True)
            t1 = time.perf_counter()
            ap_times.append(t1 - t0)

        # Time FP-Growth (median of 3 runs for stability)
        fp_times = []
        for _ in range(3):
            t0 = time.perf_counter()
            fis_fp = fpgrowth(df_bool, min_support=supp, use_colnames=True)
            t1 = time.perf_counter()
            fp_times.append(t1 - t0)

        records.append({
            "min_support": supp,
            "apriori_time_sec": statistics.median(ap_times),
            "fpgrowth_time_sec": statistics.median(fp_times),
            "itemset_count": len(fis_ap),
        })

    return pd.DataFrame(records)
