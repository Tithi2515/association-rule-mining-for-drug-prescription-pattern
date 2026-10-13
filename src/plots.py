"""
plots.py
================================================================================
Visualization generation module for the Drug Association Rule Mining Project.
Generates publication-quality charts saved into outputs/ for reports and presentations:
1. Top 10 Substances (horizontal bar)
2. Support vs Confidence scatter (colored by Lift)
3. Runtime comparison bar chart (Apriori vs FP-Growth)
4. Runtime vs Minimum Support line chart (Scalability / Figure 3 replica)
5. Top Association Rules by Lift (horizontal bar)
================================================================================
"""

from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd


def set_plot_style():
    """Configures clean, modern aesthetic styling for matplotlib figures."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["grid.color"] = "#e5e7eb"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.alpha"] = 0.7


def plot_top_substances(
    freq_df: pd.DataFrame,
    output_path: Path,
    top_n: int = 10,
) -> None:
    """
    Plots horizontal bar chart of top substances by prevalence in mortality records.
    """
    set_plot_style()
    df_top = freq_df.head(top_n).iloc[::-1]  # Reverse for descending visual display

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    bars = ax.barh(
        df_top["Substance"],
        df_top["Frequency"],
        color="#2563eb",
        edgecolor="#1d4ed8",
        height=0.65,
    )

    # Add count and percentage labels at end of each bar
    max_val = df_top["Frequency"].max()
    for bar, pct in zip(bars, df_top["Percentage"]):
        width = bar.get_width()
        ax.text(
            width + (max_val * 0.01),
            bar.get_y() + bar.get_height() / 2,
            f"{int(width):,}  ({pct:.1f}%)",
            va="center",
            ha="left",
            fontsize=9.5,
            color="#1f2937",
            fontweight="medium",
        )

    ax.set_title(f"Top {top_n} Detected Substances in Accidental Drug Deaths", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Number of Mortality Records Detected", fontsize=11, labelpad=10)
    ax.set_xlim(0, max_val * 1.18)
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_support_vs_confidence(
    rules_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Scatter plot showing Support vs Confidence, colored and sized by Lift.
    """
    set_plot_style()
    if rules_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)

    scatter = ax.scatter(
        rules_df["support"],
        rules_df["confidence"],
        c=rules_df["lift"],
        s=rules_df["lift"] * 120,
        cmap="viridis",
        alpha=0.85,
        edgecolors="#1e293b",
        linewidths=1.2,
    )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Lift Metric", fontsize=11, labelpad=10)

    # Annotate top rule with highest lift
    top_rule = rules_df.sort_values(by=["lift", "confidence"], ascending=False).iloc[0]
    ax.annotate(
        f"Strongest Rule:\n{top_rule['rule_str']}\n(Lift: {top_rule['lift']:.3f}, Conf: {top_rule['confidence']:.2f})",
        xy=(top_rule["support"], top_rule["confidence"]),
        xytext=(top_rule["support"] + 0.015, top_rule["confidence"] - 0.08),
        arrowprops=dict(facecolor="#dc2626", shrink=0.08, width=1.5, headwidth=6),
        fontsize=9,
        fontweight="semibold",
        color="#991b1b",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fef2f2", edgecolor="#f87171", alpha=0.9),
    )

    ax.set_title("Association Rules: Support vs Confidence (Colored by Lift)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Support (Relative Prevalence in Records)", fontsize=11, labelpad=10)
    ax.set_ylabel("Confidence (Conditional Probability)", fontsize=11, labelpad=10)
    ax.set_ylim(min(0.45, rules_df["confidence"].min() - 0.05), 1.05)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_runtime_comparison(
    benchmarks: Dict[str, Any],
    output_path: Path,
) -> None:
    """
    Grouped bar chart comparing execution times between Apriori and FP-Growth.
    """
    set_plot_style()
    algorithms = ["Apriori", "FP-Growth"]
    first_runs = [benchmarks["apriori"]["first_run_sec"], benchmarks["fpgrowth"]["first_run_sec"]]
    median_runs = [benchmarks["apriori"]["median_sec"], benchmarks["fpgrowth"]["median_sec"]]

    x = range(len(algorithms))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)
    b1 = ax.bar([i - width / 2 for i in x], first_runs, width, label="First Run", color="#3b82f6", edgecolor="#1d4ed8")
    b2 = ax.bar([i + width / 2 for i in x], median_runs, width, label=f"Median ({benchmarks['runs']} Runs)", color="#10b981", edgecolor="#047857")

    # Add value annotations on bars
    for bars in [b1, b2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + (max(first_runs + median_runs) * 0.02),
                f"{height:.4f}s",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="medium",
            )

    ax.set_title("Frequent Itemset Mining Runtime Benchmark (min_support = 0.02)", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylabel("Execution Time (seconds)", fontsize=11, labelpad=10)
    ax.set_xticks(list(x))
    ax.set_xticklabels(algorithms, fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(first_runs + median_runs) * 1.25)
    ax.legend(frameon=True, facecolor="#ffffff", framealpha=0.9)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_runtime_vs_support(
    support_bench_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Line chart plotting execution time as minimum support varies (replicates Report Figure 3).
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    ax.plot(
        support_bench_df["min_support"],
        support_bench_df["apriori_time_sec"],
        marker="o",
        linewidth=2.2,
        color="#2563eb",
        label="Apriori",
    )
    ax.plot(
        support_bench_df["min_support"],
        support_bench_df["fpgrowth_time_sec"],
        marker="s",
        linewidth=2.2,
        color="#dc2626",
        label="FP-Growth",
    )

    # Annotate itemset counts at each data point
    for _, row in support_bench_df.iterrows():
        ax.annotate(
            f"{int(row['itemset_count'])} itemsets",
            xy=(row["min_support"], max(row["apriori_time_sec"], row["fpgrowth_time_sec"])),
            xytext=(row["min_support"], max(row["apriori_time_sec"], row["fpgrowth_time_sec"]) * 1.15 + 0.003),
            ha="center",
            fontsize=8.5,
            color="#4b5563",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#f3f4f6", edgecolor="#d1d5db", alpha=0.8),
        )

    ax.set_title("Scalability Experiment: Runtime vs Minimum Support (Figure 3 Replica)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Minimum Support Threshold", fontsize=11, labelpad=10)
    ax.set_ylabel("Median Execution Time (seconds)", fontsize=11, labelpad=10)
    ax.set_xticks(support_bench_df["min_support"].tolist())
    ax.legend(frameon=True, facecolor="#ffffff")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_top_rules_by_lift(
    rules_df: pd.DataFrame,
    output_path: Path,
    top_n: int = 10,
) -> None:
    """
    Plots horizontal bar chart of top association rules ranked by Lift.
    """
    set_plot_style()
    if rules_df.empty:
        return

    df_top = rules_df.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)
    bars = ax.barh(
        df_top["rule_str"],
        df_top["lift"],
        color="#8b5cf6",
        edgecolor="#6d28d9",
        height=0.6,
    )

    max_lift = df_top["lift"].max()
    for bar, conf, supp in zip(bars, df_top["confidence"], df_top["support"]):
        width = bar.get_width()
        ax.text(
            width + (max_lift * 0.01),
            bar.get_y() + bar.get_height() / 2,
            f"Lift: {width:.3f} | Conf: {conf:.2f} | Supp: {supp:.3f}",
            va="center",
            ha="left",
            fontsize=8.5,
            color="#1f2937",
        )

    ax.set_title(f"Top {top_n} Association Rules Ranked by Lift", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Lift Value", fontsize=11, labelpad=10)
    ax.set_xlim(0, max_lift * 1.35)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
