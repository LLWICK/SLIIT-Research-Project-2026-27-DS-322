import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


def plot_all_taf_metrics():
    df = pd.read_csv(OUTPUT_DIR / "counterfactual_sweep_results.csv")
    crash_cols = [c for c in df.columns if c.startswith("crash_freq_")]
    vol_cols = [c for c in df.columns if c.startswith("price_volatility_")]
    eq_cols = [c for c in df.columns if c.startswith("equilibrium_score_")]
    profit_cols = [c for c in df.columns if c.startswith("mean_profit_")]

    df["mean_crash_freq"] = df[crash_cols].mean(axis=1)
    df["mean_volatility"] = df[vol_cols].mean(axis=1)
    df["mean_equilibrium"] = df[eq_cols].mean(axis=1)
    df["mean_profit"] = df[profit_cols].mean(axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # 1. Crash Frequency
    crash_summary = df.groupby("adoption_rate")["mean_crash_freq"].agg(["mean", "std"])
    axes[0, 0].errorbar(crash_summary.index, crash_summary["mean"], yerr=crash_summary["std"], marker="o", color="#d9534f", capsize=4)
    axes[0, 0].set_title("(a) Price Crash Frequency vs Adoption Rate", fontweight="bold")
    axes[0, 0].set_xlabel("MSRS Adoption Rate")
    axes[0, 0].set_ylabel("Mean Crash Frequency (ΔP < -25%)")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)

    # 2. Price Volatility
    vol_summary = df.groupby("adoption_rate")["mean_volatility"].agg(["mean", "std"])
    axes[0, 1].errorbar(vol_summary.index, vol_summary["mean"], yerr=vol_summary["std"], marker="o", color="#f0ad4e", capsize=4)
    axes[0, 1].set_title("(b) Market Price Volatility vs Adoption Rate", fontweight="bold")
    axes[0, 1].set_xlabel("MSRS Adoption Rate")
    axes[0, 1].set_ylabel("Price Volatility (Std Dev of % Change)")
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)

    # 3. Supply-Demand Equilibrium Score
    eq_summary = df.groupby("adoption_rate")["mean_equilibrium"].agg(["mean", "std"])
    axes[1, 0].errorbar(eq_summary.index, eq_summary["mean"], yerr=eq_summary["std"], marker="o", color="#5cb85c", capsize=4)
    axes[1, 0].set_title("(c) Supply-Demand Equilibrium Score vs Adoption Rate", fontweight="bold")
    axes[1, 0].set_xlabel("MSRS Adoption Rate")
    axes[1, 0].set_ylabel("Equilibrium Score (0.0 to 1.0)")
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)

    # 4. Farmer Income / Profit Stability
    profit_summary = df.groupby("adoption_rate")["mean_profit"].agg(["mean", "std"])
    axes[1, 1].errorbar(profit_summary.index, profit_summary["mean"], yerr=profit_summary["std"], marker="o", color="#0275d8", capsize=4)
    axes[1, 1].set_title("(d) Farmer Average Net Profit vs Adoption Rate", fontweight="bold")
    axes[1, 1].set_xlabel("MSRS Adoption Rate")
    axes[1, 1].set_ylabel("Mean Seasonal Profit (Rs./farmer)")
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "counterfactual_plot.png"
    plt.savefig(out_path, dpi=200)
    print(f"Saved 4-panel TAF metric dashboard -> {out_path}")


if __name__ == "__main__":
    plot_all_taf_metrics()

