#!/usr/bin/env python3
"""
Statistical analysis of delivery times from pilot experiment.

Reads raw per-run data from output/pilot_raw_results.json (saved by
pilot_experiment.py) and computes:
  - Welch's t-test (two-tailed) per connectivity level
  - Cohen's d effect size
  - 95% CI on mean difference (Adaptive − Baseline)
  - Power analysis if p >= 0.05 at 20% connectivity

Usage:
  python scripts/delivery_time_analysis.py
"""

import json
import math
import sys
from pathlib import Path
from statistics import mean, stdev

import numpy as np
from scipy import stats

DATA_PATH = Path(__file__).resolve().parent.parent / "output" / "pilot_raw_results.json"
CONNECTIVITY_LEVELS = [0.75, 0.40, 0.20]


def cohens_d(a: list[float], b: list[float]) -> float:
    """Cohen's d with pooled standard deviation."""
    na, nb = len(a), len(b)
    sa, sb = stdev(a), stdev(b)
    pooled = math.sqrt(((na - 1) * sa**2 + (nb - 1) * sb**2) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return (mean(a) - mean(b)) / pooled


def main() -> None:
    if not DATA_PATH.exists():
        print(f"Error: {DATA_PATH} not found.")
        print("Run pilot_experiment.py first to generate raw data.")
        sys.exit(1)

    with DATA_PATH.open() as f:
        raw = json.load(f)

    print(f"Loaded raw data from {DATA_PATH}\n")

    # ── Statistical analysis ─────────────────────────────────────────────
    header = (
        f"{'Conn':>6}  {'Mean Adap':>9}  {'Mean Base':>9}  {'Diff':>7}  "
        f"{'t':>7}  {'df':>5}  {'Cohen d':>8}  {'p-value':>9}  {'95% CI':>20}"
    )
    sep = "=" * len(header)
    print(sep)
    print(header)
    print(sep)

    significant_at_20 = False
    effect_size_20 = 0.0

    for connectivity in CONNECTIVITY_LEVELS:
        adap_key = f"adaptive@{connectivity}"
        base_key = f"baseline@{connectivity}"

        adap_runs = raw.get(adap_key, [])
        base_runs = raw.get(base_key, [])

        # Extract per-run average delivery times, filtering None/null
        adap = [
            r["avg_delivery_time"]
            for r in adap_runs
            if r.get("avg_delivery_time") is not None
        ]
        base = [
            r["avg_delivery_time"]
            for r in base_runs
            if r.get("avg_delivery_time") is not None
        ]

        n = min(len(adap), len(base))
        if n < 3:
            print(
                f"{connectivity:>6.2f}  insufficient data "
                f"(adap={len(adap)}, base={len(base)})"
            )
            continue

        a_vals = adap[:n]
        b_vals = base[:n]

        m_a = mean(a_vals)
        m_b = mean(b_vals)
        diff = m_a - m_b

        # Welch's t-test (two-tailed)
        t_stat, p_val = stats.ttest_ind(a_vals, b_vals, equal_var=False)

        # Welch-Satterthwaite degrees of freedom
        na, nb = len(a_vals), len(b_vals)
        sa2 = float(np.var(a_vals, ddof=1))
        sb2 = float(np.var(b_vals, ddof=1))
        num = (sa2 / na + sb2 / nb) ** 2
        den = (sa2 / na) ** 2 / (na - 1) + (sb2 / nb) ** 2 / (nb - 1)
        df = num / den if den > 0 else na + nb - 2

        # Cohen's d
        d = cohens_d(a_vals, b_vals)

        # 95% CI on mean difference (Adaptive − Baseline)
        se_diff = math.sqrt(sa2 / na + sb2 / nb)
        t_crit = stats.t.ppf(0.975, df)
        ci_lo = diff - t_crit * se_diff
        ci_hi = diff + t_crit * se_diff

        ci_str = f"[{ci_lo:+.1f}, {ci_hi:+.1f}]"
        diff_str = f"{diff:+.1f}"

        print(
            f"{connectivity:>6.2f}  {m_a:>9.1f}  {m_b:>9.1f}  {diff_str:>7}  "
            f"{t_stat:>7.3f}  {df:>5.1f}  {d:>8.3f}  {p_val:>9.4f}  {ci_str:>20}"
        )

        if connectivity == 0.20:
            significant_at_20 = p_val < 0.05
            effect_size_20 = abs(d)

    print(sep)

    # ── Verdict ───────────────────────────────────────────────────────────
    print()
    if significant_at_20:
        print(
            "p < 0.05 at 20% connectivity → 30 runs is sufficient.\n"
            "These 30 runs ARE the formal experiment."
        )
    else:
        print("p >= 0.05 at 20% connectivity → 30 runs is NOT sufficient.")
        try:
            from statsmodels.stats.power import tt_ind_solve_power

            if effect_size_20 > 0:
                n_needed = tt_ind_solve_power(
                    effect_size=effect_size_20,
                    alpha=0.05,
                    power=0.80,
                    ratio=1.0,
                    alternative="two-sided",
                )
                print(
                    f"Observed |Cohen's d| = {effect_size_20:.3f}. "
                    f"For 80% power, need n = {math.ceil(n_needed)} runs "
                    f"per group."
                )
            else:
                print("Effect size is zero — no meaningful difference detected.")
        except ImportError:
            print("Install statsmodels for power analysis: " "pip install statsmodels")


if __name__ == "__main__":
    main()
