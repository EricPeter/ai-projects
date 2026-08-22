from __future__ import annotations

import numpy as np


def compute_stats(samples: list[float]) -> dict:
    if not samples:
        return {"mean": 0, "p50": 0, "p95": 0, "p99": 0, "std": 0, "n": 0}
    arr = np.array(samples)
    return {
        "mean": round(float(arr.mean()), 4),
        "p50": round(float(np.percentile(arr, 50)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
        "p99": round(float(np.percentile(arr, 99)), 4),
        "std": round(float(arr.std()), 4),
        "n": len(samples),
    }


def welch_t_test(a_samples: list[float], b_samples: list[float]) -> dict:
    """Two-sided Welch t-test comparing two latency distributions."""
    from scipy import stats as sp

    if len(a_samples) < 2 or len(b_samples) < 2:
        return {"significant": False, "reason": "insufficient samples"}

    t_stat, p_value = sp.ttest_ind(a_samples, b_samples, equal_var=False)
    a_mean = np.mean(a_samples)
    b_mean = np.mean(b_samples)
    effect_pct = ((b_mean - a_mean) / a_mean * 100) if a_mean else 0

    return {
        "test": "welch_t",
        "t_stat": round(float(t_stat), 4),
        "p_value": round(float(p_value), 6),
        "significant": bool(p_value < 0.05),
        "winner": "b" if (p_value < 0.05 and b_mean < a_mean) else ("a" if p_value < 0.05 else "none"),
        "effect_pct": round(float(effect_pct), 2),
        "a_mean": round(float(a_mean), 4),
        "b_mean": round(float(b_mean), 4),
    }


def two_proportion_z_test(
    a_successes: int, a_total: int, b_successes: int, b_total: int
) -> dict:
    """Z-test for comparing success rates between two model variants."""
    from statsmodels.stats.proportion import proportions_ztest

    if a_total == 0 or b_total == 0:
        return {"significant": False, "reason": "no samples"}

    counts = np.array([b_successes, a_successes])
    nobs = np.array([b_total, a_total])
    stat, p_value = proportions_ztest(counts, nobs)

    a_rate = a_successes / a_total
    b_rate = b_successes / b_total

    return {
        "test": "two_proportion_z",
        "z_stat": round(float(stat), 4),
        "p_value": round(float(p_value), 6),
        "significant": bool(p_value < 0.05),
        "winner": "b" if (p_value < 0.05 and b_rate > a_rate) else ("a" if p_value < 0.05 else "none"),
        "a_rate": round(a_rate, 4),
        "b_rate": round(b_rate, 4),
        "effect_pct": round((b_rate - a_rate) / a_rate * 100 if a_rate else 0, 2),
    }
