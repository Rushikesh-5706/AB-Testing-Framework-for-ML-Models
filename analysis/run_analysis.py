"""A/B test analysis pipeline with statistical significance testing.

Connects to the SQLite database, computes performance metrics for each
model variant, runs statistical tests, and outputs structured results.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

DB_PATH = os.getenv("DATABASE_NAME", "data/ab_test_logs.db")
RESULTS_PATH = os.getenv("RESULTS_PATH", "analysis/results.json")
SIGNIFICANCE_LEVEL = 0.05


def load_predictions(db_path: str) -> pd.DataFrame:
    """Load all prediction records from the database.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        DataFrame containing all prediction records.

    Raises:
        FileNotFoundError: If the database file does not exist.
    """
    import sqlite3

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}")

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM predictions", conn)
    conn.close()

    if df.empty:
        raise ValueError("No prediction records found in the database")

    return df


def compute_variant_metrics(df: pd.DataFrame, variant: str) -> dict:
    """Compute performance metrics for a single model variant.

    Args:
        df: Full predictions DataFrame.
        variant: Model variant label ('A' or 'B').

    Returns:
        Dictionary of computed metrics for this variant.
    """
    subset = df[df["model_variant"] == variant]
    if subset.empty:
        return {"error": f"No data for variant {variant}"}

    metrics = {
        "count": int(len(subset)),
        "mean_prediction": round(float(subset["prediction"].mean()), 4),
        "std_prediction": round(float(subset["prediction"].std()), 4),
        "mean_latency_ms": round(float(subset["latency_ms"].mean()), 3),
        "std_latency_ms": round(float(subset["latency_ms"].std()), 3),
        "median_latency_ms": round(float(subset["latency_ms"].median()), 3),
        "p95_latency_ms": round(float(np.percentile(subset["latency_ms"], 95)), 3),
        "p99_latency_ms": round(float(np.percentile(subset["latency_ms"], 99)), 3),
        "min_latency_ms": round(float(subset["latency_ms"].min()), 3),
        "max_latency_ms": round(float(subset["latency_ms"].max()), 3),
    }

    if "prediction_probability" in subset.columns:
        proba_col = subset["prediction_probability"].dropna()
        if not proba_col.empty:
            metrics["mean_probability"] = round(float(proba_col.mean()), 4)
            metrics["std_probability"] = round(float(proba_col.std()), 4)

    positive_rate = (subset["prediction"] == 1).mean()
    metrics["positive_prediction_rate"] = round(float(positive_rate), 4)

    return metrics


def run_statistical_tests(df: pd.DataFrame) -> dict:
    """Run statistical significance tests between model variants.

    Performs:
    - Welch's t-test on latency (continuous metric)
    - Mann-Whitney U test on predictions (non-parametric)
    - Chi-squared test on prediction categories (categorical)

    Args:
        df: Full predictions DataFrame.

    Returns:
        Dictionary of statistical test results.
    """
    df_a = df[df["model_variant"] == "A"]
    df_b = df[df["model_variant"] == "B"]
    results = {}

    if len(df_a) < 2 or len(df_b) < 2:
        return {"error": "Insufficient data for statistical tests (need >= 2 per variant)"}

    # --- Welch's t-test on latency ---
    t_stat, p_value = stats.ttest_ind(
        df_a["latency_ms"], df_b["latency_ms"], equal_var=False
    )
    results["latency_ttest"] = {
        "test": "Welch's t-test",
        "metric": "latency_ms",
        "t_statistic": round(float(t_stat), 4),
        "p_value": round(float(p_value), 6),
        "significant": bool(p_value < SIGNIFICANCE_LEVEL),
        "interpretation": (
            "Statistically significant difference in latency"
            if p_value < SIGNIFICANCE_LEVEL
            else "No statistically significant difference in latency"
        ),
    }

    # --- Mann-Whitney U test on predictions ---
    u_stat, p_value_mw = stats.mannwhitneyu(
        df_a["prediction"], df_b["prediction"], alternative="two-sided"
    )
    results["prediction_mannwhitney"] = {
        "test": "Mann-Whitney U test",
        "metric": "prediction",
        "u_statistic": round(float(u_stat), 4),
        "p_value": round(float(p_value_mw), 6),
        "significant": bool(p_value_mw < SIGNIFICANCE_LEVEL),
        "interpretation": (
            "Statistically significant difference in prediction distributions"
            if p_value_mw < SIGNIFICANCE_LEVEL
            else "No statistically significant difference in predictions"
        ),
    }

    # --- Chi-squared test on prediction categories ---
    contingency = pd.crosstab(df["model_variant"], df["prediction"])
    if contingency.shape[1] >= 2:
        chi2, p_value_chi, dof, expected = stats.chi2_contingency(contingency)
        results["prediction_chi_squared"] = {
            "test": "Chi-squared test",
            "metric": "prediction_category",
            "chi2_statistic": round(float(chi2), 4),
            "p_value": round(float(p_value_chi), 6),
            "degrees_of_freedom": int(dof),
            "significant": bool(p_value_chi < SIGNIFICANCE_LEVEL),
            "interpretation": (
                "Statistically significant association between model variant and prediction"
                if p_value_chi < SIGNIFICANCE_LEVEL
                else "No statistically significant association between variant and prediction"
            ),
        }
    else:
        results["prediction_chi_squared"] = {
            "test": "Chi-squared test",
            "metric": "prediction_category",
            "error": "Insufficient categories for chi-squared test",
        }

    # --- Welch's t-test on prediction probabilities ---
    if "prediction_probability" in df.columns:
        proba_a = df_a["prediction_probability"].dropna()
        proba_b = df_b["prediction_probability"].dropna()
        if len(proba_a) >= 2 and len(proba_b) >= 2:
            t_stat_p, p_value_p = stats.ttest_ind(
                proba_a, proba_b, equal_var=False
            )
            results["probability_ttest"] = {
                "test": "Welch's t-test",
                "metric": "prediction_probability",
                "t_statistic": round(float(t_stat_p), 4),
                "p_value": round(float(p_value_p), 6),
                "significant": bool(p_value_p < SIGNIFICANCE_LEVEL),
                "interpretation": (
                    "Statistically significant difference in prediction probabilities"
                    if p_value_p < SIGNIFICANCE_LEVEL
                    else "No statistically significant difference in probabilities"
                ),
            }

    return results


def generate_recommendation(metrics_a: dict, metrics_b: dict, tests: dict) -> dict:
    """Generate a recommendation based on analysis results.

    Args:
        metrics_a: Metrics for Model A.
        metrics_b: Metrics for Model B.
        tests: Statistical test results.

    Returns:
        Recommendation dictionary.
    """
    points_a = 0
    points_b = 0
    reasons = []

    if metrics_a.get("mean_latency_ms", float("inf")) < metrics_b.get("mean_latency_ms", float("inf")):
        points_a += 1
        reasons.append("Model A has lower average latency")
    else:
        points_b += 1
        reasons.append("Model B has lower average latency")

    if metrics_a.get("positive_prediction_rate", 0) != metrics_b.get("positive_prediction_rate", 0):
        chi_test = tests.get("prediction_chi_squared", {})
        if chi_test.get("significant", False):
            reasons.append("Significant difference in prediction distributions detected")

    latency_test = tests.get("latency_ttest", {})
    if latency_test.get("significant", False):
        reasons.append("Latency difference is statistically significant")

    if points_a > points_b:
        winner = "Model A (Logistic Regression)"
    elif points_b > points_a:
        winner = "Model B (XGBoost)"
    else:
        winner = "No clear winner — consider extending the experiment"

    return {
        "recommended_model": winner,
        "model_a_score": points_a,
        "model_b_score": points_b,
        "reasons": reasons,
    }


def run_analysis() -> dict:
    """Execute the full analysis pipeline.

    Returns:
        Complete analysis results dictionary.
    """
    print("=" * 60)
    print("A/B Test Analysis Pipeline")
    print("=" * 60)

    print(f"\nLoading predictions from {DB_PATH}...")
    df = load_predictions(DB_PATH)
    print(f"  Loaded {len(df)} prediction records")

    total_a = len(df[df["model_variant"] == "A"])
    total_b = len(df[df["model_variant"] == "B"])
    print(f"  Model A: {total_a} requests ({total_a/len(df)*100:.1f}%)")
    print(f"  Model B: {total_b} requests ({total_b/len(df)*100:.1f}%)")

    print("\nComputing metrics for Model A...")
    metrics_a = compute_variant_metrics(df, "A")
    print(f"  Mean prediction: {metrics_a.get('mean_prediction', 'N/A')}")
    print(f"  Mean latency: {metrics_a.get('mean_latency_ms', 'N/A')} ms")

    print("\nComputing metrics for Model B...")
    metrics_b = compute_variant_metrics(df, "B")
    print(f"  Mean prediction: {metrics_b.get('mean_prediction', 'N/A')}")
    print(f"  Mean latency: {metrics_b.get('mean_latency_ms', 'N/A')} ms")

    print("\nRunning statistical significance tests...")
    stat_tests = run_statistical_tests(df)
    for test_name, result in stat_tests.items():
        p_val = result.get("p_value", "N/A")
        sig = result.get("significant", "N/A")
        print(f"  {test_name}: p={p_val}, significant={sig}")

    print("\nGenerating recommendation...")
    recommendation = generate_recommendation(metrics_a, metrics_b, stat_tests)
    print(f"  Recommended: {recommendation['recommended_model']}")

    results = {
        "experiment_summary": {
            "total_requests": int(len(df)),
            "model_a_requests": total_a,
            "model_b_requests": total_b,
            "model_a_traffic_share": round(total_a / len(df) * 100, 1),
            "model_b_traffic_share": round(total_b / len(df) * 100, 1),
        },
        "model_a": metrics_a,
        "model_b": metrics_b,
        "statistical_tests": stat_tests,
        "recommendation": recommendation,
        "significance_level": SIGNIFICANCE_LEVEL,
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    try:
        run_analysis()
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
