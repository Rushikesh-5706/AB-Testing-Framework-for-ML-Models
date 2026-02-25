"""Streamlit dashboard for visualizing A/B test results.

Displays model variant comparisons, traffic distribution, latency analysis,
prediction distributions, and statistical significance indicators.
"""

import json
import os
import sqlite3

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DB_PATH = os.getenv("DATABASE_NAME", "data/ab_test_logs.db")
RESULTS_PATH = os.getenv("RESULTS_PATH", "analysis/results.json")

st.set_page_config(
    page_title="A/B Testing Dashboard",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_data() -> pd.DataFrame | None:
    """Load prediction data from the SQLite database."""
    if not os.path.exists(DB_PATH):
        return None
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM predictions", conn)
    conn.close()
    return df if not df.empty else None


def load_results() -> dict | None:
    """Load pre-computed analysis results from JSON."""
    if not os.path.exists(RESULTS_PATH):
        return None
    with open(RESULTS_PATH, "r") as f:
        return json.load(f)


def main() -> None:
    """Render the A/B testing dashboard."""
    st.title("🧪 A/B Testing Dashboard for ML Models")
    st.markdown(
        "Real-time visualization of experiment results comparing "
        "**Model A (Logistic Regression)** vs **Model B (XGBoost)**"
    )
    st.markdown("---")

    df = load_data()
    results = load_results()

    if df is None:
        st.error(
            "No prediction data found. Start the API and send traffic, "
            "then run the analysis pipeline."
        )
        return

    # --- Offline Model Accuracy (from training metadata) ---
    st.header("🎓 Offline Training Metrics")
    metadata_path = os.getenv("MODEL_METADATA_PATH", "api/models/model_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        models_meta = metadata.get("models", {})
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            m = models_meta.get("A", {}).get("metrics", {})
            st.subheader("Model A — Logistic Regression")
            st.metric("Accuracy", f"{m.get('accuracy', 0):.2%}")
            st.metric("F1 Score", f"{m.get('f1_score', 0):.4f}")
            st.metric("ROC-AUC", f"{m.get('roc_auc', 0):.4f}")
        
        with col_m2:
            m = models_meta.get("B", {}).get("metrics", {})
            st.subheader("Model B — XGBoost")
            st.metric("Accuracy", f"{m.get('accuracy', 0):.2%}")
            st.metric("F1 Score", f"{m.get('f1_score', 0):.4f}")
            st.metric("ROC-AUC", f"{m.get('roc_auc', 0):.4f}")
        
        st.caption(
            f"Trained on {metadata.get('dataset', 'N/A')} — "
            f"{metadata.get('num_train_samples', '?')} train samples, "
            f"{metadata.get('num_test_samples', '?')} test samples"
        )
        st.markdown("---")
    else:
        st.info("Model metadata not found. Run train_models.py to generate metadata.")
        st.markdown("---")

    # --- Sidebar ---
    with st.sidebar:
        st.header("Experiment Info")
        st.metric("Total Requests", len(df))
        st.metric("Model A Requests", len(df[df["model_variant"] == "A"]))
        st.metric("Model B Requests", len(df[df["model_variant"] == "B"]))

        if results:
            st.markdown("---")
            st.header("Recommendation")
            rec = results.get("recommendation", {})
            st.success(rec.get("recommended_model", "Not available"))
            for reason in rec.get("reasons", []):
                st.caption(f"• {reason}")

    # --- KPI Summary ---
    st.header("📊 Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)

    df_a = df[df["model_variant"] == "A"]
    df_b = df[df["model_variant"] == "B"]

    with col1:
        st.metric(
            "Model A — Avg Latency",
            f"{df_a['latency_ms'].mean():.2f} ms" if not df_a.empty else "N/A",
            delta=None,
        )
    with col2:
        st.metric(
            "Model B — Avg Latency",
            f"{df_b['latency_ms'].mean():.2f} ms" if not df_b.empty else "N/A",
        )
    with col3:
        st.metric(
            "Model A — Positive Rate",
            f"{(df_a['prediction'] == 1).mean():.1%}" if not df_a.empty else "N/A",
        )
    with col4:
        st.metric(
            "Model B — Positive Rate",
            f"{(df_b['prediction'] == 1).mean():.1%}" if not df_b.empty else "N/A",
        )

    st.markdown("---")

    # --- Traffic Distribution ---
    st.header("🔀 Traffic Distribution")
    col_traffic1, col_traffic2 = st.columns(2)

    with col_traffic1:
        variant_counts = df["model_variant"].value_counts()
        fig_pie = px.pie(
            values=variant_counts.values,
            names=variant_counts.index.map(
                {"A": "Model A (Logistic Regression)", "B": "Model B (XGBoost)"}
            ),
            title="Request Distribution by Variant",
            color_discrete_sequence=["#636EFA", "#EF553B"],
            hole=0.4,
        )
        fig_pie.update_layout(margin=dict(t=50, b=20, l=20, r=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_traffic2:
        df["timestamp_parsed"] = pd.to_datetime(df["timestamp"])
        df_grouped = (
            df.groupby([pd.Grouper(key="timestamp_parsed", freq="1min"), "model_variant"])
            .size()
            .reset_index(name="count")
        )
        if not df_grouped.empty:
            fig_time = px.line(
                df_grouped,
                x="timestamp_parsed",
                y="count",
                color="model_variant",
                title="Requests Over Time",
                labels={"timestamp_parsed": "Time", "count": "Requests", "model_variant": "Variant"},
                color_discrete_map={"A": "#636EFA", "B": "#EF553B"},
            )
            fig_time.update_layout(margin=dict(t=50, b=20, l=20, r=20))
            st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.info("Not enough time variation to plot time series.")

    st.markdown("---")

    # --- Latency Analysis ---
    st.header("⏱️ Latency Comparison")
    col_lat1, col_lat2 = st.columns(2)

    with col_lat1:
        fig_box = px.box(
            df,
            x="model_variant",
            y="latency_ms",
            color="model_variant",
            title="Latency Distribution (Box Plot)",
            labels={"latency_ms": "Latency (ms)", "model_variant": "Model Variant"},
            color_discrete_map={"A": "#636EFA", "B": "#EF553B"},
        )
        fig_box.update_layout(showlegend=False, margin=dict(t=50, b=20, l=20, r=20))
        st.plotly_chart(fig_box, use_container_width=True)

    with col_lat2:
        fig_hist_lat = px.histogram(
            df,
            x="latency_ms",
            color="model_variant",
            barmode="overlay",
            title="Latency Distribution (Histogram)",
            labels={"latency_ms": "Latency (ms)", "model_variant": "Variant"},
            color_discrete_map={"A": "#636EFA", "B": "#EF553B"},
            opacity=0.7,
        )
        fig_hist_lat.update_layout(margin=dict(t=50, b=20, l=20, r=20))
        st.plotly_chart(fig_hist_lat, use_container_width=True)

    st.markdown("---")

    # --- Prediction Distribution ---
    st.header("🎯 Prediction Distribution")
    col_pred1, col_pred2 = st.columns(2)

    with col_pred1:
        fig_hist_pred = px.histogram(
            df,
            x="prediction",
            color="model_variant",
            barmode="group",
            title="Prediction Class Distribution",
            labels={"prediction": "Prediction", "model_variant": "Variant"},
            color_discrete_map={"A": "#636EFA", "B": "#EF553B"},
        )
        fig_hist_pred.update_layout(margin=dict(t=50, b=20, l=20, r=20))
        st.plotly_chart(fig_hist_pred, use_container_width=True)

    with col_pred2:
        if "prediction_probability" in df.columns:
            proba_data = df.dropna(subset=["prediction_probability"])
            if not proba_data.empty:
                fig_proba = px.histogram(
                    proba_data,
                    x="prediction_probability",
                    color="model_variant",
                    barmode="overlay",
                    title="Prediction Probability Distribution",
                    labels={
                        "prediction_probability": "Probability",
                        "model_variant": "Variant",
                    },
                    color_discrete_map={"A": "#636EFA", "B": "#EF553B"},
                    opacity=0.7,
                    nbins=30,
                )
                fig_proba.update_layout(margin=dict(t=50, b=20, l=20, r=20))
                st.plotly_chart(fig_proba, use_container_width=True)
            else:
                st.info("No probability data available.")
        else:
            st.info("Prediction probability column not available.")

    st.markdown("---")

    # --- Power Analysis & Sample Size ---
    st.header("🔋 Power Analysis (Sample Size Check)")
    min_samples = 384  # Standard simplified rule for 95% confidence, 5% margin of error
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.metric("Required Samples per Variant", min_samples)
    with col_p2:
        current_min = min(len(df_a), len(df_b))
        has_power = current_min >= min_samples
        st.metric("Current Minimum Variant Samples", current_min, delta="Sufficient Power" if has_power else "Insufficient Power")
    
    if not has_power:
        st.warning(f"⚠️ **Insufficient Data Validation:** You need at least {min_samples} samples per variant for reliable long-term statistical significance. Current minimum is {current_min}.")
    else:
        st.success(f"✅ **Sufficient Data Power:** Both variants have >{min_samples} samples. The statistical tests below are robust.")
    st.markdown("---")

    # --- Statistical Significance ---
    st.header("📈 Statistical Significance Tests")

    if results and "statistical_tests" in results:
        tests = results["statistical_tests"]

        for test_key, test_data in tests.items():
            if isinstance(test_data, dict) and "error" not in test_data:
                col_s1, col_s2, col_s3 = st.columns([2, 1, 3])
                with col_s1:
                    st.subheader(test_data.get("test", test_key))
                    st.caption(f"Metric: {test_data.get('metric', 'N/A')}")
                with col_s2:
                    p_val = test_data.get("p_value", None)
                    if p_val is not None:
                        st.metric("p-value", f"{p_val:.6f}")
                    cohens_d = test_data.get("cohens_d", None)
                    if cohens_d is not None:
                        effect = test_data.get("effect_size_interpretation", "")
                        st.metric("Cohen's d", f"{cohens_d:.4f}", delta=f"{effect} effect")
                with col_s3:
                    if test_data.get("significant", False):
                        st.success(f"✅ {test_data.get('interpretation', '')}")
                    else:
                        st.info(f"ℹ️ {test_data.get('interpretation', '')}")
                        
                    ci = test_data.get("confidence_interval_95", None)
                    if ci is not None:
                        st.caption(f"**95% Confidence Interval (A - B):** [{ci[0]}, {ci[1]}]")
                st.markdown("---")
    else:
        st.warning(
            "No analysis results found. Run the analysis pipeline first:\n\n"
            "`python analysis/run_analysis.py`"
        )

    # --- Detailed Metrics Table ---
    st.header("📋 Detailed Metrics")
    if results:
        metrics_data = []
        for label, key in [("Model A (Logistic Regression)", "model_a"), ("Model B (XGBoost)", "model_b")]:
            m = results.get(key, {})
            metrics_data.append({
                "Model": label,
                "Requests": m.get("count", 0),
                "Mean Prediction": m.get("mean_prediction", "-"),
                "Positive Rate": m.get("positive_prediction_rate", "-"),
                "Mean Latency (ms)": m.get("mean_latency_ms", "-"),
                "P95 Latency (ms)": m.get("p95_latency_ms", "-"),
                "P99 Latency (ms)": m.get("p99_latency_ms", "-"),
            })
        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)

    # --- Raw Data Sample ---
    with st.expander("📦 View Raw Prediction Logs (last 100)"):
        st.dataframe(df.tail(100), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
