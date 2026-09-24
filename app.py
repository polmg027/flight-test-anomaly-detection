"""Read-only engineering review of saved, frozen pipeline outputs."""
from pathlib import Path
import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from src.v2.config import Config, SIGNALS
from src.v2.events import truth_events

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT/"reports"
st.set_page_config(page_title="Flight Test Anomaly Detection", page_icon="✈", layout="wide")
st.title("Flight Test Anomaly Detection System")
st.caption("Hybrid engineering + machine-learning analysis of synthetic flight-test data.")
st.info("All flight data shown in this project are synthetic.")


@st.cache_data
def read_csv(path, modified):
    return pd.read_csv(path)


def load(path):
    return read_csv(str(path), path.stat().st_mtime_ns)


available = {"Final Unseen Benchmark": "final_benchmark", "Validation": "validation",
             "Development (in-sample profiles)": "development", "Historical V1 final (already inspected)": "historical"}
available = {k: v for k, v in available.items() if (ROOT/"data/v2"/v).exists()}
if not available:
    st.warning("Run `py main.py` from the repository root to generate the review artifacts.")
    st.stop()
split_label = st.sidebar.selectbox("Evaluation split", list(available))
split = available[split_label]
files = sorted((ROOT/"data/v2"/split).glob("*.csv"))
flight_id = st.sidebar.selectbox("Flight", [p.stem for p in files])
flight = load(ROOT/"data/v2"/split/f"{flight_id}.csv")
evaluation_mode = st.sidebar.checkbox("Show ground truth (evaluation mode)", value=True)
st.sidebar.caption("Offline review. Ground truth is used for evaluation only; it does not drive inference or event consolidation.")
if split == "historical":
    events = load(REPORTS/"historical_events.csv")
    metrics = load(REPORTS/"v1_recalculated.csv")
    m = metrics[(metrics.model == "V1 hybrid") & (metrics.dataset == "V1 historical final")].iloc[0]
else:
    events = load(REPORTS/f"{split}_events.csv")
    events = events[events.flight_id.astype(str).eq(flight_id)]
    metrics = load(REPORTS/f"{split}_metrics.csv")
    m = metrics[metrics.flight_id.astype(str).eq(flight_id)].iloc[0]
st.subheader(f"{split_label} · Flight {flight_id}")
if split == "historical":
    st.caption("V1 sample predictions are preserved. These event reports use V2 consolidation and one-to-one matching; original V1 reporting is archived in reports/v1/.")
cards = [("Samples", f"{len(flight):,}"), ("Flagged samples", f"{int(flight.predicted_anomaly.sum()):,}"),
         ("Reported events", str(len(events)))]
if evaluation_mode:
    cards += [("Precision", f"{m.precision:.3f}"), ("Recall", f"{m.recall:.3f}"), ("F1", f"{m.f1:.3f}")]
for col, (name, value) in zip(st.columns(len(cards)), cards):
    col.metric(name, value)
if evaluation_mode:
    st.caption(f"Event recall: {m.event_recall:.1%} · False reports: {int(m.false_reported_events)} · "
               "Undefined metrics on normal-only flights are shown as zero.")
signal_tab, event_tab, performance_tab, method_tab = st.tabs(["Signal review", "Event evidence", "Performance", "Method & provenance"])
with signal_tab:
    signals = st.multiselect("Signals", [s for s in SIGNALS if s in flight], default=["ias_kt", "altitude_ft", "roll_deg", "engine_rpm"])
    selected_event = st.selectbox("Focus on event", ["Full flight"] + [str(int(v)) for v in events.event_id])
    bounds = [float(flight.time_s.min()), float(flight.time_s.max())]
    if selected_event != "Full flight":
        e = events[events.event_id.eq(int(selected_event))].iloc[0]
        bounds = [max(bounds[0], float(e.start_s)-10), min(bounds[1], float(e.end_s)+10)]
        st.caption(f"Evidence: {e.reason} · Sources: {e.detection_source}")
    if signals:
        fig = make_subplots(rows=len(signals), cols=1, shared_xaxes=True, vertical_spacing=.035, subplot_titles=signals)
        truth = truth_events(flight, Config()) if evaluation_mode else pd.DataFrame()
        for i, signal in enumerate(signals, 1):
            fig.add_trace(go.Scattergl(x=flight.time_s, y=flight[signal], name=signal, line={"color": "#234b6c", "width": 1}), row=i, col=1)
            points = flight[flight.predicted_anomaly.eq(1)]
            fig.add_trace(go.Scattergl(x=points.time_s, y=points[signal], mode="markers", name="Flagged samples",
                                      marker={"color": "#be7924", "size": 3}, showlegend=i == 1), row=i, col=1)
            for e in truth.itertuples():
                fig.add_vrect(x0=e.start_s, x1=e.end_s, fillcolor="#168a83", opacity=.12, line_width=0, row=i, col=1)
            for e in events.itertuples():
                fig.add_vrect(x0=e.start_s, x1=e.end_s, fillcolor="rgba(0,0,0,0)", line_color="#d28b28", line_width=1, row=i, col=1)
        fig.update_xaxes(range=bounds, title_text="Time [s]", row=len(signals), col=1)
        fig.update_layout(height=max(400, len(signals)*190), template="plotly_white", margin={"t": 30, "b": 30}, showlegend=False)
        st.caption("Amber points: sample alerts · Amber outlines: reported events · Teal shading: synthetic ground truth")
        st.plotly_chart(fig, width="stretch")
with event_tab:
    st.write("Reportable intervals retain the strongest signal evidence. Descriptions are observations, not aircraft fault diagnoses.")
    display_events = events.drop(columns="flight_id", errors="ignore").round(
        {"start_s": 1, "end_s": 1, "duration_s": 1, "ml_score": 3,
         "statistical_score": 3, "fused_score": 3})
    st.dataframe(display_events, hide_index=True, width="stretch")
    st.download_button("Download this flight's event report", events.to_csv(index=False), file_name=f"flight_{flight_id}_events.csv", mime="text/csv")
    if not events.empty:
        event_id = st.selectbox("Inspect evidence for event", events.event_id.tolist())
        e = events[events.event_id.eq(event_id)].iloc[0]
        st.write(f"**{e.reason}**")
        st.write(f"Signals: {e.primary_signals}. Source: {e.detection_source}.")
        st.write(f"Triggered rules: {e.triggered_rules if pd.notna(e.triggered_rules) else 'None'}")
        evidence = flight[flight.time_s.between(e.start_s-5, e.end_s+5)]
        fig = go.Figure()
        for score in ["ml_score", "statistical_score", "fused_score"]:
            fig.add_trace(go.Scatter(x=evidence.time_s, y=evidence[score], name=score))
        fig.update_layout(template="plotly_white", xaxis_title="Time [s]", yaxis_title="Normalized evidence (not probability)")
        st.plotly_chart(fig, width="stretch")
with performance_tab:
    if not evaluation_mode:
        st.info("Enable evaluation mode to view label-based performance.")
    else:
        st.write("Sample metrics evaluate timestamps. Event metrics evaluate episodes with one-to-one temporal matching; duplicate reports are penalized.")
        st.dataframe(pd.DataFrame([[int(m.tn), int(m.fp)], [int(m.fn), int(m.tp)]],
                                  index=["True normal", "True anomaly"], columns=["Predicted normal", "Predicted anomaly"]))
        if split != "historical":
            types = load(REPORTS/f"{split}_types.csv")
            st.dataframe(types[types.flight_id.astype(str).eq(flight_id)].drop(columns="flight_id"), hide_index=True)
        st.subheader("Model progression")
        comparison = load(REPORTS/"model_comparison.csv")
        v1 = comparison[comparison.dataset.eq("V1 validation")]
        v2 = comparison[comparison.dataset.eq("V2 validation")].sort_values("objective", ascending=False).drop_duplicates(["model", "mode"])
        st.caption("V1 and V2 use different datasets and sensor assumptions. These tables show development history, not a controlled head-to-head benchmark.")
        for label, df in [("V1 validation", v1), ("V2 validation — best threshold per candidate", v2)]:
            st.write(label)
            st.dataframe(df[["model", "mode", "precision", "recall", "f1", "event_precision", "event_recall", "false_reported_events"]], hide_index=True)
        if (REPORTS/"final_benchmark_summary.json").exists():
            st.subheader("Aggregate final benchmark · eight flights")
            summary = json.loads((REPORTS/"final_benchmark_summary.json").read_text())
            st.json(summary, expanded=False)
            st.image(str(REPORTS/"figures/anomaly_type_detection_final.png"))
with method_tab:
    st.markdown("""**Normal-only fitting → validation selection → detector freeze → final benchmark.**

    Eight compact temporal/spectral features use robust local residuals, band-limited
    roll/yaw RMS and altitude/vertical-speed consistency. Missingness remains explicit.
    Final scores are evidence, not calibrated fault probabilities.

    Centered windows, interpolation and zero-phase filtering use future samples.
    This is an offline analysis tool with simplified synthetic dynamics, not a
    real-time or safety-certified aircraft system. V2 drift detection assumes an
    independent vertical-speed measurement unavailable to the original V1.
    """)
    if (REPORTS/"freeze_manifest.json").exists():
        st.json(json.loads((REPORTS/"freeze_manifest.json").read_text()), expanded=False)
