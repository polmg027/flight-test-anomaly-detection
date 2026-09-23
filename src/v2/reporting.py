"""Rebuildable portfolio artifacts; presentation does not change predictions."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from .events import truth_events

BLUE, TEAL, AMBER, RED = "#234b6c", "#168a83", "#d28b28", "#b85151"


def save(fig, directory, name):
    fig.savefig(directory/name, bbox_inches="tight", dpi=160)
    plt.close(fig)


def timeline(flight, events, directory, config, flight_id):
    fig, axes = plt.subplots(6, 1, figsize=(13, 10), sharex=True,
                             gridspec_kw={"height_ratios": [.65, 1, 1, 1, 1, 1]}, layout="constrained")
    truth = truth_events(flight, config)
    for e in truth.itertuples():
        axes[0].broken_barh([(e.start_s, e.end_s-e.start_s)], (.65, .25), facecolors=TEAL)
    for e in events.itertuples():
        axes[0].broken_barh([(e.start_s, e.end_s-e.start_s)], (.1, .25), facecolors=AMBER)
    axes[0].set_yticks([.775, .225], ["Ground truth", "Reported"])
    axes[0].set_ylim(0, 1)
    axes[0].set_title(f"Flight {flight_id}  |  Frozen V2 benchmark review", loc="left", pad=16)
    for ax, signal, unit in zip(axes[1:], ["ias_kt", "altitude_ft", "roll_deg", "yaw_rate_dps", "engine_rpm"],
                               ["IAS [kt]", "Altitude [ft]", "Roll [deg]", "Yaw rate [deg/s]", "Engine [rpm]"]):
        ax.plot(flight.time_s, flight[signal], color=BLUE, lw=.7)
        ax.set_ylabel(unit)
        for e in truth.itertuples():
            ax.axvspan(e.start_s, e.end_s, color=TEAL, alpha=.12, lw=0)
        for e in events.itertuples():
            ax.axvspan(e.start_s, e.end_s, facecolor="none", edgecolor=AMBER, lw=.7, hatch="//", alpha=.6)
    axes[1].legend(handles=[Patch(color=TEAL, alpha=.25, label="Synthetic ground truth"),
                            Patch(facecolor="none", edgecolor=AMBER, hatch="//", label="Reported interval")],
                   loc="upper right", ncol=2, fontsize=8)
    axes[-1].set_xlabel("Time [s]  ·  Synthetic data only  ·  Offline analysis")
    save(fig, directory, "final_test_anomaly_timeline.png")


def create_figures(directory, summary, metrics, types, comparison, flight, events, config, flight_id):
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#344554", "text.color": "#253744",
                         "axes.titleweight": "bold", "figure.facecolor": "white",
                         "axes.grid": True, "grid.alpha": .15})
    timeline(flight, events, directory, config, flight_id)
    fig, ax = plt.subplots(figsize=(6, 4.5), layout="constrained")
    matrix = np.array([[summary["tn"], summary["fp"]], [summary["fn"], summary["tp"]]])
    ax.imshow(np.log1p(matrix), cmap="Blues")
    for (i, j), count in np.ndenumerate(matrix):
        ax.text(j, i, f"{count:,}", ha="center", va="center", fontsize=19, color="white" if np.log1p(count)>7 else BLUE)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"],
           xlabel="Predicted", ylabel="True", title="Final benchmark · sample counts")
    ax.grid(False)
    save(fig, directory, "confusion_matrix_final.png")
    grouped = types.groupby("anomaly_type")[["samples", "detected"]].sum()
    rates = grouped.detected/grouped.samples
    fig, ax = plt.subplots(figsize=(8, 4.2), layout="constrained")
    bars = ax.barh([s.replace("_", " ") for s in rates.index], rates, color=TEAL)
    ax.bar_label(bars, labels=[f"{v:.1%}  ({int(grouped.loc[k, 'detected'])}/{int(grouped.loc[k, 'samples'])})" for k, v in rates.items()], padding=6)
    ax.set(xlim=(0, 1.35), xlabel="Detected anomalous samples / labeled samples", title="Final benchmark · recall by anomaly family")
    save(fig, directory, "anomaly_type_detection_final.png")
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    vals = [summary[k] for k in ("true_events", "matched_events", "missed_events", "false_reported_events")]
    bars = ax.bar(["True", "Matched", "Missed", "False reports"], vals, color=[BLUE, TEAL, RED, AMBER])
    ax.bar_label(bars, padding=4)
    ax.set(ylim=(0, max(vals)*1.2+1), ylabel="Events", title="Final benchmark · one-to-one matching (IoU ≥ 0.10)")
    save(fig, directory, "event_level_performance.png")
    historical = comparison[comparison.dataset.eq("V1 validation")]
    candidates = comparison[comparison.dataset.eq("V2 validation")].sort_values("objective", ascending=False).drop_duplicates(["model", "mode"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.7), layout="constrained", sharex=True)
    for ax, df, title in zip(axes, [historical, candidates], ["Historical V1 validation", "V2 multi-flight validation"]):
        labels = [str(row.model).replace("local_outlier_factor", "LOF").replace("isolation_forest", "IF") +
                  (" / "+str(row["mode"]) if pd.notna(row.get("mode")) else "") for _, row in df.iterrows()]
        y = np.arange(len(df))
        for shift, metric, color in [(-.23, "precision", BLUE), (0, "recall", AMBER), (.23, "f1", TEAL)]:
            ax.barh(y+shift, df[metric], height=.22, label=metric, color=color)
        ax.set(yticks=y, yticklabels=labels, xlim=(0, 1.04), title=title, xlabel="Sample-level metric")
        ax.invert_yaxis()
    axes[1].legend(loc="lower right", ncol=3)
    fig.suptitle("Model progression · different datasets; not a controlled V1–V2 comparison", fontsize=12)
    save(fig, directory, "model_comparison.png")
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    x = np.arange(len(metrics))
    ax.bar(x-.18, metrics.precision, .35, label="Precision", color=BLUE)
    ax.bar(x+.18, metrics.recall, .35, label="Recall", color=TEAL)
    labels = [str(r.flight_id) + ("\nnormal only" if r.true_events == 0 else "") for r in metrics.itertuples()]
    ax.set(xticks=x, xticklabels=labels, ylim=(0, 1.1), title="Final benchmark · flight variability", ylabel="Sample-level metric")
    ax.legend(ncol=2)
    save(fig, directory, "per_flight_performance.png")


def markdown_table(df):
    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v).replace("|", "/")
    return "| " + " | ".join(df.columns) + " |\n| " + " | ".join(["---"]*len(df.columns)) + " |\n" + "\n".join(
        "| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False, name=None))


def write_report(root, manifest, summary, metrics, events, types):
    report = root/"reports"
    grouped = types.groupby("anomaly_type", as_index=False)[["samples", "detected"]].sum()
    grouped["recall"] = grouped.detected/grouped.samples
    text = f"""# V2 final benchmark — automated engineering report

All data are synthetic, unrelated to Airbus or any real aircraft/test campaign.
Version {manifest['config']['version']}; freeze: {manifest['frozen_at_utc']}.
Validation-selected configuration: `{json.dumps(manifest['selection'])}`.

Eight independent 600-second flights at 10 Hz; six with five injections each,
two normal-only controls. The timeline uses the first predetermined benchmark
flight, not a selected best case.

## Sample and event metrics

{markdown_table(pd.DataFrame([summary]).T.reset_index().rename(columns={'index': 'metric', 0: 'value'}))}

## Per-flight results

{markdown_table(metrics[['flight_id', 'precision', 'recall', 'f1', 'event_precision', 'event_recall', 'false_reported_events', 'missed_events']])}

Undefined precision/recall/F1 are zero, including normal-only flights with no
alerts. Interpret these controls using false reports and true negatives.
Aggregate ratios sum counts first.

## Anomaly-family recall

{markdown_table(grouped)}

## Reported events

{markdown_table(events[['flight_id', 'event_id', 'start_s', 'end_s', 'reason', 'detection_source']])}

## Interpretation and limitations

One-to-one matching maximizes eligible match count then IoU; minimum IoU is
{manifest['config']['min_iou']}. Extra fragments remain false reports. Half-open
intervals include the final sample period. Consolidation does not change sample
predictions. Signed onset error can be negative because centered offline features
use future data; it is not operational detection latency.

Normal-only development fits reference scales, imputation, scaling and novelty
models. Validation labels select thresholds and fusion: a semi-supervised protocol.
Final data are generated after freezing source and fitted-state hashes. Later runs
verify original data, score and prediction hashes against the first receipt.

Altitude consistency depends on an independent noisy vertical-speed reference.
This extra sensor makes V2 a different task from V1; metrics do not establish a
controlled improvement on identical data. Shared generator assumptions, limited
anomaly families and synthetic noise restrict external validity. No real-aircraft,
operational, streaming or safety-critical validation has been performed.
"""
    (report/"technical_report.md").write_text(text, encoding="utf-8")
    (report/"final_test_summary.txt").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    grouped.to_csv(report/"anomaly_type_detection_final.csv", index=False)
