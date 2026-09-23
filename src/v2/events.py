"""Label-free consolidation and one-to-one event evaluation on half-open intervals."""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from .config import Config

EVENT_COLUMNS = ["event_id", "start_s", "end_s", "duration_s", "anomalous_samples",
                 "detection_source", "primary_signals", "triggered_rules", "reason",
                 "ml_score", "statistical_score", "fused_score"]
TRUTH_COLUMNS = ["anomaly_type", "start_s", "end_s"]


def consolidate(flight: pd.DataFrame, predictions: pd.DataFrame, config: Config = Config()):
    """Bridge <=max_gap_s of unflagged time; keep rule alerts of any duration.

    Minimum duration counts flagged samples, not the span of bridged gaps.
    Raw sample predictions are never modified.
    """
    dt = 1/config.sample_rate_hz
    indices = np.flatnonzero(predictions.predicted_anomaly.to_numpy())
    if not len(indices):
        return pd.DataFrame(columns=EVENT_COLUMNS)
    groups = np.split(indices, np.flatnonzero(np.diff(indices)*dt > config.max_gap_s+dt+1e-8)+1)
    rows = []
    for group in groups:
        p = predictions.iloc[group]
        if len(group)*dt < config.min_event_s and not p.rule_triggered.any():
            continue
        peak = p.loc[p.fused_score.idxmax()]
        rows.append({
            "event_id": len(rows)+1, "start_s": float(flight.time_s.iloc[group[0]]),
            "end_s": float(flight.time_s.iloc[group[-1]]+dt),
            "duration_s": float((group[-1]-group[0]+1)*dt), "anomalous_samples": len(group),
            "detection_source": "; ".join(sorted(p.detection_source.unique())),
            "primary_signals": "; ".join(sorted(p.primary_signals.unique())),
            "triggered_rules": "; ".join(sorted(p.loc[p.rule_triggered, "reason"].unique())),
            "reason": peak.reason, "ml_score": float(p.ml_score.max()),
            "statistical_score": float(p.statistical_score.max()), "fused_score": float(p.fused_score.max()),
        })
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def truth_events(flight: pd.DataFrame, config: Config = Config()):
    dt = 1/config.sample_rate_hz
    types = flight.anomaly_type.to_numpy()
    flags = flight.anomaly.to_numpy().astype(bool)
    changes = np.r_[0, np.flatnonzero((types[1:] != types[:-1]) | (flags[1:] != flags[:-1]))+1, len(flight)]
    rows = []
    for a, b in zip(changes[:-1], changes[1:]):
        if flags[a]:
            rows.append({"anomaly_type": types[a], "start_s": float(flight.time_s.iloc[a]),
                         "end_s": float(flight.time_s.iloc[b-1]+dt)})
    return pd.DataFrame(rows, columns=TRUTH_COLUMNS)


def match_events(reported: pd.DataFrame, truth: pd.DataFrame, min_iou: float = .1):
    """Maximum-cardinality eligible matching, then greatest total IoU.

    IoU >=0.1 requires nontrivial overlap; duplicates and merged truth episodes
    cannot earn multiple true positives. Unmatched duplicates count as false reports.
    Signed onset error is descriptive in offline analysis, not online latency.
    """
    n, m = len(reported), len(truth)
    iou = np.zeros((n, m))
    for i, r in enumerate(reported.itertuples()):
        for j, t in enumerate(truth.itertuples()):
            overlap = max(0, min(r.end_s, t.end_s)-max(r.start_s, t.start_s))
            union = max(r.end_s, t.end_s)-min(r.start_s, t.start_s)
            iou[i, j] = overlap/union if union else 0
    pairs = []
    if n and m:
        eligible = (iou >= min_iou) & (iou > 0)
        # Cardinality bonus exceeds any possible change in the total IoU.
        a, b = linear_sum_assignment(-(eligible*(min(n, m)+1)+iou*eligible))
        pairs = [(int(i), int(j)) for i, j in zip(a, b) if eligible[i, j]]
    hits = len(pairs)
    precision, recall = hits/n if n else 0., hits/m if m else 0.
    duplicate = sum(bool((iou[i] > 0).any()) for i in range(n) if i not in {p[0] for p in pairs})
    offsets = [float(reported.iloc[i].start_s-truth.iloc[j].start_s) for i, j in pairs]
    return {
        "true_events": m, "reported_events": n, "matched_events": hits,
        "missed_events": m-hits, "false_reported_events": n-hits,
        "event_precision": precision, "event_recall": recall,
        "event_f1": 2*precision*recall/(precision+recall) if precision+recall else 0.,
        "unmatched_overlapping_reports": duplicate,
        "duplicate_detection_rate": duplicate/n if n else 0.,
        "mean_matched_iou": float(np.mean([iou[i, j] for i, j in pairs])) if pairs else None,
        "mean_signed_onset_error_s": float(np.mean(offsets)) if offsets else None,
    }
