"""Sample and event summaries; micro aggregates do not average ratios."""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from .config import ANOMALIES, Config
from .events import consolidate, truth_events, match_events


def ratios(tp, fp, fn):
    p = tp/(tp+fp) if tp+fp else 0.
    r = tp/(tp+fn) if tp+fn else 0.
    return p, r, 2*p*r/(p+r) if p+r else 0.


def evaluate(flight, prediction, config: Config = Config()):
    tn, fp, fn, tp = map(int, confusion_matrix(flight.anomaly, prediction.predicted_anomaly, labels=[0, 1]).ravel())
    precision, recall, f1 = ratios(tp, fp, fn)
    events = consolidate(flight, prediction, config)
    metrics = {"samples": len(flight), "precision": precision, "recall": recall, "f1": f1,
               "tp": tp, "fp": fp, "fn": fn, "tn": tn,
               "duration_h": len(flight)/config.sample_rate_hz/3600,
               **match_events(events, truth_events(flight, config), config.min_iou)}
    metrics["false_events_per_hour"] = metrics["false_reported_events"]/metrics["duration_h"]
    types = []
    for kind in ANOMALIES:
        mask = flight.anomaly_type.eq(kind).to_numpy()
        count = int(mask.sum())
        detected = int(prediction.predicted_anomaly.to_numpy()[mask].sum())
        types.append({"anomaly_type": kind, "samples": count, "detected": detected,
                      "recall": detected/count if count else np.nan})
    return metrics, events, pd.DataFrame(types)


def aggregate(metrics: pd.DataFrame):
    keys = ["samples", "tp", "fp", "fn", "tn", "true_events", "reported_events", "matched_events",
            "missed_events", "false_reported_events", "unmatched_overlapping_reports"]
    result = {k: int(metrics[k].sum()) for k in keys}
    result["precision"], result["recall"], result["f1"] = ratios(result["tp"], result["fp"], result["fn"])
    result["event_precision"], result["event_recall"], result["event_f1"] = ratios(
        result["matched_events"], result["false_reported_events"], result["missed_events"])
    result["duration_h"] = float(metrics.duration_h.sum())
    result["false_events_per_hour"] = result["false_reported_events"]/result["duration_h"]
    result["duplicate_detection_rate"] = result["unmatched_overlapping_reports"]/max(1, result["reported_events"])
    weights = metrics.matched_events
    for key in ("mean_matched_iou", "mean_signed_onset_error_s"):
        result[key] = float((metrics[key].fillna(0)*weights).sum()/weights.sum()) if weights.sum() else None
    result["per_flight_f1_min"] = float(metrics.f1.min())
    result["per_flight_f1_max"] = float(metrics.f1.max())
    return result
