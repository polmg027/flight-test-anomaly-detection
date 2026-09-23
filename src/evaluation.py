from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def evaluate_model(results):
    y_true = results["anomaly"]
    y_pred = results["predicted_anomaly"]

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
    }

    return metrics

def detection_by_anomaly_type(results):
    anomalous_data = results[results["anomaly"] == 1]

    summary = (
        anomalous_data
        .groupby("anomaly_type")["predicted_anomaly"]
        .agg(["count", "sum"])
    )

    summary["detection_rate"] = summary["sum"] / summary["count"]

    return summary