import pandas as pd


def extract_ground_truth_events(df):
    anomalous = df[df["anomaly"] == 1].copy()

    events = []

    for anomaly_type, group in anomalous.groupby("anomaly_type"):
        times = group["time_s"].to_numpy()

        start = times[0]
        previous = times[0]

        for current in times[1:]:
            if current - previous > 0.2:
                events.append({
                    "anomaly_type": anomaly_type,
                    "start_s": start,
                    "end_s": previous,
                })
                start = current

            previous = current

        events.append({
            "anomaly_type": anomaly_type,
            "start_s": start,
            "end_s": previous,
        })

    return pd.DataFrame(events)


def evaluate_events(detected_events, true_events):
    detected_true_events = set()
    matched_detected_events = set()

    for true_idx, true_event in true_events.iterrows():

        for detected_idx, detected_event in detected_events.iterrows():

            overlap = (
                detected_event["start_s"] <= true_event["end_s"]
                and
                detected_event["end_s"] >= true_event["start_s"]
            )

            if overlap:
                detected_true_events.add(true_idx)
                matched_detected_events.add(detected_idx)

    true_event_count = len(true_events)
    detected_event_count = len(detected_events)

    true_events_detected = len(detected_true_events)
    matched_detections = len(matched_detected_events)

    event_recall = (
        true_events_detected / true_event_count
        if true_event_count > 0
        else 0
    )

    false_reported_events = (
        detected_event_count - matched_detections
    )

    return {
        "true_events": true_event_count,
        "reported_events": detected_event_count,
        "true_events_detected": true_events_detected,
        "event_recall": event_recall,
        "false_reported_events": false_reported_events,
    }