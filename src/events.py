import pandas as pd


RULE_COLUMNS = [
    "sensor_dropout",
    "airspeed_spike",
    "engine_rpm_drop",
    "altitude_drift",
]


def extract_anomaly_events(
    results,
    rules,
    max_gap_s=0.5,
    min_ml_samples=3,
):
    anomaly_times = (
        results.loc[
            results["predicted_anomaly"] == 1,
            "time_s"
        ]
        .to_numpy()
    )

    if len(anomaly_times) == 0:
        return pd.DataFrame()

    # Group nearby anomalous samples
    groups = []
    event_start = anomaly_times[0]
    previous_time = anomaly_times[0]

    for current_time in anomaly_times[1:]:
        if current_time - previous_time > max_gap_s:
            groups.append((event_start, previous_time))
            event_start = current_time

        previous_time = current_time

    groups.append((event_start, previous_time))

    events = []

    for start, end in groups:
        event_mask = (
            (results["time_s"] >= start) &
            (results["time_s"] <= end)
        )

        anomalous_samples = int(
            results.loc[
                event_mask,
                "predicted_anomaly"
            ].sum()
        )

        active_rules = [
            rule
            for rule in RULE_COLUMNS
            if rules.loc[event_mask, rule].any()
        ]

        if active_rules:
            detection_source = " + ".join(active_rules)

        else:
            detection_source = "ml_multivariate"

            # Ignore isolated ML detections in event reporting
            if anomalous_samples < min_ml_samples:
                continue

        events.append({
            "start_s": start,
            "end_s": end,
            "duration_s": end - start,
            "anomalous_samples": anomalous_samples,
            "detection_source": detection_source,
        })

    events_df = pd.DataFrame(events)

    if not events_df.empty:
        events_df.insert(
            0,
            "event_id",
            range(1, len(events_df) + 1)
        )

    return events_df