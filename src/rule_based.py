import pandas as pd


def engineering_rules(df):
    rules = pd.DataFrame(index=df.index)

    # 1. Missing sensor data
    rules["sensor_dropout"] = (
        df["yaw_rate_dps"].isna()
    )

    # 2. IAS spike relative to local behaviour
    ias_local = (
        df["ias_kt"]
        .rolling(window=50, center=True, min_periods=1)
        .median()
    )

    rules["airspeed_spike"] = (
        (df["ias_kt"] - ias_local).abs() > 8
    )

    # 3. Engine RPM sudden drop
    rpm_local = (
        df["engine_rpm"]
        .rolling(window=100, center=True, min_periods=1)
        .median()
    )

    rules["engine_rpm_drop"] = (
        df["engine_rpm"] < rpm_local - 300
    )

    # 4. Sustained altitude trend
    altitude_smooth = (
        df["altitude_ft"]
        .rolling(window=50, min_periods=1)
        .mean()
    )

    altitude_rate = altitude_smooth.diff(300) / 30

    rules["altitude_drift"] = (
        altitude_rate.abs() > 2.0
    )

    # If any engineering rule is triggered
    rules["rule_anomaly"] = rules.any(axis=1).astype(int)

    return rules

def combine_with_ml(ml_results, rules):
    hybrid = ml_results.copy()

    hybrid["rule_anomaly"] = rules["rule_anomaly"]

    hybrid["predicted_anomaly"] = (
        (hybrid["predicted_anomaly"] == 1) |
        (hybrid["rule_anomaly"] == 1)
    ).astype(int)

    return hybrid