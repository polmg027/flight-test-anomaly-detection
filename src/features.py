import pandas as pd


def create_engineered_features(df):
    features = df[
        [
            "altitude_ft",
            "ias_kt",
            "roll_deg",
            "pitch_deg",
            "yaw_rate_dps",
            "nz_g",
            "engine_rpm",
        ]
    ].copy()

    # Preserve information about missing sensor data
    features["yaw_rate_missing"] = (
        df["yaw_rate_dps"].isna().astype(int)
    )

    # Short-term changes
    features["ias_change_1s"] = df["ias_kt"].diff(10)
    features["engine_rpm_change_1s"] = df["engine_rpm"].diff(10)

    # Longer-term altitude trend
    features["altitude_change_10s"] = df["altitude_ft"].diff(100)

    # Smoothed altitude trend
    altitude_smooth = (
        df["altitude_ft"]
        .rolling(window=50, min_periods=1)
        .mean()
    )

    features["altitude_trend_30s"] = (
        altitude_smooth.diff(300) / 30
    )

    # Oscillation intensity over a 2-second window
    features["roll_std_2s"] = (
        df["roll_deg"]
        .rolling(window=20, min_periods=1)
        .std()
    )

    features["yaw_rate_std_2s"] = (
        df["yaw_rate_dps"]
        .rolling(window=20, min_periods=1)
        .std()
    )

    return features