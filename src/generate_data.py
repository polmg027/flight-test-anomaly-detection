from pathlib import Path

import numpy as np
import pandas as pd


def create_base_flight(seed, altitude_base, ias_base, rpm_base):
    rng = np.random.default_rng(seed)

    # 20-minute flight sampled at 10 Hz
    dt = 0.1
    time = np.arange(0, 1200 + dt, dt)
    n = len(time)

    altitude_ft = (
        altitude_base
        + 120 * np.sin(2 * np.pi * 0.002 * time)
        + rng.normal(0, 8, n)
    )

    ias_kt = (
        ias_base
        + 3 * np.sin(2 * np.pi * 0.01 * time)
        + rng.normal(0, 0.8, n)
    )

    roll_deg = rng.normal(0, 0.4, n)

    pitch_deg = (
        2
        + 0.4 * np.sin(2 * np.pi * 0.015 * time)
        + rng.normal(0, 0.15, n)
    )

    yaw_rate_dps = rng.normal(0, 0.08, n)

    nz_g = 1 + rng.normal(0, 0.015, n)

    engine_rpm = (
        rpm_base
        + 40 * np.sin(2 * np.pi * 0.005 * time)
        + rng.normal(0, 15, n)
    )

    df = pd.DataFrame({
        "time_s": time,
        "altitude_ft": altitude_ft,
        "ias_kt": ias_kt,
        "roll_deg": roll_deg,
        "pitch_deg": pitch_deg,
        "yaw_rate_dps": yaw_rate_dps,
        "nz_g": nz_g,
        "engine_rpm": engine_rpm,
    })

    df["anomaly"] = 0
    df["anomaly_type"] = "normal"

    return df


def generate_development_flight():
    df = create_base_flight(
        seed=42,
        altitude_base=10000,
        ias_base=220,
        rpm_base=5200,
    )

    # Airspeed spike
    mask = (df["time_s"] >= 280) & (df["time_s"] <= 282)
    df.loc[mask, "ias_kt"] += 25
    df.loc[mask, ["anomaly", "anomaly_type"]] = [1, "airspeed_spike"]

    # Altitude drift
    mask = (df["time_s"] >= 500) & (df["time_s"] <= 560)
    df.loc[mask, "altitude_ft"] += np.linspace(0, 180, mask.sum())
    df.loc[mask, ["anomaly", "anomaly_type"]] = [1, "altitude_drift"]

    # Lateral oscillation
    mask = (df["time_s"] >= 720) & (df["time_s"] <= 745)
    local_time = df.loc[mask, "time_s"] - 720

    df.loc[mask, "roll_deg"] += (
        5 * np.sin(2 * np.pi * 0.9 * local_time)
    )

    df.loc[mask, "yaw_rate_dps"] += (
        1.5 * np.sin(2 * np.pi * 0.9 * local_time + 0.4)
    )

    df.loc[mask, ["anomaly", "anomaly_type"]] = [1, "lateral_oscillation"]

    # Engine RPM drop
    mask = (df["time_s"] >= 930) & (df["time_s"] <= 940)
    df.loc[mask, "engine_rpm"] -= 900
    df.loc[mask, ["anomaly", "anomaly_type"]] = [1, "engine_rpm_drop"]

    # Sensor dropout
    mask = (df["time_s"] >= 1050) & (df["time_s"] <= 1053)
    df.loc[mask, "yaw_rate_dps"] = np.nan
    df.loc[mask, ["anomaly", "anomaly_type"]] = [1, "sensor_dropout"]

    return df


def generate_validation_flight():
    df = create_base_flight(
        seed=123,
        altitude_base=10020,
        ias_base=219.5,
        rpm_base=5190,
    )

    # Airspeed spike: different time and magnitude
    mask = (df["time_s"] >= 190) & (df["time_s"] <= 192)
    df.loc[mask, "ias_kt"] += 20
    df.loc[mask, ["anomaly", "anomaly_type"]] = [1, "airspeed_spike"]

    # Altitude drift: different interval and magnitude
    mask = (df["time_s"] >= 410) & (df["time_s"] <= 470)
    df.loc[mask, "altitude_ft"] += np.linspace(0, 150, mask.sum())
    df.loc[mask, ["anomaly", "anomaly_type"]] = [1, "altitude_drift"]

    # Lateral oscillation: different frequency and amplitude
    mask = (df["time_s"] >= 660) & (df["time_s"] <= 685)
    local_time = df.loc[mask, "time_s"] - 660

    df.loc[mask, "roll_deg"] += (
        4.5 * np.sin(2 * np.pi * 0.8 * local_time)
    )

    df.loc[mask, "yaw_rate_dps"] += (
        1.2 * np.sin(2 * np.pi * 0.8 * local_time + 0.35)
    )

    df.loc[mask, ["anomaly", "anomaly_type"]] = [1, "lateral_oscillation"]

    # Engine RPM drop
    mask = (df["time_s"] >= 880) & (df["time_s"] <= 892)
    df.loc[mask, "engine_rpm"] -= 750
    df.loc[mask, ["anomaly", "anomaly_type"]] = [1, "engine_rpm_drop"]

    # Sensor dropout
    mask = (df["time_s"] >= 1080) & (df["time_s"] <= 1084)
    df.loc[mask, "yaw_rate_dps"] = np.nan
    df.loc[mask, ["anomaly", "anomaly_type"]] = [1, "sensor_dropout"]

    return df


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"

    development = generate_development_flight()
    validation = generate_validation_flight()

    development_file = DATA_DIR / "development_flight.csv"
    validation_file = DATA_DIR / "validation_flight.csv"

    development.to_csv(development_file, index=False)
    validation.to_csv(validation_file, index=False)

    print("Datasets generated successfully.")

    print("\nDevelopment flight:")
    print(f"Samples: {len(development)}")
    print(f"Anomalous samples: {development['anomaly'].sum()}")

    print("\nValidation flight:")
    print(f"Samples: {len(validation)}")
    print(f"Anomalous samples: {validation['anomaly'].sum()}")

    print(f"\nSaved to: {DATA_DIR}")