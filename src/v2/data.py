"""Illustrative profiles, not a flight-dynamics or aircraft-performance model."""
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from .config import ANOMALIES, Config


def generate_flight(seed: int, config: Config = Config(), anomalies: bool = True) -> pd.DataFrame:
    """Generate an exclusive-end regular time grid and nonoverlapping injections.

    Noise, operating point, phase boundaries, injection order and severity vary.
    Separate RNG streams keep normal profiles identical in development twins.
    """
    if config.duration_s < 240 or config.sample_rate_hz <= 4:
        raise ValueError("Require duration >=240 s and sample rate >4 Hz.")
    rng = np.random.default_rng(seed)
    t = np.arange(round(config.duration_s * config.sample_rate_hz)) / config.sample_rate_hz
    n, fs = len(t), config.sample_rate_hz
    knots = np.array([0, .18, .36, .55, .73, .86, 1.0]) * config.duration_s
    knots[1:-1] += rng.uniform(-.025, .025, 5) * config.duration_s
    rates = rng.uniform(.75, 1.25) * np.array([8, 0, 0, 1.5, 0, -7, -4])
    rate = gaussian_filter1d(np.interp(t, knots, rates), fs * 4)
    noise = rng.uniform(.7, 1.65)
    altitude = rng.uniform(7000, 14000) + np.cumsum(rate) / fs
    speed = rng.uniform(185, 260) + np.interp(t, knots, rng.uniform(-12, 12, 7))
    roll = rng.uniform(3, 12) * np.sin(2*np.pi*t / rng.uniform(80, 150))
    roll *= np.exp(-((t - knots[3]) / (config.duration_s * .18))**2)
    rpm = rng.uniform(4600, 5600) + 12*rate + 60*np.sin(2*np.pi*t/160)
    df = pd.DataFrame({
        "time_s": t,
        "altitude_ft": altitude + rng.normal(0, 4*noise, n),
        "ias_kt": speed + 1.5*np.sin(t/16) + rng.normal(0, .55*noise, n),
        "roll_deg": roll + rng.normal(0, .25*noise, n),
        "pitch_deg": 2 + .22*rate + rng.normal(0, .12*noise, n),
        "yaw_rate_dps": .07*roll + rng.normal(0, .06*noise, n),
        "nz_g": 1 + .0006*roll**2 + rng.normal(0, .012*noise, n),
        "engine_rpm": rpm + rng.normal(0, 12*noise, n),
        # Independent imperfect reference: bias, slow mismatch and measurement noise.
        "vertical_speed_fps": rate + rng.normal(0, .18) + .25*np.sin(t/43)
                              + rng.normal(0, .35*noise, n),
        "phase": np.select([t < knots[1], t < knots[2], t < knots[4], t < knots[5]],
                           ["climb", "cruise", "gentle_manoeuvre", "level"], "descent"),
        "anomaly": np.zeros(n, dtype=int), "anomaly_type": "normal",
    })
    if not anomalies:
        return df
    arng = np.random.default_rng(seed + 50000)
    order = arng.permutation(ANOMALIES)
    for k, kind in enumerate(order):
        start = config.duration_s * (.08 + .175*k) + arng.uniform(-7, 7)
        duration = {"airspeed_spike": arng.uniform(.3, 3),
                    "altitude_drift": arng.uniform(30, 65),
                    "lateral_oscillation": arng.uniform(12, 28),
                    "engine_rpm_drop": arng.uniform(5, 17),
                    "sensor_dropout": arng.uniform(.5, 5)}[kind]
        # Scale durations for short test fixtures while preserving five separate events.
        duration *= min(1, config.duration_s / 600)
        mask = (t >= start) & (t < start + duration)
        u = (t[mask] - start) / duration
        if kind == "airspeed_spike":
            df.loc[mask, "ias_kt"] += arng.choice([-1, 1]) * arng.uniform(5, 24)
        elif kind == "altitude_drift":
            # Gradual ramp and recovery; no artificial instantaneous reset to exploit.
            envelope = np.where(u < .75, u/.75, (1-u)/.25)
            df.loc[mask, "altitude_ft"] += arng.choice([-1, 1]) * arng.uniform(35, 150) * envelope
        elif kind == "lateral_oscillation":
            wave = 2*np.pi*arng.uniform(.35, 1.35)*(t[mask]-start)
            taper = np.minimum(1, np.minimum(u, 1-u)*10)
            df.loc[mask, "roll_deg"] += arng.uniform(1.2, 5) * np.sin(wave)*taper
            df.loc[mask, "yaw_rate_dps"] += arng.uniform(.25, 1.2)*np.sin(wave+.4)*taper
        elif kind == "engine_rpm_drop":
            df.loc[mask, "engine_rpm"] -= arng.uniform(140, 800)
        else:
            df.loc[mask, arng.choice(["yaw_rate_dps", "ias_kt", "engine_rpm"])] = np.nan
        df.loc[mask, "anomaly"] = 1
        df.loc[mask, "anomaly_type"] = kind
    return df


def make_split(seeds, config: Config, normal_count: int = 0):
    return {str(s): generate_flight(s, config, i < len(seeds)-normal_count)
            for i, s in enumerate(seeds)}
