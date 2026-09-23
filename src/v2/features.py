"""Compact offline features. No label, seed, phase or injection-time inputs."""
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from .config import Config, SIGNALS


def validate_flight(df: pd.DataFrame, config: Config):
    required = {"time_s", *SIGNALS}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing sensor columns: {sorted(required - set(df.columns))}")
    if len(df) < 32 or not np.allclose(np.diff(df.time_s), 1/config.sample_rate_hz, atol=1e-7):
        raise ValueError("Flight must have >=32 rows on a regular increasing configured time grid.")
    if not np.isfinite(df.time_s).all():
        raise ValueError("Time must be finite.")
    if np.isinf(df[list(SIGNALS)].to_numpy(dtype=float)).any():
        raise ValueError("Infinite sensor values are invalid; represent missing readings as NaN.")


def build_features(df: pd.DataFrame, config: Config = Config()) -> pd.DataFrame:
    validate_flight(df, config)
    fs = config.sample_rate_hz
    def window(seconds):
        return max(3, round(seconds*fs) | 1)
    # Interpolation is local offline preprocessing, not a learned test-set fit.
    x = df[list(SIGNALS)].interpolate(limit_direction="both").fillna(0)
    f = pd.DataFrame(index=df.index)
    for signal, seconds, name in [("ias_kt", config.ias_window_s, "ias_residual"),
                                  ("engine_rpm", config.rpm_window_s, "rpm_residual")]:
        baseline = x[signal].rolling(window(seconds), center=True, min_periods=1).median()
        f[name] = (x[signal] - baseline).abs()
    # Endpoint averages and integrated independent vertical speed over the same interval.
    smooth = x.altitude_ft.rolling(window(2), center=True, min_periods=1).mean()
    half = round(config.trend_window_s*fs/2)
    measured = (smooth.shift(-half) - smooth.shift(half)) / config.trend_window_s
    reference = x.vertical_speed_fps.rolling(2*half+1, center=True, min_periods=1).mean()
    f["altitude_consistency"] = (measured-reference).abs().fillna(0)
    sos = butter(3, [config.band_low_hz, config.band_high_hz], btype="bandpass", fs=fs, output="sos")
    for signal in ("roll_deg", "yaw_rate_dps"):
        band = sosfiltfilt(sos, x[signal].to_numpy())
        f[signal + "_band_rms"] = np.sqrt(pd.Series(band**2, index=df.index).rolling(
            window(config.energy_window_s), center=True, min_periods=1).mean())
    for signal in ("pitch_deg", "nz_g"):
        local = x[signal].rolling(window(10), center=True, min_periods=1).median()
        f[signal + "_residual"] = (x[signal]-local).abs()
    f["missing_any"] = df[list(SIGNALS)].isna().any(axis=1).astype(float)
    return f
