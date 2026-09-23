"""Experimental protocol fixed before final data generation."""
from dataclasses import asdict, dataclass

DEVELOPMENT_SEEDS = (101, 113, 127, 139, 151, 163, 179, 191)
VALIDATION_SEEDS = (307, 311, 331, 347, 359, 373)
# Predetermined here; pipeline does not generate these until the freeze exists.
BENCHMARK_SEEDS = (1009, 1031, 1061, 1091, 1151, 1181, 1213, 1237)
SIGNALS = ("altitude_ft", "ias_kt", "roll_deg", "pitch_deg", "yaw_rate_dps",
           "nz_g", "engine_rpm", "vertical_speed_fps")
ANOMALIES = ("airspeed_spike", "altitude_drift", "lateral_oscillation",
             "engine_rpm_drop", "sensor_dropout")


@dataclass(frozen=True)
class Config:
    version: str = "2.0"
    sample_rate_hz: float = 10.0
    duration_s: float = 600.0
    ias_window_s: float = 15.0
    rpm_window_s: float = 45.0
    trend_window_s: float = 10.0
    energy_window_s: float = 2.0
    band_low_hz: float = 0.3
    band_high_hz: float = 1.8
    max_gap_s: float = 1.0
    min_event_s: float = 0.5
    min_iou: float = 0.1
    n_estimators: int = 150
    max_train_samples: int = 12000
    random_state: int = 2026

    def to_dict(self):
        return asdict(self)
