"""Normal-only fitted novelty models and transparent score fusion."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from .config import Config
from .features import build_features

EVIDENCE = {
    "ias_residual": ("ias_kt", "local airspeed residual"),
    "rpm_residual": ("engine_rpm", "local RPM residual"),
    "altitude_consistency": ("altitude_ft, vertical_speed_fps", "cross-sensor trend inconsistency"),
    "roll_deg_band_rms": ("roll_deg", "lateral band energy"),
    "yaw_rate_dps_band_rms": ("yaw_rate_dps", "lateral band energy"),
    "pitch_deg_residual": ("pitch_deg", "local pitch residual"),
    "nz_g_residual": ("nz_g", "local load-factor residual"),
}


@dataclass
class Detector:
    config: Config
    imputer: SimpleImputer
    scaler: RobustScaler
    models: dict
    normalizers: dict
    feature_scale: np.ndarray

    def score(self, flight):
        f = build_features(flight, self.config)
        x = self.scaler.transform(self.imputer.transform(f))
        evidence = f[list(EVIDENCE)].to_numpy() / self.feature_scale
        out = pd.DataFrame(index=flight.index)
        out["statistical_score"] = evidence.max(axis=1)
        top = evidence.argmax(axis=1)
        names = list(EVIDENCE)
        out["primary_signals"] = [EVIDENCE[names[i]][0] for i in top]
        out["reason"] = [EVIDENCE[names[i]][1] for i in top]
        out["missing"] = f.missing_any.astype(bool)
        # High-confidence data-quality override, calibrated from normal development.
        out["rule_score"] = np.maximum(evidence[:, 0], evidence[:, 1])
        for name, model in self.models.items():
            raw = -model.score_samples(x)
            median, high = self.normalizers[name]
            out[name] = np.maximum(0, (raw-median) / max(high-median, 1e-9))
        return out


def fit_detector(normal_flights: dict, config: Config = Config()) -> Detector:
    f = pd.concat([build_features(df, config) for df in normal_flights.values()], ignore_index=True)
    # Every normal sample contributes to robust reference scales; models use a fixed subsample.
    scale = np.maximum(f[list(EVIDENCE)].quantile(.995).to_numpy(), 1e-6)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = RobustScaler()
    all_x = scaler.fit_transform(imputer.fit_transform(f))
    rng = np.random.default_rng(config.random_state)
    idx = rng.choice(len(all_x), min(config.max_train_samples, len(all_x)), replace=False)
    x = all_x[idx]
    models = {
        "isolation_forest": IsolationForest(n_estimators=config.n_estimators, random_state=config.random_state, n_jobs=1),
        "local_outlier_factor": LocalOutlierFactor(n_neighbors=35, novelty=True, n_jobs=1),
    }
    normalizers = {}
    for name, model in models.items():
        model.fit(x)
        raw = -model.score_samples(all_x)
        normalizers[name] = tuple(np.quantile(raw, [.5, .995]))
    return Detector(config, imputer, scaler, models, normalizers, scale)


def predict(scores: pd.DataFrame, selection: dict) -> pd.DataFrame:
    """No truth fields accepted or read. Threshold parameters come from validation."""
    out = scores.copy()
    model, mode = selection["model"], selection["mode"]
    ml = scores[model] if model in scores else np.zeros(len(scores))
    stat = scores.statistical_score
    if mode == "ml":
        score = ml
    elif mode == "statistical":
        score = stat
    elif mode == "rules":
        score = scores.rule_score
    elif mode == "or":
        score = np.maximum(stat, ml)
    elif mode == "weighted":
        score = .65*stat + .35*ml
    else:
        raise ValueError(f"Unknown fusion mode {mode}")
    threshold = selection["threshold"]
    rule = (scores.rule_score > 3) | scores.missing
    override = rule if mode in ("or", "weighted", "rules") else scores.missing
    out["ml_score"] = ml
    out["fused_score"] = score
    out["predicted_anomaly"] = ((score > threshold) | override).astype(int)
    out["rule_triggered"] = override
    out["detection_source"] = np.where(override, "engineering rule", mode)
    out.loc[scores.missing, "primary_signals"] = "missing sensor measurement"
    out.loc[scores.missing, "reason"] = "sensor dropout; missingness preserved before interpolation"
    return out
