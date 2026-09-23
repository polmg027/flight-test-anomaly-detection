from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.features import create_engineered_features


RAW_FEATURES = [
    "altitude_ft",
    "ias_kt",
    "roll_deg",
    "pitch_deg",
    "yaw_rate_dps",
    "nz_g",
    "engine_rpm",
]


def get_features(df, engineered=False):
    if engineered:
        return create_engineered_features(df)

    return df[RAW_FEATURES].copy()


def fit_detector(df, engineered=False):
    features = get_features(df, engineered)

    # Learn how to replace missing values
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(features)

    # Learn scaling parameters from development data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Train Isolation Forest
    model = IsolationForest(
        n_estimators=200,
        contamination=0.08,
        random_state=42,
    )

    model.fit(X_scaled)

    detector = {
        "imputer": imputer,
        "scaler": scaler,
        "model": model,
        "engineered": engineered,
    }

    return detector


def predict_anomalies(df, detector):
    features = get_features(
        df,
        engineered=detector["engineered"]
    )

    # Important: no fit here
    X_imputed = detector["imputer"].transform(features)
    X_scaled = detector["scaler"].transform(X_imputed)

    predictions = detector["model"].predict(X_scaled)

    results = df.copy()
    results["predicted_anomaly"] = (
        predictions == -1
    ).astype(int)

    return results