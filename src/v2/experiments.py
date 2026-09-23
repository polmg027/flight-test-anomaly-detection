"""Validation-only finite grid; historical V1 comparisons remain separate."""
import pandas as pd
from .models import predict
from .evaluation import evaluate, aggregate


def evaluate_split(flights, scores, selection, config):
    metrics, events, types, predictions = [], [], [], {}
    for flight_id, flight in flights.items():
        p = predict(scores[flight_id], selection)
        m, e, t = evaluate(flight, p, config)
        metrics.append({"flight_id": flight_id, **m})
        events.append(e.assign(flight_id=flight_id))
        types.append(t.assign(flight_id=flight_id))
        predictions[flight_id] = p
    return pd.DataFrame(metrics), pd.concat(events, ignore_index=True), pd.concat(types, ignore_index=True), predictions


def select_model(flights, scores, config):
    rows = []
    specs = [("none", "rules"), ("none", "statistical")]
    for model in ("isolation_forest", "local_outlier_factor"):
        specs.extend((model, mode) for mode in ("ml", "or", "weighted"))
    for model, mode in specs:
        for threshold in (1., 1.5, 2., 3.):
            selection = {"model": model, "mode": mode, "threshold": threshold}
            metrics, _, types, _ = evaluate_split(flights, scores, selection, config)
            summary = aggregate(metrics)
            by_type = types.groupby("anomaly_type")[["samples", "detected"]].sum()
            macro_recall = float((by_type.detected / by_type.samples).mean())
            # Fixed before viewing validation: reward samples, events and class coverage.
            objective = .5*summary["f1"] + .3*summary["event_f1"] + .2*macro_recall
            rows.append({**selection, **summary, "macro_type_recall": macro_recall,
                         "objective": objective, "dataset": "V2 validation", "feature_set": "temporal_spectral_v2"})
    experiments = pd.DataFrame(rows).sort_values(["objective", "false_reported_events"], ascending=[False, True], kind="stable")
    best = experiments.iloc[0]
    selection = {"model": str(best.model), "mode": str(best["mode"]), "threshold": float(best.threshold)}
    return selection, experiments


def historical_comparison(root):
    from src.anomaly_detection import fit_detector, predict_anomalies
    from src.rule_based import engineering_rules, combine_with_ml
    from src.v2.config import Config
    development = pd.read_csv(root / "data/development_flight.csv")
    validation = pd.read_csv(root / "data/validation_flight.csv")
    historical = pd.read_csv(root / "data/final_test_flight.csv")
    baseline = fit_detector(development, engineered=False)
    engineered = fit_detector(development, engineered=True)
    rows, outputs = [], {}
    for split, df in (("V1 validation", validation), ("V1 historical final", historical)):
        rules = engineering_rules(df)
        eng = predict_anomalies(df, engineered)
        candidates = {"V1 baseline IF": predict_anomalies(df, baseline), "V1 engineered IF": eng,
                      "V1 rules": df.assign(predicted_anomaly=rules.rule_anomaly),
                      "V1 hybrid": combine_with_ml(eng, rules)}
        for name, p in candidates.items():
            # New one-to-one event metrics; V1 original matching is preserved separately.
            p = p.assign(rule_triggered=rules.rule_anomaly.astype(bool) if name in ("V1 rules", "V1 hybrid") else False,
                         primary_signals="V1 sensors", reason="historical V1 detector", ml_score=0.,
                         statistical_score=0., fused_score=p.predicted_anomaly.astype(float), detection_source=name)
            m, e, _ = evaluate(df, p, Config())
            rows.append({"model": name, "dataset": split, "feature_set": "V1", **m})
            if name == "V1 hybrid":
                outputs[split] = (df, p, e)
    return pd.DataFrame(rows), outputs
