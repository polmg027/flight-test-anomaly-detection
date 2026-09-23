"""Behavioral checks use development fixtures only; never generate the holdout."""
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from src.v2.config import Config, SIGNALS, DEVELOPMENT_SEEDS, VALIDATION_SEEDS, BENCHMARK_SEEDS
from src.v2.data import generate_flight
from src.v2.features import build_features
from src.v2.models import fit_detector, predict
from src.v2.events import consolidate, truth_events, match_events
from src.v2.evaluation import evaluate, aggregate
from src.v2.protocol import source_hashes, verify_freeze, frame_hash

CFG = replace(Config(), duration_s=240, n_estimators=12, max_train_samples=500)


@pytest.fixture(scope="module")
def flight():
    return generate_flight(101, CFG)


@pytest.fixture(scope="module")
def detector():
    return fit_detector({"101": generate_flight(101, CFG, False),
                         "113": generate_flight(113, CFG, False)}, CFG)


def test_generation_reproducible_and_varied(flight):
    pd.testing.assert_frame_equal(flight, generate_flight(101, CFG))
    assert not flight.equals(generate_flight(113, CFG))
    assert len(flight) == 2400
    assert {"time_s", "anomaly", "anomaly_type", *SIGNALS}.issubset(flight)
    assert flight.time_s.iloc[-1] == 239.9
    assert flight.anomaly_type.nunique() == 6


def test_split_disjoint():
    splits = [set(DEVELOPMENT_SEEDS), set(VALIDATION_SEEDS), set(BENCHMARK_SEEDS)]
    assert not splits[0]&splits[1] and not splits[0]&splits[2] and not splits[1]&splits[2]


def test_normal_flight_has_manoeuvres_without_labels():
    normal = generate_flight(113, CFG, False)
    assert normal.anomaly.sum() == 0
    assert normal.altitude_ft.max()-normal.altitude_ft.min() > 100
    assert normal.phase.nunique() == 5


def test_features_missingness_and_label_independence(flight):
    f = build_features(flight, CFG)
    assert f.shape == (2400, 8)
    assert np.isfinite(f).all().all()
    assert f.missing_any.sum() == flight[list(SIGNALS)].isna().any(axis=1).sum()
    stripped = flight.drop(columns=["anomaly", "anomaly_type", "phase"])
    pd.testing.assert_frame_equal(f, build_features(stripped, CFG))
    altered = flight.assign(anomaly=1-flight.anomaly, anomaly_type="invented", phase="invented")
    pd.testing.assert_frame_equal(f, build_features(altered, CFG))


def test_all_missing_sensor_is_explicit(flight):
    f = build_features(flight.assign(ias_kt=np.nan), CFG)
    assert f.missing_any.eq(1).all() and np.isfinite(f).all().all()


@pytest.mark.parametrize("mutation", ["missing", "irregular", "infinite"])
def test_invalid_inputs_rejected(flight, mutation):
    bad = flight.copy()
    if mutation == "missing":
        bad = bad.drop(columns="ias_kt")
    elif mutation == "irregular":
        bad.loc[10, "time_s"] += .01
    else:
        bad.loc[10, "ias_kt"] = np.inf
    with pytest.raises(ValueError):
        build_features(bad, CFG)


def test_inference_never_refits_and_labels_are_unused(detector, flight):
    before = detector.scaler.center_.copy()
    with patch.object(detector.imputer, "fit", side_effect=AssertionError("refit")), \
         patch.object(detector.scaler, "fit", side_effect=AssertionError("refit")):
        scores = detector.score(flight.drop(columns=["anomaly", "anomaly_type"]))
        p = predict(scores, {"model": "local_outlier_factor", "mode": "ml", "threshold": 2.})
    np.testing.assert_array_equal(before, detector.scaler.center_)
    assert len(p) == len(flight) and set(p.predicted_anomaly.unique()) <= {0, 1}
    assert p.loc[scores.missing, "predicted_anomaly"].eq(1).all()
    pd.testing.assert_frame_equal(scores, detector.score(flight))


def test_training_reproducibility(detector, flight):
    other = fit_detector({"101": generate_flight(101, CFG, False),
                          "113": generate_flight(113, CFG, False)}, CFG)
    pd.testing.assert_frame_equal(detector.score(flight), other.score(flight))


def sample_predictions(values, rules=None):
    n = len(values)
    return pd.DataFrame({"predicted_anomaly": values, "rule_triggered": rules if rules is not None else [False]*n,
                         "primary_signals": "ias_kt", "reason": "residual", "ml_score": 1.,
                         "statistical_score": 1., "fused_score": 1., "detection_source": "ml"})


def test_consolidation_bridge_duration_and_rule_override():
    f = pd.DataFrame({"time_s": np.arange(40)/10})
    p = sample_predictions([1]*5+[0]*5+[1]*5+[0]*15+[1]+[0]*9)
    original = p.copy()
    events = consolidate(f, p)
    assert len(events) == 1 and events.iloc[0].anomalous_samples == 10
    assert events.iloc[0].end_s == pytest.approx(1.5)
    p.loc[30, "rule_triggered"] = True
    assert len(consolidate(f, p)) == 2
    pd.testing.assert_frame_equal(original, p.assign(rule_triggered=False))


def test_empty_events_have_schema():
    e = consolidate(pd.DataFrame({"time_s": [0., .1]}), sample_predictions([0, 0]))
    assert e.empty and "event_id" in e and "reason" in e


def test_truth_event_boundaries_and_types(flight):
    e = truth_events(flight, CFG)
    assert len(e) == 5 and e.anomaly_type.nunique() == 5
    assert ((e.end_s-e.start_s) > 0).all()
    normal = generate_flight(113, CFG, False)
    assert truth_events(normal, CFG).empty


def test_matching_penalizes_duplicates_and_merged_events():
    truth = pd.DataFrame({"start_s": [0., 20.], "end_s": [10., 30.]})
    reported = pd.DataFrame({"start_s": [0., 2., 20.], "end_s": [10., 8., 30.]})
    m = match_events(reported, truth)
    assert m["matched_events"] == 2 and m["false_reported_events"] == 1
    assert m["unmatched_overlapping_reports"] == 1
    merged = pd.DataFrame({"start_s": [0.], "end_s": [30.]})
    assert match_events(merged, truth)["missed_events"] == 1


def test_tiny_overlap_and_touching_do_not_match():
    truth = pd.DataFrame({"start_s": [10.], "end_s": [20.]})
    for end in (10., 10.01):
        report = pd.DataFrame({"start_s": [0.], "end_s": [end]})
        assert match_events(report, truth)["matched_events"] == 0


def test_sample_metrics_and_normal_only_case():
    f = pd.DataFrame({"time_s": np.arange(10)/10, "anomaly": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                      "anomaly_type": ["airspeed_spike"]*2+["normal"]*8})
    m, _, _ = evaluate(f, sample_predictions([1, 0, 1, 0, 0, 0, 0, 0, 0, 0]))
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 7)
    assert m["f1"] == .5
    normal = f.assign(anomaly=0, anomaly_type="normal")
    m, _, _ = evaluate(normal, sample_predictions([0]*10))
    assert m["tn"] == 10 and m["false_reported_events"] == 0
    assert aggregate(pd.DataFrame([m, m]))["tn"] == 20


def test_freeze_rejects_source_change(tmp_path):
    from src.v2.protocol import CORE_FILES
    for name in CORE_FILES:
        p = tmp_path/name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("frozen")
    manifest = {"source_sha256": source_hashes(tmp_path)}
    (tmp_path/"main.py").write_text("changed")
    with pytest.raises(RuntimeError, match="Frozen code changed"):
        verify_freeze(tmp_path, manifest, None)


def test_hash_detects_changed_predictions():
    a = pd.DataFrame({"prediction": [0, 1]})
    assert frame_hash(a) != frame_hash(a.assign(prediction=[1, 1]))
