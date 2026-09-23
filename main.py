"""Validate, freeze once, then reproduce the same benchmark without retuning."""
import argparse
import json
from pathlib import Path
import pandas as pd
from src.v2.config import Config, DEVELOPMENT_SEEDS, VALIDATION_SEEDS, BENCHMARK_SEEDS
from src.v2.data import generate_flight, make_split
from src.v2.models import fit_detector
from src.v2.experiments import select_model, evaluate_split, historical_comparison
from src.v2.evaluation import aggregate
from src.v2.protocol import (utc_now, write_json, source_hashes, environment, frame_hash,
                             fit_fingerprint, verify_freeze)
from src.v2.reporting import create_figures, write_report

ROOT = Path(__file__).resolve().parent


def export_split(root, name, flights, result):
    metrics, events, types, predictions = result
    destination = root/"data/v2"/name
    destination.mkdir(parents=True, exist_ok=True)
    for flight_id, flight in flights.items():
        pd.concat([flight, predictions[flight_id]], axis=1).to_csv(destination/f"{flight_id}.csv", index=False)
    metrics.to_csv(root/"reports"/f"{name}_metrics.csv", index=False)
    events.to_csv(root/"reports"/f"{name}_events.csv", index=False)
    types.to_csv(root/"reports"/f"{name}_types.csv", index=False)


def run(develop_only=False):
    root, config = ROOT, Config()
    reports = root/"reports"
    reports.mkdir(exist_ok=True)
    freeze_path, receipt_path = reports/"freeze_manifest.json", reports/"benchmark_receipt.json"
    manifest = json.loads(freeze_path.read_text()) if freeze_path.exists() else None
    if manifest:
        config = Config(**manifest["config"])
        if source_hashes(root) != manifest["source_sha256"]:
            raise RuntimeError("Frozen source changed. Do not reuse this holdout for new development.")
    print("Fitting normal-only development references and novelty models...", flush=True)
    normal = {str(s): generate_flight(s, config, anomalies=False) for s in DEVELOPMENT_SEEDS}
    detector = fit_detector(normal, config)
    validation = make_split(VALIDATION_SEEDS, config, normal_count=1)
    scores = {k: detector.score(v) for k, v in validation.items()}
    if manifest:
        verify_freeze(root, manifest, detector)
        selection = manifest["selection"]
        experiments = pd.read_csv(reports/"validation_experiments.csv")
    else:
        print("Selecting on six validation flights only...", flush=True)
        selection, experiments = select_model(validation, scores, config)
        experiments.to_csv(reports/"validation_experiments.csv", index=False)
        write_json(reports/"validation_selection.json", selection)
    validation_result = evaluate_split(validation, scores, selection, config)
    export_split(root, "validation", validation, validation_result)
    validation_summary = aggregate(validation_result[0])
    write_json(reports/"validation_summary.json", validation_summary)
    historical, history = historical_comparison(root)
    historical.to_csv(reports/"v1_recalculated.csv", index=False)
    comparison = pd.concat([historical, experiments], ignore_index=True)
    comparison.to_csv(reports/"model_comparison.csv", index=False)
    for name, (df, p, e) in history.items():
        if name == "V1 historical final":
            folder = root/"data/v2/historical"
            folder.mkdir(parents=True, exist_ok=True)
            df.join(p.drop(columns=df.columns, errors="ignore")).to_csv(folder/"v1_final.csv", index=False)
            e.to_csv(reports/"historical_events.csv", index=False)
    development = make_split(DEVELOPMENT_SEEDS, config)
    dev_scores = {k: detector.score(v) for k, v in development.items()}
    export_split(root, "development", development, evaluate_split(development, dev_scores, selection, config))
    print("Validation selection:", selection, flush=True)
    print("Validation metrics:", json.dumps(validation_summary), flush=True)
    if develop_only:
        print("Development only: no final benchmark generated.")
        return
    if manifest is None:
        manifest = {"frozen_at_utc": utc_now(), "config": config.to_dict(), "selection": selection,
                    "development_seeds": list(DEVELOPMENT_SEEDS), "validation_seeds": list(VALIDATION_SEEDS),
                    "benchmark_seeds": list(BENCHMARK_SEEDS), "validation_normal_count": 1, "benchmark_normal_count": 2,
                    "source_sha256": source_hashes(root), "environment": environment(),
                    "fitted_state_sha256": fit_fingerprint(detector),
                    "selection_objective": "0.5 sample F1 + 0.3 event F1 + 0.2 macro anomaly-family recall"}
        with freeze_path.open("x", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    print("Freeze verified. Generating predetermined benchmark flights...", flush=True)
    benchmark = make_split(manifest["benchmark_seeds"], config, normal_count=manifest["benchmark_normal_count"])
    benchmark_scores = {k: detector.score(v) for k, v in benchmark.items()}
    result = evaluate_split(benchmark, benchmark_scores, selection, config)
    metrics, events, types, predictions = result
    summary = aggregate(metrics)
    verification = {"data_sha256": {k: frame_hash(v) for k, v in benchmark.items()},
                    "prediction_sha256": {k: frame_hash(v) for k, v in predictions.items()}, "summary": summary}
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["verification"] != verification:
            raise RuntimeError("Benchmark reproduction differs from first-evaluation receipt.")
        print("Exact original benchmark data, scores, predictions and metrics reproduced.", flush=True)
    else:
        with receipt_path.open("x", encoding="utf-8") as f:
            json.dump({"first_evaluated_at_utc": utc_now(), "verification": verification}, f, indent=2)
        print("First benchmark evaluation recorded. No further detector tuning permitted.", flush=True)
    export_split(root, "final_benchmark", benchmark, result)
    events.to_csv(reports/"final_detected_events.csv", index=False)
    write_json(reports/"final_benchmark_summary.json", summary)
    first = str(manifest["benchmark_seeds"][0])
    create_figures(reports/"figures", summary, metrics, types, comparison, benchmark[first],
                   events[events.flight_id.eq(first)], config, first)
    write_report(root, manifest, summary, metrics, events, types)
    print("Final benchmark:", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--develop", action="store_true", help="Develop/validate only; never generate final flights.")
    run(parser.parse_args().develop)
