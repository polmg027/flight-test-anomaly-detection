from pathlib import Path

import pandas as pd

from src.anomaly_detection import fit_detector, predict_anomalies
from src.evaluation import evaluate_model, detection_by_anomaly_type
from src.rule_based import engineering_rules, combine_with_ml
from src.events import extract_anomaly_events
from src.event_evaluation import (
    extract_ground_truth_events,
    evaluate_events,
)
from src.visualization import plot_anomaly_timeline

BASE_DIR = Path(__file__).resolve().parent
FIGURES_DIR = BASE_DIR / "reports" / "figures"

DEVELOPMENT_FILE = BASE_DIR / "data" / "development_flight.csv"
VALIDATION_FILE = BASE_DIR / "data" / "validation_flight.csv"
FINAL_TEST_FILE = BASE_DIR / "data" / "final_test_flight.csv"

final_test = pd.read_csv(FINAL_TEST_FILE)

print("\nFinal test flight:")
print(f"Samples: {len(final_test)}")
print(f"Ground-truth anomalies: {final_test['anomaly'].sum()}")

def print_results(title, results):
    metrics = evaluate_model(results)
    summary = detection_by_anomaly_type(results)

    predicted_anomalies = results["predicted_anomaly"].sum()
    anomaly_rate = 100 * predicted_anomalies / len(results)

    print(f"\n=== {title} ===")

    print(f"Predicted anomalies: {predicted_anomalies}")
    print(f"Predicted anomaly rate: {anomaly_rate:.2f}%")

    print("\nModel performance:")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1 score: {metrics['f1']:.3f}")

    print("\nConfusion matrix:")
    print(f"True positives:  {metrics['true_positives']}")
    print(f"False positives: {metrics['false_positives']}")
    print(f"False negatives: {metrics['false_negatives']}")
    print(f"True negatives:  {metrics['true_negatives']}")

    print("\nDetection by anomaly type:")
    print(summary)


def main():
    print("=== FLIGHT TEST ANOMALY DETECTION ===")

    # Load independent flights
    development = pd.read_csv(DEVELOPMENT_FILE)
    validation = pd.read_csv(VALIDATION_FILE)

    print("\nDevelopment flight:")
    print(f"Samples: {len(development)}")
    print(f"Ground-truth anomalies: {development['anomaly'].sum()}")

    print("\nValidation flight:")
    print(f"Samples: {len(validation)}")
    print(f"Ground-truth anomalies: {validation['anomaly'].sum()}")

    # --------------------------------------------------
    # BASELINE MODEL
    # --------------------------------------------------

    baseline_detector = fit_detector(
        development,
        engineered=False
    )

    baseline_development = predict_anomalies(
        development,
        baseline_detector
    )

    baseline_validation = predict_anomalies(
        validation,
        baseline_detector
    )

    print_results(
        "BASELINE - DEVELOPMENT",
        baseline_development
    )

    print_results(
        "BASELINE - VALIDATION",
        baseline_validation
    )

    # --------------------------------------------------
    # FEATURE-ENGINEERED MODEL
    # --------------------------------------------------

    engineered_detector = fit_detector(
        development,
        engineered=True
    )
    engineered_final_test = predict_anomalies(
        final_test,
        engineered_detector
    )

    engineered_development = predict_anomalies(
        development,
        engineered_detector
    )

    engineered_validation = predict_anomalies(
        validation,
        engineered_detector
    )

    print_results(
        "ENGINEERED - DEVELOPMENT",
        engineered_development
    )

    print_results(
        "ENGINEERED - VALIDATION",
        engineered_validation
    )

    # --------------------------------------------------
    # ENGINEERING RULES
    # --------------------------------------------------

    development_rules = engineering_rules(development)
    validation_rules = engineering_rules(validation)

    rule_development = development.copy()
    rule_development["predicted_anomaly"] = (
        development_rules["rule_anomaly"]
    )

    rule_validation = validation.copy()
    rule_validation["predicted_anomaly"] = (
        validation_rules["rule_anomaly"]
    )

    print_results(
        "RULE-BASED - DEVELOPMENT",
        rule_development
    )

    print_results(
        "RULE-BASED - VALIDATION",
        rule_validation
    )

    # --------------------------------------------------
    # HYBRID MODEL
    # --------------------------------------------------

    hybrid_development = combine_with_ml(
        engineered_development,
        development_rules
    )

    hybrid_validation = combine_with_ml(
        engineered_validation,
        validation_rules
    )

    print_results(
        "HYBRID - DEVELOPMENT",
        hybrid_development
    )

    print_results(
        "HYBRID - VALIDATION",
        hybrid_validation
    )

        # --------------------------------------------------
    # FINAL UNSEEN TEST
    # --------------------------------------------------

    final_test_rules = engineering_rules(final_test)

    hybrid_final_test = combine_with_ml(
        engineered_final_test,
        final_test_rules
    )

    final_events = extract_anomaly_events(
        hybrid_final_test,
        final_test_rules
    )

    print("\n=== DETECTED EVENTS - FINAL TEST ===")
    print(f"Number of detected events: {len(final_events)}")

    print(
        final_events[
            [
                "event_id",
                "start_s",
                "end_s",
                "duration_s",
                "detection_source",
            ]
        ].to_string(index=False)
    )
    print_results(
        "HYBRID - FINAL UNSEEN TEST",
        hybrid_final_test
    )

    true_final_events = extract_ground_truth_events(final_test)

    event_metrics = evaluate_events(
        final_events,
        true_final_events
    )

    print("\n=== EVENT-LEVEL PERFORMANCE ===")
    print(f"True anomaly events: {event_metrics['true_events']}")
    print(f"Reported events: {event_metrics['reported_events']}")
    print(
        f"True events detected: "
        f"{event_metrics['true_events_detected']}"
    )
    print(
        f"Event recall: "
        f"{event_metrics['event_recall']:.1%}"
    )
    print(
        f"False reported events: "
        f"{event_metrics['false_reported_events']}"
    )

    timeline_file = (
        FIGURES_DIR / "final_test_anomaly_timeline.png"
    )

    plot_anomaly_timeline(
        final_test,
        final_events,
        timeline_file,
    )

    print(f"\nTimeline saved: {timeline_file}")
    
if __name__ == "__main__":
    main()