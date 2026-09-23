# V2 final benchmark — automated engineering report

All data are synthetic, unrelated to Airbus or any real aircraft/test campaign.
Version 2.0; freeze: 2026-09-23T17:40:17.146069+00:00.
Validation-selected configuration: `{"model": "local_outlier_factor", "mode": "ml", "threshold": 2.0}`.

Eight independent 600-second flights at 10 Hz; six with five injections each,
two normal-only controls. The timeline uses the first predetermined benchmark
flight, not a selected best case.

## Sample and event metrics

| metric | value |
| --- | --- |
| samples | 48000.0000 |
| tp | 4769.0000 |
| fp | 370.0000 |
| fn | 118.0000 |
| tn | 42743.0000 |
| true_events | 30.0000 |
| reported_events | 36.0000 |
| matched_events | 30.0000 |
| missed_events | 0.0000 |
| false_reported_events | 6.0000 |
| unmatched_overlapping_reports | 5.0000 |
| precision | 0.9280 |
| recall | 0.9759 |
| f1 | 0.9513 |
| event_precision | 0.8333 |
| event_recall | 1.0000 |
| event_f1 | 0.9091 |
| duration_h | 1.3333 |
| false_events_per_hour | 4.5000 |
| duplicate_detection_rate | 0.1389 |
| mean_matched_iou | 0.9297 |
| mean_signed_onset_error_s | -0.3700 |
| per_flight_f1_min | 0.0000 |
| per_flight_f1_max | 0.9653 |

## Per-flight results

| flight_id | precision | recall | f1 | event_precision | event_recall | false_reported_events | missed_events |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1009 | 0.9563 | 0.9745 | 0.9653 | 0.8333 | 1.0000 | 1 | 0 |
| 1031 | 0.9278 | 0.9779 | 0.9522 | 0.8333 | 1.0000 | 1 | 0 |
| 1061 | 0.8750 | 0.9820 | 0.9254 | 1.0000 | 1.0000 | 0 | 0 |
| 1091 | 0.9281 | 0.9834 | 0.9549 | 0.8333 | 1.0000 | 1 | 0 |
| 1151 | 0.9390 | 0.9556 | 0.9472 | 0.7143 | 1.0000 | 2 | 0 |
| 1181 | 0.9250 | 0.9818 | 0.9525 | 0.8333 | 1.0000 | 1 | 0 |
| 1213 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| 1237 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |

Undefined precision/recall/F1 are zero, including normal-only flights with no
alerts. Interpret these controls using false reports and true negatives.
Aggregate ratios sum counts first.

## Anomaly-family recall

| anomaly_type | samples | detected | recall |
| --- | --- | --- | --- |
| airspeed_spike | 95 | 95 | 1.0000 |
| altitude_drift | 2893 | 2776 | 0.9596 |
| engine_rpm_drop | 566 | 566 | 1.0000 |
| lateral_oscillation | 1170 | 1169 | 0.9991 |
| sensor_dropout | 163 | 163 | 1.0000 |

## Reported events

| flight_id | event_id | start_s | end_s | reason | detection_source |
| --- | --- | --- | --- | --- | --- |
| 1009 | 1 | 45.3000 | 67.1000 | lateral band energy | ml |
| 1009 | 2 | 152.5000 | 154.6000 | sensor dropout; missingness preserved before interpolation | engineering rule |
| 1009 | 3 | 258.4000 | 258.9000 | local airspeed residual | ml |
| 1009 | 4 | 363.7000 | 370.4000 | local RPM residual | ml |
| 1009 | 5 | 465.9000 | 511.0000 | cross-sensor trend inconsistency | ml |
| 1009 | 6 | 513.3000 | 533.2000 | cross-sensor trend inconsistency | ml |
| 1031 | 1 | 52.3000 | 90.0000 | cross-sensor trend inconsistency | ml |
| 1031 | 2 | 91.8000 | 110.5000 | cross-sensor trend inconsistency | ml |
| 1031 | 3 | 155.6000 | 157.2000 | local airspeed residual | ml |
| 1031 | 4 | 258.9000 | 266.2000 | local RPM residual | ml |
| 1031 | 5 | 356.8000 | 373.5000 | lateral band energy | ml |
| 1031 | 6 | 467.7000 | 471.6000 | sensor dropout; missingness preserved before interpolation | engineering rule |
| 1061 | 1 | 44.9000 | 51.5000 | local RPM residual | ml |
| 1061 | 2 | 157.1000 | 195.2000 | cross-sensor trend inconsistency | ml |
| 1061 | 3 | 264.5000 | 268.7000 | sensor dropout; missingness preserved before interpolation | engineering rule |
| 1061 | 4 | 356.5000 | 358.1000 | local airspeed residual | ml |
| 1061 | 5 | 470.3000 | 483.2000 | lateral band energy | ml |
| 1091 | 1 | 47.9000 | 76.4000 | lateral band energy | ml |
| 1091 | 2 | 150.0000 | 181.5000 | cross-sensor trend inconsistency | ml |
| 1091 | 3 | 183.0000 | 199.1000 | cross-sensor trend inconsistency | ml |
| 1091 | 4 | 253.9000 | 256.2000 | local airspeed residual | ml |
| 1091 | 5 | 358.5000 | 359.1000 | sensor dropout; missingness preserved before interpolation | engineering rule |
| 1091 | 6 | 472.0000 | 488.9000 | local RPM residual | ml |
| 1151 | 1 | 0.0000 | 1.2000 | lateral band energy | ml |
| 1151 | 2 | 44.4000 | 45.3000 | local airspeed residual | ml |
| 1151 | 3 | 152.4000 | 153.3000 | sensor dropout; missingness preserved before interpolation | engineering rule |
| 1151 | 4 | 262.1000 | 272.4000 | local RPM residual | ml |
| 1151 | 5 | 367.1000 | 402.2000 | cross-sensor trend inconsistency | ml |
| 1151 | 6 | 404.8000 | 422.4000 | cross-sensor trend inconsistency | ml |
| 1151 | 7 | 465.6000 | 480.1000 | lateral band energy | ml |
| 1181 | 1 | 50.7000 | 55.3000 | sensor dropout; missingness preserved before interpolation | engineering rule |
| 1181 | 2 | 156.6000 | 159.2000 | local airspeed residual | ml |
| 1181 | 3 | 259.0000 | 267.8000 | local RPM residual | ml |
| 1181 | 4 | 363.5000 | 389.9000 | lateral band energy | ml |
| 1181 | 5 | 472.2000 | 505.9000 | cross-sensor trend inconsistency | ml |
| 1181 | 6 | 507.5000 | 524.7000 | cross-sensor trend inconsistency | ml |

## Interpretation and limitations

One-to-one matching maximizes eligible match count then IoU; minimum IoU is
0.1. Extra fragments remain false reports. Half-open
intervals include the final sample period. Consolidation does not change sample
predictions. Signed onset error can be negative because centered offline features
use future data; it is not operational detection latency.

Normal-only development fits reference scales, imputation, scaling and novelty
models. Validation labels select thresholds and fusion: a semi-supervised protocol.
Final data are generated after freezing source and fitted-state hashes. Later runs
verify original data, score and prediction hashes against the first receipt.

Altitude consistency depends on an independent noisy vertical-speed reference.
This extra sensor makes V2 a different task from V1; metrics do not establish a
controlled improvement on identical data. Shared generator assumptions, limited
anomaly families and synthetic noise restrict external validity. No real-aircraft,
operational, streaming or safety-critical validation has been performed.
