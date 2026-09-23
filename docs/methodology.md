# Methodology and engineering decisions

## What is learned and when?

| Component | Fitting source | Inference behaviour |
| --- | --- | --- |
| Interpolation/centered filters | No fitted state | Process one complete flight locally |
| Imputer/RobustScaler | Eight normal development profiles | Transform only |
| Feature reference scale | Normal development 99.5th percentile | Divide by fixed scale |
| IF and LOF | Fixed random subset of 12,000 normal development samples | Novelty score only |
| Score normalization | All normal development scores | Fixed median/99.5th percentile map |
| Model, mode, threshold | Six labeled validation flights | Frozen choice |
| Events | Fixed 1 s gap / 0.5 s flagged duration | No truth access |
| Event matching | Evaluation labels only | Never used to create predictions |

Normal development profiles are generated without injections. Their injected twins
are useful in-sample diagnostics, not independent performance evidence. Different
seeds define validation/final profiles. Labels, phase, time and seed never enter the
eight-dimensional ML input matrix. Time is checked for a regular 10 Hz grid.

All candidates have a missing-value override so data-quality faults cannot disappear
through imputation. The `rules` candidate additionally overrides IAS/RPM residual
ratios above 3; `or`/`weighted` candidates retain that same rule. The selected `ml`
mode uses LOF and the missingness rule only. A score above 2 is anomalous; it is not
a probability of failure. Contamination in V1 sets the development score cutoff;
V2 does not select its threshold using a requested anomaly fraction.

## Features and assumptions

IAS median residual (15 s) and RPM median residual (45 s) tolerate different nominal
operating values but can miss long faults that become part of the local baseline.
Normal speed/RPM changes that are abrupt relative to these windows can cause alerts.

Altitude is smoothed over approximately 2 s, differenced across 10 s and compared
with independent vertical speed over the same interval. Reference noise/bias and
slow mismatch are injected into normal profiles. At the first/last half-window,
undefined trend consistency is set to zero: a known edge blind spot. This is not a
general solution to drift without redundant information. A real sensor's independence,
time alignment and accuracy would have to be justified separately.

Roll/yaw use a third-order Butterworth 0.3–1.8 Hz bandpass and approximately 2 s RMS.
The upper band is below the 5 Hz Nyquist frequency. `sosfiltfilt` applies a zero-phase
forward/backward filter; its effective magnitude response differs from one pass,
and it uses future samples. Filtering can spread energy outside a true interval.
The sample rate must match configuration; irregular data are rejected, not silently
resampled. Full missing channels remain explicit faults even though zero filling is
needed to calculate finite intermediate features.

Pitch and load factor use 10 s local median residuals. Eight features are sufficient
for this illustrative task; no hundreds-feature search or deep-learning architecture
was introduced. Model feature contributions are descriptive normalized residuals,
not causal explanations or SHAP values for LOF.

## Event semantics

Predicted samples separated by at most one second of unflagged data form an event.
The duration spans bridged gaps; minimum support counts flagged samples only.
Rule overrides preserve even a one-sample missing measurement. Event starts/ends
use sample intervals `[t, t+dt)`. Ground truth is split whenever the label/type changes.

Matching requires positive overlap and IoU ≥0.10. A linear assignment solves a
maximum-cardinality match with IoU as secondary objective. Each true/reported event
can be used only once. Extra overlapping fragments count as false reports, and
merged events can leave missed true events. `unmatched_overlapping_reports` is a
diagnostic: it can include low-IoU overlaps as well as genuine duplicate fragments.

Signed onset error can be negative with offline look-ahead. It does not measure
how long a live system would take to alert. Undefined metric ratios are zero;
normal-only controls should be judged by false alarms and true negatives.

## Experiment record and integrity

The initial predeclared grid had 32 candidates (8 architectures × 4 thresholds).
It was run during development, then reproduced before freeze with identical
selection. There was no additional detector search after final evaluation.
Validation is a model-selection estimate and can be optimistic. Final aggregate
metrics use counts over all eight predetermined flights; no flights are dropped.

The freeze hashes all V2 numerical/protocol modules and main orchestration, with
normalized source line endings. Fitted references, tree structures and LOF training
matrix contribute to a state fingerprint. A first-evaluation receipt stores hashes
of every full input table and output score/prediction table plus aggregate metrics.
Reports/UI can be restyled afterward; frozen numerical code cannot change silently.
This is a local reproducibility safeguard, not protection against deliberate edits
to both code and provenance records.

V1 figures and console output are preserved. V1 sample metrics are recalculated
without changes to its models/features/rules. The V2 comparison additionally applies
V2 event consolidation/matching to V1 predictions, so its event totals differ from
the historical original protocol. Keep those definitions separate in presentations.

## Interpreting improvement

V2 F1 0.951 exceeds V1 historical F1 0.638, but data generation, extra sensor inputs,
training setup and event semantics changed. This is evidence of a stronger system
on a new synthetic task, not a controlled causal estimate of improvement. The shared
generator makes nominal/fault patterns consistent across splits; unfamiliar faults,
sensor correlations and real turbulence could materially reduce performance.

Six extra final reports remain, five overlapping true episodes. Keeping these
results visible is more useful than tuning event merging on this inspected holdout.
Mean matched IoU of 0.930 complements event recall and exposes boundary quality.
