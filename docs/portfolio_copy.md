# Portfolio and interview copy

All wording below refers to this repository's frozen V2 synthetic benchmark.
Do not imply use of employer data, real-aircraft validation or operational deployment.

## CV entry (2–3 lines)

**FLIGHT TEST ANOMALY DETECTION SYSTEM — Personal Python/ML Project**  
Developed an offline hybrid ML and engineering-rule pipeline for synthetic flight-test
data using temporal/spectral features, independent validation and frozen unseen-flight
evaluation; achieved sample F1 0.951 across eight synthetic benchmark flights.

## LinkedIn project description

I built a Python project to explore how anomaly detection can support the review
of flight-test-style time-series data. The work progressed from an Isolation Forest
baseline to a reproducible multi-flight framework combining temporal analysis,
band-limited signal energy, engineering consistency checks and machine-learning
novelty detection.

I compared Isolation Forest, Local Outlier Factor, statistical detection and several
hybrid strategies using independent validation flights. After freezing the selected
LOF detector and missing-data rule, I evaluated eight new synthetic flights: precision
0.928, recall 0.976 and sample F1 0.951. One-to-one event evaluation matched all
30 injected episodes, with six additional false reports.

The repository includes a Streamlit/Plotly review dashboard, explainable event tables,
automated reports and pytest checks. Technologies include Python, NumPy, pandas,
SciPy, scikit-learn and Git. All data are synthetic; the project makes no claim of
real-aircraft or safety-critical validation.

## Suggested LinkedIn skills

Python · Machine Learning · Data Analysis · Signal Processing · Aerospace Engineering ·
Flight Testing (project context) · pandas · NumPy · SciPy · scikit-learn · Data Visualization ·
Git · Software Testing

## 30–60 second explanation

“Before starting the internship, I developed a Python portfolio project for reviewing
synthetic flight-test signals. I started with Isolation Forest, studied its errors,
then added temporal and frequency-band features, engineering consistency checks
and a missing-data rule. I compared several detectors on separate validation flights
and froze the selected system before generating eight final flights. It achieved
sample F1 0.951 and found all 30 injected events, with six extra reports. I also built
an interactive dashboard, automated engineering reports and tests. The main lesson
was that useful features, careful validation and honest false-alert reporting matter
more than adding model complexity. The data are entirely synthetic.”

## 2–3 minute technical explanation

“The problem I wanted to study was how to identify unusual episodes in flight-test-style
time series without flagging every normal change in operating condition. I used
synthetic data so the project could be shared and reproduced without proprietary
information. The signals include altitude, airspeed, attitudes, yaw rate, load factor,
RPM and, in V2, an independent noisy vertical-speed measurement.

The first version used Isolation Forest. Temporal features improved validation F1,
and combining ML with engineering rules improved it further. However, performance
fell on the original unseen flight, and altitude drift remained weak. That led to
V2: multiple flights with different operating points, noise, phases and anomaly
timing, including normal-only controls.

I engineered eight compact features. Local median residuals identify airspeed and
RPM deviations. Bandpassed roll and yaw RMS identify oscillatory behaviour. Altitude
trend is compared with independent vertical speed. That last point is an explicit
assumption: altitude alone cannot generally distinguish drift from a legitimate climb.
Missingness is retained before interpolation and triggers a deterministic rule.

I fitted scaling, reference distributions and novelty models only on normal development
profiles. Validation compared Isolation Forest, Local Outlier Factor, statistical
scores, rules and fusion variants. The selection objective combined sample F1,
event F1 and anomaly-family recall. LOF plus the missing-data override won; a weighted
fusion had slightly higher sample F1 but more extra event reports.

Before generating the final eight flights I froze code, configuration and fitted-state
hashes. The final sample F1 was 0.951. One-to-one event matching found all 30 true
episodes with six false reports; duplicate fragments were penalized. Later runs
reproduced the original inputs and predictions exactly.

The dashboard lets an engineer inspect signals, event intervals and evidence. But
this is an offline synthetic demonstrator: centered filtering uses future samples,
the same generator family underlies all splits, and there is no real-aircraft validation.
The next steps would be independently sourced data, more diverse normal behaviour,
common-mode sensor faults and causal processing, evaluated on a new holdout.”

## Recommended Git commit message

`Build reproducible V2 synthetic flight anomaly pipeline, frozen benchmark and review dashboard`

## Before publishing

Review the staged diff and choose a repository license intentionally. No license
grant was invented on the author's behalf. Generated V2 CSVs are ignored; retain
the compact reports, freeze/receipt, figures and original representative V1 data.
No remote push has been performed.
