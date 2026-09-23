# V2 delivery summary

## Outcome

V1 was audited and preserved; V2 adds multi-flight synthetic generation, eight
temporal/spectral features, normal-only fitted preprocessing/novelty models,
validation-only selection, freeze/replay integrity, event evidence, a Streamlit
dashboard, six static figures, machine-readable reports and portfolio documentation.

Final architecture: sensor measurements → per-flight offline features → robust
development-fitted preprocessing → LOF novelty score >2 OR missing measurement →
raw sample alerts → event consolidation → explanatory reports/dashboard.

## Metrics

| Split | Precision | Recall | F1 | Event precision | Event recall | Event F1 | False reports | Missed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V1 original final | 0.613 | 0.666 | 0.638 | Different original matching | 1.000 | Not comparable | 2 nonoverlapping | 0 |
| V2 validation | 0.923 | 0.972 | 0.947 | 0.862 | 1.000 | 0.926 | 4 | 0 |
| V2 final | 0.928 | 0.976 | 0.951 | 0.833 | 1.000 | 0.909 | 6 | 0 |

V1 original counts: TP 646, FP 408, FN 324, TN 10,623; five true events, eight reports.
V2 validation: TP 3,757, FP 312, FN 108, TN 31,823; 25 true events, 29 reports.
V2 final: TP 4,769, FP 370, FN 118, TN 42,743; 30 true events, 36 reports.

Final sample recall by family: airspeed 95/95; altitude 2,776/2,893; RPM 566/566;
lateral oscillation 1,169/1,170; dropout 163/163. Two normal-only final flights have
zero alerts. Final false-report rate: 4.5 per simulated flight-hour.

Candidates included original raw/engineered IF, V1 rules/hybrid, and V2 IF, LOF,
statistical/rule detectors plus OR/weighted fusion. LOF plus the missingness rule won
the predeclared validation objective. Weighted LOF fusion slightly improved sample
F1 but worsened event reporting; OR fusion tied the simpler winner. Final benchmark
was not used for tuning, and no frozen numerical module was edited afterward.

## Files and structure

Modified by this work: `main.py`, `README.md`, `requirements.txt`, `.gitignore`, and
the V2 replacement `reports/figures/final_test_anomaly_timeline.png` (original preserved).

Created:

- `app.py`, `requirements-lock.txt`, `pytest.ini`, `.gitattributes`, `.streamlit/config.toml`.
- `src/v2/__init__.py`, `config.py`, `data.py`, `features.py`, `models.py`,
  `events.py`, `evaluation.py`, `experiments.py`, `protocol.py`, `reporting.py`.
- `tests/conftest.py`, `tests/test_pipeline.py`, `tests/test_artifacts.py`.
- `docs/audit.md`, `docs/methodology.md`, `docs/interview.md`,
  `docs/portfolio_copy.md`, and this summary.
- `archive/v1/main.py`, `reports/v1/original_run.txt`, original V1 figure copy.
- Freeze/first-evaluation JSON receipts; validation experiment CSV and selection;
  development/validation/final per-flight, event and type CSVs; final summary;
  V1 recalculated/model comparison CSVs; technical report; six V2 figures.
- Local regenerated `data/v2/{development,validation,final_benchmark,historical}`
  CSVs, deliberately ignored by Git.

Pre-existing user changes retained: `src/generate_data.py`, the original final CSV,
V1 event/evaluation/visualization modules and original main content archived before
replacement. No user work was reset; no commit or remote push was made.
The README contains the complete directory tree and links to the outputs.

## Verification

- `py -m pip install -r requirements.txt`: successful.
- `py main.py`: successful initial freeze/final run.
- `py main.py`: subsequent exact input/score/prediction/metric replay successful.
- `py -m pytest`: 21 tests passed; no intentionally skipped tests.
- `py -m streamlit run app.py`: server started; dashboard visually inspected.
- AppTest covered split changes, event focus and ground-truth/evaluation controls.
- All six figures visually reviewed; all README local links/images resolved.
- Frozen source hashes still match; Git whitespace check passed.

Initial sandbox restrictions blocked pytest temporary-directory cleanup and dependency
installation. Approved reruns succeeded. Temporary failed-test cache folders were
removed; these environment issues are not unresolved project failures.

## Commands

```powershell
py -m pip install -r requirements.txt
py main.py
py -m pytest
py -m streamlit run app.py
```

## Limits and next extensions

No real-aircraft validation; shared simplified generator; eight final flights;
nonoverlapping anomaly families; extra reference-sensor assumption; offline
look-ahead; six extra event reports. V1 and V2 are different tasks, not a controlled
head-to-head test. Future work: public flight data, common-mode/reference faults,
overlapping events and heavier-tailed noise, causal processing and latency validation.
Any revised detector must use a new future holdout.

Before GitHub publication, review the diff and choose an intended repository license.
There are no known failing functional checks. Recommended commit:

`Build reproducible V2 synthetic flight anomaly pipeline, frozen benchmark and review dashboard`

## Career materials

See [portfolio_copy.md](portfolio_copy.md) for the final 2–3-line CV entry,
100–180-word LinkedIn description, suggested skills, 30–60-second introduction and
2–3-minute engineering explanation. See [interview.md](interview.md) for 18 concise
questions/answers, including model choice, leakage, drift identifiability, signal
processing, event matching, real-time limitations and honest result interpretation.
