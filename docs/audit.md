# V1 audit — before V2 implementation

Reviewed all nine `src/*.py` modules, root `main.py`, all four CSVs,
requirements, ignore rules, the existing timeline, and Git history/status.
The single existing commit is `2683f5e`; pre-existing local changes and
untracked V1 files are preserved. No remote push is part of this work.

- README was empty; no tests, dashboard, or machine-readable reports existed.
- Four CSVs each contain 12,001 samples at 10 Hz. The original synthetic CSV
  duplicates the development scenario. Yaw dropouts are retained as NaNs.
- V1 imputer/scaler correctly fit only development data. IF fits mixed normal
  and anomalous development data with contamination 0.08. Labels are not features.
- Features and ground-truth extraction assume 10 Hz via fixed sample counts
  and a 0.2-second gap. No schema or time-grid validation exists.
- Centered median rules use future information: appropriate for offline review,
  unsuitable for a claim of causal real-time operation.
- Absolute altitude-rate threshold cannot distinguish drift from climb/descent.
  Nominal conditions vary little; generalization evidence is correspondingly weak.
- RPM median window is shorter than some injected drops, causing missed interiors.
- Event matching is many-to-many: duplicate reports receive no penalty. Empty
  event tables have no schema. Single-class confusion matrices can fail to unpack.
- Root main reads the historical final dataset on import. Console-only metrics
  are difficult to reuse. Historical final is already inspected, so not V2 test data.
- Existing timeline has indistinguishable overlay colors and an inferred legend;
  it omits altitude despite drift being a major weakness.
- No clearly unused functions in the small pipeline; repeated orchestration and
  print blocks are the principal duplication. Keep historical modules intact.

V1 entry point is archived in `archive/v1/main.py` (historical reference only).
Its console output and original figure are preserved in `reports/v1/`.
V2 lives in `src/v2/`; this avoids silently changing historical experiments.
Generated V2 CSVs and local model caches are excluded from Git; reproducible
configs, first-evaluation receipts, compact reports and portfolio figures are kept.

V2 design decision: add a noisy independent vertical-speed measurement. Altitude
drift is not generally identifiable from altitude alone during arbitrary climbs.
Cross-sensor trend consistency is therefore an explicit extra assumption, not a
claim that a smarter threshold solved the unobservable single-sensor problem.
