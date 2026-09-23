# Technical interview preparation

1. **Why anomaly detection instead of supervised classification?**
   Real anomaly labels are often scarce and do not cover every failure mode. A
   novelty model learns normal behaviour. Here synthetic labels still supervise
   validation threshold selection, so the full protocol is semi-supervised.

2. **Why compare Isolation Forest and LOF?**
   IF isolates unusual points through random partitions; LOF compares a point's
   local density with its neighbours. They provide different geometric notions
   of novelty. LOF won the predeclared validation objective on these features;
   that does not establish that LOF is universally better.

3. **What does contamination mean?**
   In V1 it determines the development-score cutoff associated with an assumed
   outlier fraction of 0.08. It is not a measured probability and does not force
   8% of every future flight to be anomalous. V2 selects its score threshold on validation.

4. **Why scale inputs?**
   LOF uses distances, so a feature measured in large numerical units could
   otherwise dominate. RobustScaler uses development medians and interquartile
   ranges. Scaling does not give the same benefit to every model, notably IF.

5. **Why preserve missingness before imputation?**
   Interpolation can make a dropout look normal. A separate missing indicator and
   deterministic override preserve the original data-quality evidence.

6. **What is feature engineering?**
   Transforming measurements into informative quantities, such as residuals or
   oscillatory energy. V1 already showed a jump in validation F1 from 0.395 to
   0.574 after temporal features. V2 uses eight interpretable features.

7. **Why is altitude drift difficult?**
   A trend in altitude can be an actual climb. Without independent information
   the cause is not identifiable in general. V2 adds a noisy vertical-speed
   reference and tests trend consistency. Correlated/common-mode faults remain a limitation.

8. **Precision, recall and F1?**
   Precision is TP/(TP+FP); recall is TP/(TP+FN); F1 is their harmonic mean.
   Higher recall can increase review workload if precision falls. Final sample
   F1 was 0.951, not perfect classification despite detecting all 30 events.

9. **Why evaluate events and why one-to-one matching?**
   Engineers review episodes. Hundreds of alerts during one episode should not
   earn hundreds of successful detections. One-to-one matching with IoU ≥0.10
   penalizes duplicate reports and merges that conceal separate events.

10. **What are data leakage and a clean split?**
    Leakage uses unavailable information from validation/test to fit or select
    a system. Fit references/models on normal development; select on validation;
    generate final flights only after freezing. Never tune after looking at final
    labels/results. The old V1 final flight was already inspected and was retired as a holdout.

11. **Why combine rules with ML? Can a rule outperform ML?**
    Missing data are directly observable; a rule is simpler and more reliable
    than hoping a model notices imputed samples. ML can combine temporal evidence.
    A statistical baseline was also competitive, so model complexity is not the goal.

12. **What is Welch PSD and did you use it?**
    Welch estimates power spectral density by averaging periodograms from
    windowed, usually overlapping segments. V2 instead uses a Butterworth bandpass
    and rolling RMS. The chosen band isolates synthetic oscillations from slower
    manoeuvres without calculating an entire spectrum at every timestamp.

13. **What are rolling standard deviation and RMS?**
    Standard deviation measures spread about the local mean. RMS is the square
    root of mean squared amplitude. For a near-zero-mean bandpassed signal RMS
    represents local oscillation energy. Window length trades temporal resolution
    against stability; filtered event boundaries can smear into adjacent samples.

14. **What would have to change for real-time operation?**
    Replace centered windows, bidirectional interpolation and `sosfiltfilt` with
    causal alternatives; carry state between chunks; handle late/irregular data;
    revalidate latency, startup behaviour, event boundaries and alert rates on new data.

15. **What are the main limits, and why not deep learning?**
    Eight final synthetic flights do not represent aircraft operational diversity.
    The generator is shared across splits, anomalies do not overlap, and vertical
    speed is assumed independently informative. Classical methods are fast and
    explainable; this dataset does not justify neural-network complexity.

16. **What failed or did not help?**
    V1 features alone did not fix altitude generalization. Weighted V2 LOF fusion
    slightly increased sample F1 but worsened event reporting; OR fusion tied the
    simpler model. Final split-event false reports remain. No post-test retuning occurred.

17. **Is 0.951 versus 0.638 a fair head-to-head improvement?**
    No. Those are results on different synthetic tasks with an extra V2 sensor.
    Report both as historical progression, disclose the change, and avoid causal claims.

18. **How would you validate on real data?**
    Establish provenance/permission, sensor definitions, time alignment and
    quality; involve domain experts in labels; split by flight/campaign; measure
    false-alert workload, unfamiliar faults and uncertainty. Certification and
    operational deployment would require substantially different assurance work.
