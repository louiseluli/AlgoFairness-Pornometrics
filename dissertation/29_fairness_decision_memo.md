# Fairness Decision Memo

This memo consolidates **Step 21** model evaluations (if available) and the **Step 25** accuracy–fairness Pareto frontier to recommend a single operating point suitable for productization.

## Selected operating point
- **Model**: RF postproc
- **Accuracy**: 0.854
- **Fairness**: 0.758
- **Criterion**: Knee (distance to ideal)

## Caveats & next steps
- Metrics are point estimates within [0,1]. We recommend monitoring with uncertainty (bootstrap CIs) and periodic re-evaluation under temporal drift.
- If two candidates are indistinguishable, we break ties in favour of **higher fairness** to respect equity goals with minimal engagement loss.
- Deployment guidance: re-rank within a guard-rail region around the selected point; log exposures and auditable fairness telemetry.

*Notes:* Titles may be non-English; tags/categories (MPU) keep semantics interpretable.
