# Authorization-first retrieval development pilot

> Status: synthetic development evidence only. These results do not confirm the paper hypotheses.

Every action was replayed for every query against exact authorized top-k ground truth. The unsafe global baseline tests the security accounting. It is not a deployable action.

| scenario | strategy | n | recall@10 | evidence success | unauthorized context | p95 ms | effort proxy |
|---|---|---:|---:|---:|---:|---:|---:|
| aligned | exact-authorized | 80 | 1.000 | 1.000 | 0 | 0.055 | 754.5 |
| aligned | hnsw-high | 80 | 1.000 | 1.000 | 0 | 0.051 | 96.0 |
| aligned | hnsw-low | 80 | 0.919 | 0.800 | 0 | 0.025 | 10.0 |
| aligned | unsafe-unfiltered | 80 | 0.918 | 0.812 | 0 | 0.024 | 10.0 |
| embedding-drift | exact-authorized | 80 | 1.000 | 1.000 | 0 | 0.048 | 754.5 |
| embedding-drift | hnsw-high | 80 | 1.000 | 1.000 | 0 | 0.038 | 96.0 |
| embedding-drift | hnsw-low | 80 | 0.835 | 0.537 | 0 | 0.021 | 10.0 |
| embedding-drift | unsafe-unfiltered | 80 | 0.838 | 0.537 | 0 | 0.015 | 10.0 |
| policy-scrambled | exact-authorized | 80 | 1.000 | 1.000 | 0 | 0.047 | 754.5 |
| policy-scrambled | hnsw-high | 80 | 1.000 | 1.000 | 0 | 0.029 | 96.0 |
| policy-scrambled | hnsw-low | 80 | 0.924 | 0.787 | 0 | 0.014 | 10.0 |
| policy-scrambled | unsafe-unfiltered | 80 | 0.267 | 0.000 | 586 | 0.014 | 10.0 |

## Controller-selected outcomes

The fixed development rule selected these actions. Thresholds were set on the synthetic engineering tier and cannot be carried into a confirmatory claim without the sealed calibration procedure.

| scenario | selected strategy | n | recall@10 | evidence success | unauthorized context | effort proxy |
|---|---|---:|---:|---:|---:|---:|
| aligned | hnsw-low | 80 | 0.919 | 0.800 | 0 | 10.0 |
| embedding-drift | hnsw-high | 56 | 1.000 | 1.000 | 0 | 96.0 |
| embedding-drift | hnsw-low | 24 | 0.867 | 0.667 | 0 | 10.0 |
| policy-scrambled | exact-authorized | 2 | 1.000 | 1.000 | 0 | 758.0 |
| policy-scrambled | hnsw-high | 78 | 1.000 | 1.000 | 0 | 96.0 |

## Interpretation boundary

The pilot tests code paths, exact policy-conditioned truth, counterfactual action replay, and fail-closed accounting. It does not establish external validity, answer faithfulness, production latency, or incremental value from fractal features. Those claims remain behind the gates in `research/preregistration.md`.
