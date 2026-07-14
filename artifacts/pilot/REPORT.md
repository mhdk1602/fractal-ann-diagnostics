# Authorization-first retrieval development pilot

> Status: synthetic development evidence only. These results do not confirm the paper hypotheses.

Every pilot search strategy was replayed for every query against exact authorized top-k ground truth. The unsafe global baseline tests the security accounting. It is not a deployable action.

The pilot fixes `M=3` as an intentionally sparse development graph so recall failures remain observable despite the 101-neighbor geometry probe. It is not a production recommendation.

| scenario | strategy | n | recall@10 | recall target | unauthorized context | p95 ms | effort proxy |
|---|---|---:|---:|---:|---:|---:|---:|
| aligned | exact-authorized | 80 | 1.000 | 1.000 | 0 | 0.050 | 754.5 |
| aligned | hnsw-high | 80 | 0.993 | 1.000 | 0 | 0.097 | 512.0 |
| aligned | hnsw-low | 80 | 0.964 | 0.925 | 0 | 0.044 | 128.0 |
| aligned | unsafe-unfiltered | 80 | 0.961 | 0.912 | 0 | 0.042 | 10.0 |
| embedding-drift | exact-authorized | 80 | 1.000 | 1.000 | 0 | 0.042 | 754.5 |
| embedding-drift | hnsw-high | 80 | 0.994 | 1.000 | 0 | 0.090 | 512.0 |
| embedding-drift | hnsw-low | 80 | 0.968 | 0.938 | 0 | 0.042 | 128.0 |
| embedding-drift | unsafe-unfiltered | 80 | 0.941 | 0.925 | 20 | 0.039 | 10.0 |
| policy-scrambled | exact-authorized | 80 | 1.000 | 1.000 | 0 | 0.059 | 754.5 |
| policy-scrambled | hnsw-high | 80 | 0.995 | 1.000 | 0 | 0.111 | 512.0 |
| policy-scrambled | hnsw-low | 80 | 0.990 | 1.000 | 0 | 0.036 | 128.0 |
| policy-scrambled | unsafe-unfiltered | 80 | 0.267 | 0.000 | 586 | 0.045 | 10.0 |

## Controller-selected outcomes

The fixed development rule selected these actions. Thresholds were set on the synthetic engineering tier and cannot be carried into a confirmatory claim without the sealed calibration procedure.

| scenario | selected strategy | n | recall@10 | recall target | unauthorized context | effort proxy |
|---|---|---:|---:|---:|---:|---:|
| aligned | hnsw-high | 2 | 0.900 | 1.000 | 0 | 512.0 |
| aligned | hnsw-low | 78 | 0.965 | 0.923 | 0 | 128.0 |
| embedding-drift | hnsw-high | 62 | 0.992 | 1.000 | 0 | 512.0 |
| embedding-drift | hnsw-low | 18 | 0.989 | 1.000 | 0 | 128.0 |
| policy-scrambled | exact-authorized | 45 | 1.000 | 1.000 | 0 | 753.7 |
| policy-scrambled | hnsw-high | 35 | 0.997 | 1.000 | 0 | 512.0 |

## Interpretation boundary

The pilot tests code paths, exact policy-conditioned truth, counterfactual action replay, and fail-closed accounting. It does not establish external validity, answer faithfulness, production latency, or incremental value from fractal features. Those claims remain behind the gates in `research/preregistration.md`.
