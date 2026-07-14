# v0.1.x calibration

> **Historical artifact, superseded by v0.2.** This run did not measure backend recall or regret,
> used Euclidean descriptors on angular datasets, and reported an MFDFA statistic that changes
> under row permutation. Its numeric table is retained as an integrity record, not current
> evidence. See the [current draft protocol](../research/preregistration.md) and
> [development pilot](../artifacts/pilot/REPORT.md).

Descriptor panel and rule-based recommendation on the canonical ANN-benchmarks quartet plus one optional dataset (NYTimes). Produced by `experiments/calibrate_v0_1_0.py`. Train splits subsampled to 5000 points for lid_mle and hubness; `correlation_dimension` and `multifractal_width` use a tighter 2000-point cap because both materialise an (n, n, d) broadcast that exhausts RAM at d=784, n=5000. `multifractal_width` falls back to 1500 points if its pass exceeds 300 s.

| dataset | n | d | D2 | lid_p50 | lid_p95 | hubness_skew | multifractal_width | recommended | predicted_drop |
|---|---|---|---|---|---|---|---|---|---|
| mnist-784-euclidean | 60000 | 784 | 9.404 | 11.929 | 23.720 | 0.796 | 0.234 | hnsw | 0.300 |
| fashion-mnist-784-euclidean | 60000 | 784 | 4.869 | 9.892 | 20.628 | 1.775 | 0.166 | hnsw | 0.300 |
| glove-25-angular | 1183514 | 25 | 6.759 | 13.882 | 24.345 | 2.415 | 0.232 | flat-nsw | 0.000 |
| sift-128-euclidean | 1000000 | 128 | 6.528 | 15.521 | 23.033 | 1.910 | 0.241 | hnsw | 0.300 |
| nytimes-256-angular | 290000 | 256 | 34.746 | 41.026 | 48.365 | 40.055 | 0.255 | flat-nsw | 0.000 |

Notes. `D2` is the Grassberger-Procaccia correlation dimension. `lid_p50` / `lid_p95` are the 50th / 95th percentiles of the per-point MLE local intrinsic dimensionality. `hubness_skew` is the skewness of the reverse-kNN count distribution (Radovanović et al. 2010). `multifractal_width` is α_max − α_min of the MFDFA singularity spectrum on all-pairs distances. `predicted_drop` is meaningful only when `recommended == hnsw`; the other rules zero it out because v0.1.x does not yet model recall drop on flat-NSW / IVF / DiskANN.

## Analysis

### Rule-fire histogram

| rule | description | datasets | which |
|---|---|---|---|
| 1 | Rule 1 (D2/ambient > 0.7 -> flat-nsw) | 0 | — |
| 2 | Rule 2 (hubness_skew > 2.0 -> flat-nsw) | 2 | glove-25-angular, nytimes-256-angular |
| 3 | Rule 3 (heterogeneous LID and n > 1e6 -> diskann) | 0 | — |
| 4 | Rule 4 (D2 < 10 and n < 5e4 -> ivf) | 0 | — |
| 5 | Rule 5 (default -> hnsw) | 3 | fashion-mnist-784-euclidean, mnist-784-euclidean, sift-128-euclidean |

### Practitioner intuition

On the canonical ANN-benchmarks leaderboard (https://ann-benchmarks.com/), HNSW (hnswlib, nmslib) sits at or near the recall@10 Pareto frontier for SIFT-128-euclidean, GloVe-25-angular, NYTimes-256-angular, MNIST-784-euclidean and Fashion-MNIST-784-euclidean. FAISS-IVF and FAISS-IVFPQ trail HNSW on every one of these. So a v0.1.x recommender that defaults to HNSW on all five is matching practitioner consensus; any other recommendation needs a strong justification from the descriptors.

| dataset | recommended | practitioner default | match? |
|---|---|---|---|
| mnist-784-euclidean | hnsw | hnsw | yes |
| fashion-mnist-784-euclidean | hnsw | hnsw | yes |
| glove-25-angular | flat-nsw | hnsw | no |
| sift-128-euclidean | hnsw | hnsw | yes |
| nytimes-256-angular | flat-nsw | hnsw | no |

3/5 match. Mismatches: glove-25-angular -> flat-nsw (expected hnsw), nytimes-256-angular -> flat-nsw (expected hnsw).

### Where the cascade misfires

**Rule 1 (D2/ambient > 0.7 -> flat-nsw):** did not fire on any dataset.

**Rule 4 (D2 < 10 and n < 5e4 -> ivf):** did not fire on any dataset in this run. Either the v0.1.1 tightening to 5e4 is holding, or every tested dataset has n >= 5e4.

### v0.2.0 calibration target

v0.2.0 should learn the rule cascade's thresholds from ANN-benchmarks rather than hand-setting them. The concrete evidence from this run: the cascade disagrees with the practitioner default on `glove-25-angular, nytimes-256-angular`. The two thresholds most in need of calibration are the Hub Highway cutoff (`D2 / ambient`, currently 0.7) and the IVF cardinality cutoff (currently n < 5e4); both were chosen by reading papers, not by fitting to held-out ANN-benchmarks recall. v0.2.0 should turn the five-rule cascade into a classifier trained on per-dataset recall @10 across the ANN-benchmarks corpus, with the descriptors as features. The current rules become its prior, not its posterior.
