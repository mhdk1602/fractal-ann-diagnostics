"""v0.1.x calibration on 4-5 ANN-benchmark datasets.

Extends the v0.1.0 two-dataset run (MNIST, Fashion-MNIST) to the canonical
ANN-benchmarks quartet plus an optional fifth:

  - mnist-784-euclidean
  - fashion-mnist-784-euclidean
  - glove-25-angular
  - sift-128-euclidean
  - nytimes-256-angular           (optional; skipped if it 403s or times out)

For each dataset we:
  1. Download via ``load_ann_benchmark`` into ``./data/ann_cache`` (the loader
     sets a custom User-Agent to get past Cloudflare).
  2. Subsample the train split to 5000 vectors (deterministic seed).
  3. Compute the full descriptor panel including ``multifractal_width``. If a
     single dataset's descriptor pass exceeds ``MFW_SAMPLE_BUDGET_SECONDS``,
     we fall back to a 1500-point subsample for the multifractal step only;
     the other descriptors stay at 5000.
  4. Run ``diagnose()`` and record the recommendation + predicted recall drop.

The script then writes ``experiments/calibration-v0.1.0.md`` containing:
  - The descriptor / recommendation table.
  - An Analysis section with: rule-fire histogram, per-dataset comparison
    against practitioner intuition, the specific rules that misfire, the
    threshold change that would fix them, and a paragraph stating the v0.2.0
    calibration target.

Run::

    python experiments/calibrate_v0_1_0.py

This script does **not** modify the rule cascade. Threshold tweaks belong to
v0.2.0.
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

# Make the in-repo ``src/`` importable so this script runs from a clean
# checkout without requiring ``pip install -e .`` first.
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402

from fractal_ann_diagnostics.benchmark import load_ann_benchmark  # noqa: E402
from fractal_ann_diagnostics.descriptors import (  # noqa: E402
    DescriptorPanel,
    correlation_dimension,
    hubness,
    lid_mle,
    multifractal_width,
)
from fractal_ann_diagnostics.diagnostic import _recommend  # noqa: E402

DATASETS_CORE = (
    "mnist-784-euclidean",
    "fashion-mnist-784-euclidean",
    "glove-25-angular",
    "sift-128-euclidean",
)
DATASETS_OPTIONAL = ("nytimes-256-angular",)

CACHE_DIR = Path("./data/ann_cache").resolve()
OUTPUT_PATH = Path(__file__).resolve().parent / "calibration-v0.1.0.md"

SAMPLE_SIZE = 5000           # subsample for the descriptor pass
# correlation_dimension and multifractal_width both materialise an
# (n, n, d) broadcast for the pairwise-distance step. At n=5000, d=784 that
# is ~78 GB. We cap those two descriptors at D2_MFW_SAMPLE_SIZE to keep the
# script runnable on a 32 GB workstation. lid_mle and hubness avoid the
# broadcast (they use sklearn NearestNeighbors) and so run at the full
# SAMPLE_SIZE. v0.2.0 should rewrite the pairwise computation in chunks so
# this distinction goes away.
D2_MFW_SAMPLE_SIZE = 2000
MFW_FALLBACK_SIZE = 1500     # further fallback if mfw exceeds the budget
MFW_SAMPLE_BUDGET_SECONDS = 300.0  # 5 min per dataset before we shrink mfw

# Practitioner ground truth for the recommendation column. The ANN-benchmarks
# leaderboard at https://ann-benchmarks.com/ shows HNSW (hnswlib / nmslib) at
# or near the Pareto frontier for SIFT-128-euclidean, GloVe-25-angular,
# GloVe-100-angular, GloVe-200-angular, NYTimes-256-angular, Fashion-MNIST,
# and MNIST at recall@10. IVF backends (FAISS-IVF, FAISS-IVFPQ) trail HNSW on
# all of these. Source: ann-benchmarks/ann-benchmarks README and the leader
# board plots at https://ann-benchmarks.com/.
PRACTITIONER_INDEX: dict[str, str] = {
    "mnist-784-euclidean": "hnsw",
    "fashion-mnist-784-euclidean": "hnsw",
    "glove-25-angular": "hnsw",
    "sift-128-euclidean": "hnsw",
    "nytimes-256-angular": "hnsw",
}


def _subsample(x: np.ndarray, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if len(x) <= k:
        return x
    idx = rng.choice(len(x), size=k, replace=False)
    return x[idx]


def _descriptors_with_mfw_timeout(
    vectors: np.ndarray,
    sample_size: int,
    d2_mfw_sample_size: int,
    mfw_fallback_size: int,
    mfw_budget_seconds: float,
    rng_seed: int,
) -> tuple[DescriptorPanel, dict[str, float]]:
    """Compute the descriptor panel.

    correlation_dimension and multifractal_width are computed at
    ``d2_mfw_sample_size`` because they allocate an (n, n, d) broadcast that
    blows up beyond ~2000 in 784 dim. lid_mle and hubness use kd-tree
    neighbours and run at the full ``sample_size``.

    If the multifractal step exceeds ``mfw_budget_seconds`` we redo only that
    step at ``mfw_fallback_size``.
    """
    n, d = vectors.shape
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    rng = np.random.default_rng(rng_seed)
    d2 = correlation_dimension(vectors, sample_size=d2_mfw_sample_size, rng=rng)
    timings["d2_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    rng = np.random.default_rng(rng_seed + 1)
    lid = lid_mle(vectors, sample_size=sample_size, rng=rng)
    timings["lid_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    rng = np.random.default_rng(rng_seed + 2)
    hub = hubness(vectors, sample_size=sample_size, rng=rng)
    timings["hub_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    rng = np.random.default_rng(rng_seed + 3)
    mfw = multifractal_width(vectors, sample_size=d2_mfw_sample_size, rng=rng)
    mfw_elapsed = time.perf_counter() - t0
    if mfw_elapsed > mfw_budget_seconds:
        print(
            f"  [mfw] {mfw_elapsed:.1f}s > budget {mfw_budget_seconds:.0f}s; "
            f"falling back to sample_size={mfw_fallback_size}."
        )
        t0 = time.perf_counter()
        rng = np.random.default_rng(rng_seed + 4)
        mfw = multifractal_width(vectors, sample_size=mfw_fallback_size, rng=rng)
        timings["mfw_s"] = mfw_elapsed + (time.perf_counter() - t0)
        timings["mfw_subsample"] = float(mfw_fallback_size)
    else:
        timings["mfw_s"] = mfw_elapsed
        timings["mfw_subsample"] = float(d2_mfw_sample_size)

    panel = DescriptorPanel(
        correlation_dimension=d2,
        lid_distribution=lid,
        multifractal_width=mfw,
        hubness_skew=hub,
        ambient_dimension=d,
        n_points=n,
    )
    return panel, timings


def _which_rule_fires(panel: DescriptorPanel) -> int:
    """Return 1..5 indicating which rule of the v0.1.1 cascade fired."""
    d2 = panel.correlation_dimension
    ambient = panel.ambient_dimension
    n = panel.n_points
    lid_p50 = float(np.quantile(panel.lid_distribution, 0.5))
    lid_p95 = float(np.quantile(panel.lid_distribution, 0.95))
    skew = panel.hubness_skew

    if ambient > 0 and (d2 / ambient) > 0.7:
        return 1
    if skew > 2.0:
        return 2
    if lid_p95 > 2.0 * lid_p50 and n > 1_000_000:
        return 3
    if d2 < 10.0 and n < 50_000:
        return 4
    return 5


def _load_one(name: str) -> np.ndarray | None:
    """Return the train split, or None on download failure."""
    try:
        ds = load_ann_benchmark(name, CACHE_DIR)
        return ds.train
    except Exception as e:  # noqa: BLE001
        print(f"[error] {name}: download/load failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None


def main() -> None:
    rows: list[dict[str, str]] = []
    panels: dict[str, DescriptorPanel] = {}
    recs: dict[str, str] = {}
    rules_fired: dict[str, int] = {}
    failures: dict[str, str] = {}

    targets = list(DATASETS_CORE) + list(DATASETS_OPTIONAL)
    for name in targets:
        print(f"\n--- {name} ---")
        train = _load_one(name)
        if train is None:
            failures[name] = "download/load failed"
            rows.append({
                "dataset": name,
                "n": "n/a",
                "d": "n/a",
                "D2": "n/a",
                "lid_p50": "n/a",
                "lid_p95": "n/a",
                "hubness_skew": "n/a",
                "multifractal_width": "n/a",
                "recommended": "**[FAILED: download/load failed]**",
                "predicted_drop": "n/a",
            })
            continue

        print(f"  train shape: {train.shape}")
        x = _subsample(train, SAMPLE_SIZE, seed=20250514)
        print(f"  descriptor sample: {x.shape}")

        try:
            panel, timings = _descriptors_with_mfw_timeout(
                x,
                sample_size=SAMPLE_SIZE,
                d2_mfw_sample_size=D2_MFW_SAMPLE_SIZE,
                mfw_fallback_size=MFW_FALLBACK_SIZE,
                mfw_budget_seconds=MFW_SAMPLE_BUDGET_SECONDS,
                rng_seed=20250514,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[error] {name}: descriptor pass failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            failures[name] = f"descriptor pass failed ({type(e).__name__})"
            rows.append({
                "dataset": name,
                "n": str(len(train)),
                "d": str(train.shape[1]),
                "D2": "n/a",
                "lid_p50": "n/a",
                "lid_p95": "n/a",
                "hubness_skew": "n/a",
                "multifractal_width": "n/a",
                "recommended": f"**[FAILED: {type(e).__name__}]**",
                "predicted_drop": "n/a",
            })
            continue

        # Re-use the panel we just computed. Rewrite n_points to len(train) so
        # rule 4 (which depends on n) sees the full-corpus cardinality rather
        # than the descriptor subsample size.
        panel = DescriptorPanel(
            correlation_dimension=panel.correlation_dimension,
            lid_distribution=panel.lid_distribution,
            multifractal_width=panel.multifractal_width,
            hubness_skew=panel.hubness_skew,
            ambient_dimension=panel.ambient_dimension,
            n_points=len(train),
        )
        index, drop, rationale = _recommend(panel)
        rule = _which_rule_fires(panel)

        panels[name] = panel
        recs[name] = index
        rules_fired[name] = rule

        lid_p50 = float(np.quantile(panel.lid_distribution, 0.5))
        lid_p95 = float(np.quantile(panel.lid_distribution, 0.95))

        rows.append({
            "dataset": name,
            "n": f"{panel.n_points}",
            "d": f"{panel.ambient_dimension}",
            "D2": f"{panel.correlation_dimension:.3f}",
            "lid_p50": f"{lid_p50:.3f}",
            "lid_p95": f"{lid_p95:.3f}",
            "hubness_skew": f"{panel.hubness_skew:.3f}",
            "multifractal_width": (
                f"{panel.multifractal_width:.3f}"
                if np.isfinite(panel.multifractal_width)
                else "nan"
            ),
            "recommended": index,
            "predicted_drop": f"{drop:.3f}",
        })
        print(
            f"  D2={panel.correlation_dimension:.3f} ambient={panel.ambient_dimension} "
            f"D2/d={panel.correlation_dimension / panel.ambient_dimension:.3f} "
            f"lid_p50={lid_p50:.2f} lid_p95={lid_p95:.2f} "
            f"hubness={panel.hubness_skew:.2f} mfw={panel.multifractal_width:.3f}"
        )
        print(f"  recommended: {index}  (rule {rule})  predicted_drop={drop:.3f}")
        print(f"  rationale:   {rationale}")
        print(f"  timings:     {timings}")

    _write_markdown(rows, panels, recs, rules_fired, failures)
    print(f"\nWrote {OUTPUT_PATH}")


# --------------------------------------------------------------------------
# Markdown output
# --------------------------------------------------------------------------

HEADERS = [
    "dataset",
    "n",
    "d",
    "D2",
    "lid_p50",
    "lid_p95",
    "hubness_skew",
    "multifractal_width",
    "recommended",
    "predicted_drop",
]


def _rule_label(i: int) -> str:
    return {
        1: "Rule 1 (D2/ambient > 0.7 -> flat-nsw)",
        2: "Rule 2 (hubness_skew > 2.0 -> flat-nsw)",
        3: "Rule 3 (heterogeneous LID and n > 1e6 -> diskann)",
        4: "Rule 4 (D2 < 10 and n < 5e4 -> ivf)",
        5: "Rule 5 (default -> hnsw)",
    }[i]


def _build_analysis(
    panels: dict[str, DescriptorPanel],
    recs: dict[str, str],
    rules_fired: dict[str, int],
    failures: dict[str, str],
) -> list[str]:
    lines: list[str] = []
    lines.append("## Analysis")
    lines.append("")
    if not panels:
        lines.append(
            "No datasets completed the descriptor pass; the rule-fire histogram "
            "and threshold analysis below are empty. See the failure list."
        )
        if failures:
            lines.append("")
            lines.append("**Failures:**")
            lines.append("")
            for name, reason in failures.items():
                lines.append(f"- `{name}`: {reason}")
        return lines

    # 1) Rule-fire histogram
    lines.append("### Rule-fire histogram")
    lines.append("")
    counts = {i: 0 for i in range(1, 6)}
    for r in rules_fired.values():
        counts[r] += 1
    lines.append("| rule | description | datasets | which |")
    lines.append("|---|---|---|---|")
    for i in range(1, 6):
        which = ", ".join(sorted(name for name, r in rules_fired.items() if r == i)) or "—"
        lines.append(f"| {i} | {_rule_label(i)} | {counts[i]} | {which} |")
    lines.append("")

    # 2) Match against practitioner intuition
    lines.append("### Practitioner intuition")
    lines.append("")
    lines.append(
        "On the canonical ANN-benchmarks leaderboard "
        "(https://ann-benchmarks.com/), HNSW (hnswlib, nmslib) sits at or "
        "near the recall@10 Pareto frontier for SIFT-128-euclidean, "
        "GloVe-25-angular, NYTimes-256-angular, MNIST-784-euclidean and "
        "Fashion-MNIST-784-euclidean. FAISS-IVF and FAISS-IVFPQ trail HNSW "
        "on every one of these. So a v0.1.x recommender that defaults to "
        "HNSW on all five is matching practitioner consensus; any other "
        "recommendation needs a strong justification from the descriptors."
    )
    lines.append("")
    lines.append("| dataset | recommended | practitioner default | match? |")
    lines.append("|---|---|---|---|")
    matches = 0
    mismatches: list[tuple[str, str, str]] = []
    for name, rec in recs.items():
        practitioner = PRACTITIONER_INDEX.get(name, "?")
        ok = rec == practitioner
        if ok:
            matches += 1
        else:
            mismatches.append((name, rec, practitioner))
        lines.append(f"| {name} | {rec} | {practitioner} | {'yes' if ok else 'no'} |")
    lines.append("")
    lines.append(
        f"{matches}/{len(recs)} match. Mismatches: "
        + (", ".join(f"{n} -> {r} (expected {p})" for n, r, p in mismatches) or "none")
        + "."
    )
    lines.append("")

    # 3) Threshold-sensitivity diagnostic
    lines.append("### Where the cascade misfires")
    lines.append("")
    rule1_fires: list[tuple[str, float]] = []
    rule4_fires: list[tuple[str, float, int]] = []
    for name, panel in panels.items():
        ratio = panel.correlation_dimension / max(panel.ambient_dimension, 1)
        if rules_fired[name] == 1:
            rule1_fires.append((name, ratio))
        if rules_fired[name] == 4:
            rule4_fires.append((name, panel.correlation_dimension, panel.n_points))

    if rule1_fires:
        lines.append("**Rule 1 (D2/ambient > 0.7 -> flat-nsw) fires on:**")
        lines.append("")
        for name, ratio in rule1_fires:
            practitioner = PRACTITIONER_INDEX.get(name, "?")
            verdict = (
                "matches practitioner intuition"
                if practitioner == "flat-nsw"
                else f"contradicts practitioner default `{practitioner}`"
            )
            lines.append(f"- `{name}`: D2/ambient = {ratio:.3f}; {verdict}.")
        lines.append("")
        contradictions = [
            (n, r) for n, r in rule1_fires
            if PRACTITIONER_INDEX.get(n) and PRACTITIONER_INDEX[n] != "flat-nsw"
        ]
        if contradictions:
            max_ratio = max(r for _, r in contradictions)
            new_thresh = max(0.85, round(max_ratio + 0.05, 2))
            lines.append(
                f"Rule 1 is misfiring at threshold 0.7. To suppress every "
                f"contradicting dataset above, raise the threshold to roughly "
                f"`D2/ambient > {new_thresh}`. The Hub Highway Hypothesis itself "
                f"is qualitative on the cutoff; 0.7 was a guess, not a calibrated "
                f"boundary."
            )
            lines.append("")
    else:
        lines.append("**Rule 1 (D2/ambient > 0.7 -> flat-nsw):** did not fire on any dataset.")
        lines.append("")

    if rule4_fires:
        lines.append("**Rule 4 (D2 < 10 and n < 5e4 -> ivf) fires on:**")
        lines.append("")
        for name, d2, n in rule4_fires:
            practitioner = PRACTITIONER_INDEX.get(name, "?")
            verdict = (
                "matches practitioner intuition"
                if practitioner == "ivf"
                else f"contradicts practitioner default `{practitioner}`"
            )
            lines.append(f"- `{name}`: D2={d2:.3f}, n={n}; {verdict}.")
        lines.append("")
        contradictions = [
            (n, d2, npts) for n, d2, npts in rule4_fires
            if PRACTITIONER_INDEX.get(n) and PRACTITIONER_INDEX[n] != "ivf"
        ]
        if contradictions:
            max_n = max(npts for _, _, npts in contradictions)
            new_n = max(10_000, max_n // 2)
            lines.append(
                f"Rule 4 still misfires on at least one dataset where "
                f"practitioner intuition is HNSW. The cardinality cutoff was "
                f"already tightened from 1e5 to 5e4 in v0.1.1; the next "
                f"calibration should drop it further (toward n < {new_n}) "
                f"or replace the hand-set boundary with a learned classifier."
            )
            lines.append("")
    else:
        lines.append(
            "**Rule 4 (D2 < 10 and n < 5e4 -> ivf):** did not fire on any dataset in "
            "this run. Either the v0.1.1 tightening to 5e4 is holding, or every "
            "tested dataset has n >= 5e4."
        )
        lines.append("")

    # 4) v0.2.0 calibration target
    lines.append("### v0.2.0 calibration target")
    lines.append("")
    if mismatches:
        ds_str = ", ".join(n for n, _, _ in mismatches)
        lines.append(
            f"v0.2.0 should learn the rule cascade's thresholds from "
            f"ANN-benchmarks rather than hand-setting them. The concrete "
            f"evidence from this run: the cascade disagrees with the "
            f"practitioner default on `{ds_str}`. The two thresholds most "
            f"in need of calibration are the Hub Highway cutoff "
            f"(`D2 / ambient`, currently 0.7) and the IVF cardinality cutoff "
            f"(currently n < 5e4); both were chosen by reading papers, not by "
            f"fitting to held-out ANN-benchmarks recall. v0.2.0 should turn the "
            f"five-rule cascade into a classifier trained on per-dataset recall "
            f"@10 across the ANN-benchmarks corpus, with the descriptors as "
            f"features. The current rules become its prior, not its posterior."
        )
    else:
        lines.append(
            "Every dataset in this run matches practitioner intuition. The "
            "v0.2.0 calibration target therefore shifts from fixing wrong "
            "recommendations to calibrating the `predicted_drop` ramp (`(lid_p95 "
            "- 5) / 50`, currently uncalibrated) and replacing the fixed 0.5 "
            "confidence with a Bayesian posterior over recall drop fit to "
            "ANN-benchmarks recall@10."
        )
    lines.append("")

    if failures:
        lines.append("### Failures")
        lines.append("")
        for name, reason in failures.items():
            lines.append(f"- `{name}`: {reason}")
        lines.append("")
    return lines


def _write_markdown(
    rows: list[dict[str, str]],
    panels: dict[str, DescriptorPanel],
    recs: dict[str, str],
    rules_fired: dict[str, int],
    failures: dict[str, str],
) -> None:
    lines = [
        "# v0.1.x calibration",
        "",
        "Descriptor panel and rule-based recommendation on the canonical "
        "ANN-benchmarks quartet plus one optional dataset (NYTimes). Generated "
        "by `experiments/calibrate_v0_1_0.py`. Train splits subsampled to "
        f"{SAMPLE_SIZE} points for lid_mle and hubness; "
        f"`correlation_dimension` and `multifractal_width` use a tighter "
        f"{D2_MFW_SAMPLE_SIZE}-point cap because both materialise an (n, n, d) "
        "broadcast that exhausts RAM at d=784, n=5000. "
        f"`multifractal_width` falls back to {MFW_FALLBACK_SIZE} points if its "
        f"pass exceeds {int(MFW_SAMPLE_BUDGET_SECONDS)} s.",
        "",
        "| " + " | ".join(HEADERS) + " |",
        "|" + "|".join(["---"] * len(HEADERS)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[h] for h in HEADERS) + " |")
    lines.append("")
    lines.append(
        "Notes. `D2` is the Grassberger-Procaccia correlation dimension. "
        "`lid_p50` / `lid_p95` are the 50th / 95th percentiles of the per-point "
        "MLE local intrinsic dimensionality. `hubness_skew` is the skewness of "
        "the reverse-kNN count distribution (Radovanović et al. 2010). "
        "`multifractal_width` is α_max − α_min of the MFDFA singularity "
        "spectrum on all-pairs distances. `predicted_drop` is meaningful only "
        "when `recommended == hnsw`; the other rules zero it out because "
        "v0.1.x does not yet model recall drop on flat-NSW / IVF / DiskANN."
    )
    lines.append("")
    lines.extend(_build_analysis(panels, recs, rules_fired, failures))
    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
