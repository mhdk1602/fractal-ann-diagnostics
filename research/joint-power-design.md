# Joint H2/H3 family-cluster power design

## Decision this artifact makes

The design report selects the smallest registered number of query families per corpus for which
the simultaneous lower Monte Carlo confidence bound reaches the target power for every primary
endpoint and for their conjunction. The fixed selection family contains six candidate counts by
two required scenarios, or 12 cells. It receives a familywise confidence guarantee of at least
95% by assigning each cell one-sided alpha `0.05 / 12`. A point estimate of 0.90 is retained but
cannot qualify a cell when its multiplicity-adjusted lower bound is below 0.90.

The unit sampled is a query family. The five corpora stay fixed and receive equal weight. Nested
subject-policy draws, timing repeats, model predictions, retrieval outcomes, evidence outcomes,
and denied-emission counts move together with their family. This retains the dependence observed
across H2 and H3 instead of combining separately simulated endpoints after the fact.

The implementation is in
[`joint_power_design.py`](../src/fractal_ann_diagnostics/joint_power_design.py). Its config,
development panel, and report use closed canonical JSON schemas with one terminal newline.

## Admissible source data

`DevelopmentScenarioPanel` accepts only `development-fit` or `development-calibration`. The class
rejects `sealed`, missing corpora, nonfinite values, boundary probabilities, duplicate row IDs,
and invalid evidence annotations. The design therefore cannot consume sealed outcomes through its
typed path.

Each row contains the minimum raw material needed to recompute all registered endpoints:

- the binary low-effort failure label and both frozen model probabilities;
- proposed and static-comparator request latency;
- the observed zero-based execution position for each paired action;
- paired retrieval-target attainment;
- paired complete-evidence sufficiency for the three evidence corpora; and
- the number of denied items emitted at the controlled boundary across the full registered action
  panel for that nested trial.

A family must have exactly `nested_rows_per_family` rows. Every corpus must have at least two
families and both outcome classes. BRIGHT and MIRACL rows must encode evidence sufficiency as JSON
`null`; they cannot be counted as negative evidence outcomes.

The config pins two levels of provenance. `dependence_source` identifies the development artifact
from which scenario panels were made. Every `EffectScenario` separately pins the canonical SHA-256
of its exact panel. The runner rejects an unlisted panel, a missing panel, a digest mismatch, or a
partition mismatch.

Effect scenarios are data artifacts, not verbal modifiers applied inside the simulator. For
example, an expected scenario may use paired calibration replay as observed. A conservative
scenario may be produced before freeze by attenuating the full-model logits, increasing proposed
latency, and degrading fidelity under a declared rule, then storing the resulting raw rows as a
second pinned panel. This keeps assumptions inspectable. The simulator does not invent an effect
or silently clip an outcome.

## Exact gate set

The endpoint order is closed:

1. H2 equal-corpus log-loss reduction;
2. H2 equal-corpus Brier-score reduction;
3. H2 equal-corpus AUPRC gain;
4. H2 gain above all three thresholds in at least four of five corpora;
5. H3 equal-corpus family-relative latency reduction;
6. H3 retrieval-target noninferiority;
7. H3 complete-evidence noninferiority over the three evidence corpora;
8. H3 equal-corpus ratio of within-corpus p95 family-mean latency; and
9. H3 zero denied emissions.

For H2, log loss and Brier score are averaged within family, then within corpus, then equally across
corpora. AUPRC is not additive. The runner carries raw labels and probabilities through each family
draw, applies equal family weights, and recomputes tie-aware average precision inside each corpus.
A simulated study with one outcome class in any corpus receives an undefined AUPRC gate and fails
that endpoint.

For H3, latency is averaged across nested rows before the proposed-to-comparator ratio is formed.
The p95 is calculated over family-mean latency within each corpus. Retrieval attainment is averaged
over all five corpora. Evidence sufficiency is averaged only over SciFact, HotpotQA FullWiki, and
T2-RAGBench. The separately reported action-position sensitivity uses the observed positions bound
to those latency rows. It never reconstructs positions from family names, row order, or simulation
draw order.

The zero-emission condition is exact: the simulated study passes only when the total observed count
is zero. The report also gives the one-sided 95% Clopper-Pearson upper bound that would follow from
zero event families at that candidate count,

\[
1 - \alpha^{1/(5F)},
\]

where \(F\) is families per corpus. This reports what a zero count can exclude at finite sample
size; it does not convert absence of observed violations into proof of impossibility.

## Percentile-bootstrap plug-in calibration

For each scenario and candidate count, the simulator uses two disjoint PCG64 streams derived by
SHA-256 from:

```text
base seed | scenario ID | families per corpus | phase | corpus ID
```

The first stream estimates a centered percentile-bootstrap offset under the pinned
development-family distribution. The second estimates power. Neither corpus nor nested row is
sampled independently. For each corpus, both streams draw exactly \(F\) whole families with
replacement.

Let \(\hat\theta_s\) be a scenario estimand computed from every development family and let
\(\hat\theta_b^*\) be its value in calibration draw \(b\). Define the centered error
\(e_b=\hat\theta_b^*-\hat\theta_s\). For a lower-bound endpoint, the plug-in offset is

\[
c_L=q_{0.05}(e_b).
\]

An independently drawn simulated study with estimate \(\tilde\theta_j\) uses

\[
L_j = \tilde\theta_j + c_L.
\]

The p95 safety endpoint uses \(c_U=q_{0.95}(e_b)\) and
\(U_j=\tilde\theta_j+c_U\). This is a plug-in approximation to the registered directional
percentile family bootstrap, not a basic-bootstrap interval. The calibration and evaluation
streams are separate, so the same Monte Carlo draws do not estimate an offset and then count their
own gate decisions.

The approximation is auditable against the exact confirmatory calculation.
`audit_percentile_approximation` reconstructs a selected evaluation study from its SHA-256-derived
PCG64 stream, runs the registered 10,000-replicate inner bootstrap with base seed `20260713` and the
same endpoint seed offsets as `confirmatory_analysis.py`, and compares the seven continuous gate
decisions that govern H2/H3. The test suite exercises both passing and failing H3 latency
decisions. `decisions_agree` covers those seven primary decisions;
`sensitivity_decisions_agree` reports the action-position comparison separately and cannot affect
family-count selection. Exact inner bootstrapping remains tractable for a closed subset of studies,
but not for every cell of the 5,000-study, six-candidate, multi-scenario design grid.

The approximation may disagree with the exact gate for a study whose bound lies close to its
threshold, particularly for AUPRC or the p95 ratio. A primary-gate disagreement now aborts the
selection audit. It cannot be averaged away or excused by agreement elsewhere. The sealed analysis
itself always uses the registered 10,000-replicate family bootstrap. A design report estimates
operating behavior; it cannot replace observed inference.

At least 5,000 calibration studies and 5,000 power-evaluation studies are required outside explicit
test mode. Test-mode reports always set `freeze_ready=false`.

## Monte Carlo probability bounds and selection

For endpoint \(g\), let \(x_g\) of \(M\) independently generated studies pass. The report stores:

- \(x_g/M\);
- its Monte Carlo standard error; and
- the exact one-sided Clopper-Pearson lower bound for the pass probability.

For the registered selection grid, the familywise error rate is \(\alpha_F=0.05\), the number of
cells is \(G=6\times2=12\), and the Bonferroni cellwise error rate is

\[
\alpha_C=\frac{\alpha_F}{G}=\frac{0.05}{12}.
\]

Each required scenario-candidate cell therefore uses confidence
\(1-\alpha_C=0.995\overline{83}\). The union bound gives simultaneous coverage of at least 0.95
over all 12 cells without assuming that cells are independent. That assumption would be false or,
at minimum, unproved here because candidates and scenarios share development panels and
deterministic simulation machinery. Nonrequired scenarios retain pointwise 95% limits.

The joint event is the row-wise intersection of all nine gates. It receives the same probability
estimate and multiplicity-adjusted lower bound. A candidate count qualifies only when every
primary endpoint lower bound and the joint lower bound meet `target_power` in every required
scenario. The report then chooses the first qualifying count in the strictly increasing registered
candidate list. If none qualifies, the selected count is JSON `null` and the design remains
blocked. The action-position sensitivity retains a pointwise 95% limit and cannot affect this
choice.

This explicit endpoint check is redundant in population logic because joint success is a subset of
each gate. The multiplicity family therefore has 12 scenario-candidate cells, not 120
endpoint-by-cell claims. It is retained in the artifact contract so a corrupted report cannot omit
an endpoint while preserving a plausible joint field.

## Exact closed selection certificate

`freeze_ready=true` requires a canonical `selection-audit.json`. A report without this file remains
non-freezable even when its plug-in estimates select a family count.

For $M$ evaluation studies, target $p_0$, and Bonferroni cellwise error rate $\alpha_C$, define

\[
k=\min\{x:\operatorname{CP}_{L}(x,M;1-\alpha_C)\ge p_0\}.
\]

Observing $k$ exact joint passes proves that the lower probability bound reaches the target,
regardless of every unaudited study. Observing $M-k+1$ exact joint failures proves that at most
$k-1$ studies can pass, so the target is unreachable. This gives a sufficient early-stopping
certificate without treating a small convenience sample as the full simulation grid.

The audit processes candidate counts in registered ascending order and required scenarios in their
canonical order. For a provisionally qualifying cell, it checks approximate-pass study indices
first, in ascending order, until it has $k$ exact joint passes. For a provisionally failing cell,
it checks approximate-fail indices first until it has $M-k+1$ exact failures. Each checked study
uses the registered 10,000-replicate directional family bootstrap. The file binds the config,
panels, complete plug-in selection basis, family-draw digest for every checked study, exact and
approximate bounds, gate decisions, stopping thresholds, and selected count.

At the production values $M=5{,}000$, $p_0=0.90$, $G=12$, and $\alpha_C=0.05/12$,
$k=4{,}556$ and the blocking threshold is 445 failures. The lower limit is approximately
0.900073 at 4,556 passes and 0.899862 at 4,555 passes. With the two required scenarios:

- the first candidate qualifying in both scenarios requires 9,112 exact study audits, or
  91,120,000 registered inner-bootstrap replicates;
- a candidate blocked in its first scenario stops after 445 exact failures;
- a candidate that qualifies in the first scenario and fails the second consumes 5,001 audits;
  and
- the largest path ending in selection at the sixth candidate consumes 34,117 audits, or
  341,170,000 registered inner-bootstrap replicates.

These counts follow from the registered probability target. Reducing the exact bootstrap count,
testing fewer than the sufficient stopping set, or dropping a candidate or required scenario
changes the evidentiary contract.

A local Apple arm64 benchmark over the registered development geometry (75 calibration families
per corpus and three nested rows per family) measured about 0.358 seconds per exact study at
`F=25` and 3.118 seconds at `F=200`, with peak resident memory near 220 MiB after batching. Applying
the candidate-specific rates to the new stopping counts gives about 54.4 minutes for a
first-candidate certificate and 16.24 hours for the sixth-candidate worst path. The planning
envelopes are 55–65 minutes and 16–19 hours per exact execution. A successful fresh production
chain performs two such executions: generation in the post-embedding operator and one independent
replay in the freeze-package verifier. Standalone verification later adds one more replay.

Bonferroni is conservative, but its guarantee holds under arbitrary cell dependence and maps to a
fixed deterministic threshold before any result is seen. A Sidak correction would require an
independence argument that this design does not have. Holm or closed-testing procedures could be
less conservative, but they would require a different all-cell testing and stopping contract. They
are not substituted after observing the simulations.

The exact position and AUPRC calculations process 500 replicates at a time. Batching does not change
the PCG64 seed, corpus order, random draw order, per-replicate arithmetic, or percentile definition.
Conformance tests require bit-identical bounds against an unbatched execution. Memory therefore
scales with the batch and candidate family count, not with all 10,000 row cubes at once.

The position-adjusted latency comparison is carried in every audit record but remains a sensitivity
endpoint. Its approximate and exact decisions may differ without changing selection. The
disagreement remains visible, and fresh reproduction must recover it exactly.

## Canonical execution

The panel must be created and pinned before the config because the config contains its digest:

```python
from fractal_ann_diagnostics.joint_power_design import (
    DependenceSource,
    DevelopmentScenarioPanel,
    EffectScenario,
    GeometryGainThresholds,
    JointPowerDesignConfig,
    canonical_development_panel_bytes,
    canonical_joint_power_config_bytes,
    canonical_joint_power_report_bytes,
    canonical_joint_power_selection_audit_bytes,
    run_joint_power_design,
    run_joint_power_selection_audit,
)

expected_panel = DevelopmentScenarioPanel(
    scenario_id="expected-development-effect",
    partition="development-calibration",
    rows=expected_rows,
)
conservative_panel = DevelopmentScenarioPanel(
    scenario_id="conservative-attenuation",
    partition="development-calibration",
    rows=conservative_rows,
)

config = JointPowerDesignConfig(
    dependence_source=DependenceSource(
        artifact_uri="https://registry.example/design/development-source.json",
        artifact_sha256=development_source_sha256,
        partition="development-calibration",
        description="Paired development families created before sealed-label release.",
    ),
    effect_scenarios=(
        EffectScenario(
            scenario_id=expected_panel.scenario_id,
            panel_sha256=expected_panel.sha256,
            description="Empirical paired calibration replay.",
            selection_required=True,
        ),
        EffectScenario(
            scenario_id=conservative_panel.scenario_id,
            panel_sha256=conservative_panel.sha256,
            description="Pinned pre-freeze attenuation rule applied to raw paired rows.",
            selection_required=True,
        ),
    ),
    candidate_families_per_corpus=(25, 50, 75, 100, 150, 200),
    nested_rows_per_family=registered_nested_rows,
    geometry_gain_thresholds=GeometryGainThresholds(
        log_loss_reduction=registered_log_loss_gain,
        brier_score_reduction=registered_brier_gain,
        auprc_gain=registered_auprc_gain,
    ),
    simulation_seed=registered_seed,
)

panels = (expected_panel, conservative_panel)
selection_audit = run_joint_power_selection_audit(config, panels)
report = run_joint_power_design(config, panels, selection_audit=selection_audit)

panel_bytes = canonical_development_panel_bytes(expected_panel)
config_bytes = canonical_joint_power_config_bytes(config)
report_bytes = canonical_joint_power_report_bytes(report)
selection_audit_bytes = canonical_joint_power_selection_audit_bytes(selection_audit)
```

Writers should use exclusive creation and place the canonical files in the registered
`analysis/joint-power-design/` bundle. Each panel filename is its registered SHA-256. The report
records the config digest and every panel digest; the frozen manifest pins the exact bundle-tree
digest. Loading functions reject duplicate keys, nonfinite numbers, unknown fields, alternate
whitespace, a missing terminal newline, and any other byte representation that is not canonical.
The freeze-package compiler freshly reproduces the exact selection certificate, then reruns the
design and requires byte-identical audit and report output. A report with internally consistent but
invented pass counts is inadmissible.

## Freeze acceptance checks

A design report is admissible only when all of these conditions hold:

- the dependence source and every scenario panel exist at the pinned immutable locations;
- scenario construction was completed without sealed outcomes;
- config and panel bytes pass their canonical loaders and exact SHA-256 checks;
- all candidate counts were run with at least 5,000 calibration and 5,000 evaluation studies;
- the candidate grid is exactly 25, 50, 75, 100, 150, and 200, in that order;
- every required scenario is present and marked before simulation;
- `selection-audit.json` satisfies the closed exact stopping rule with 10,000 bootstrap replicates
  per checked study and contains no primary approximate/exact decision disagreement;
- `selected_families_per_corpus` is the smallest candidate that satisfies the closed rule;
- `freeze_ready=true`; and
- an independent rerun from the same bytes produces the same audit and report bytes.

The result remains conditional on the empirical development-family distribution and the declared
effect scenarios. Rare production policy structures, hardware faults absent from development, and
new drift mechanisms are outside that distribution. The conservative scenario should therefore be
substantive, not a nominal copy with a different name.

The clustered simulation approach follows Green and MacLeod's argument for simulation-based power
assessment when analytic approximations do not encode the fitted dependence structure
([2016](https://doi.org/10.1111/2041-210X.12504)). Exact binomial probability limits follow
Clopper and Pearson ([1934](https://doi.org/10.1093/biomet/26.4.404)).
