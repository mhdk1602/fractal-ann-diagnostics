# Paper outline

**Registered title:** Adaptive Policy-Aware Vector Retrieval Under Corpus, Embedding, and
Authorization Drift

**Conditional title:** Fractal Risk Control for Policy-Aware RAG

The conditional title is permitted only if the preregistered geometric and controller gates pass.

## Claim hierarchy

1. Authorization is a deterministic boundary. The controller cannot grant access.
2. Exact top-k over the live authorized corpus is the retrieval reference.
3. Query-local geometry is a candidate failure signal, not a presumed mechanism.
4. Adaptive value exists only if a controller beats a cost-matched static action.
5. Answer-level claims require annotated evidence and validated generation evaluation.

## Sections

### 1. Problem

Enterprise RAG joins approximate vector search to live, heterogeneous IAM. A globally accurate
index can still fail a specific authorized query because the permitted evidence occupies a sparse
or geometrically difficult subset. Missing authorized evidence can produce unsupported answers;
retrieve-then-filter can expose denied material.

### 2. Prior work and novelty boundary

Position the paper after Authorization-First Retrieval, Permission-Aware RAG, Filtered-DiskANN,
ACORN, Global-Local Selectivity, Ada-ef, Fiber-Navigable Search, and RAG extraction attacks. State
what each already resolves. Do not claim the first permission-aware or geometry-aware filtered
retrieval system.

### 3. Formal contract

Define the live authorized universe, exact authorized neighbor truth, evidence sufficiency,
structural entitlement violation, false permit, false denial, compute budget, and controller regret.

### 4. Reference architecture

Describe the policy plane, authorized index, query geometry, action controller, deterministic
evidence gate, audit record, and fail-closed states. Explain why geometry cannot decide permission.

### 5. Candidate geometric mechanism

Define LID, cross-scale instability, relative contrast, neighbor-radius expansion, hub exposure,
and offline vector-policy correlation. Document metric handling and permutation tests. Include the
rejected MFDFA feature as an integrity case study.

### 6. Benchmark

Describe corpora, evidence annotations, RBAC/ABAC generators, live policy revisions, exact truth,
ANN backends, action grid, embedding revisions, drift sequences, hardware, warmups, and hashes.

### 7. Preregistered analysis

Report H1–H4 in order. Show corpus-specific estimates before pooled estimates. Keep authorization,
retrieval fidelity, evidence sufficiency, latency, and cost in separate panels.

### 8. Controller results

Compare the adaptive policy with cost-matched low-effort, high-effort, exact, and abstention
baselines. Report counterfactual action regret, calibration, fallback rate, and the security
invariant.

### 9. Drift and external transfer

Show corpus, embedding, and policy drift separately. Failed transfer remains visible and limits the
claim.

### 10. Limitations

Cover synthetic policies, effective corpus-level sample size, backend specificity, timing
sensitivity, authorization-oracle assumptions, evaluator error, and the limits of zero observed
violations.

## Planned figures

1. Authorization-first two-plane architecture.
2. Query geometry under aligned and fragmented policy subsets.
3. Failure probability by LID and allow-rate strata.
4. Action frontier: authorized recall, p95 latency, and exact-fallback rate.
5. Controller regret against the per-query oracle.
6. Corpus, embedding, and policy drift trajectories.
7. Fractal-feature ablation and calibration.
8. Security and evidence-policy outcome matrix.

## Release rule

The benchmark, protocol, action matrix, drift traces, and negative results are released even when
the geometric or controller hypotheses fail. The title and abstract change according to the gates;
the data do not change according to the preferred story.
