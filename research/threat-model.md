# Retrieval authorization threat model

## Protected claim

A document identifier may leave the controlled retrieval boundary only when the authoritative
policy decision permits that subject, action, environment, and exact document universe. Policy is
the authority. Embedding distance, controller score, cached index membership, and generated text
cannot enlarge a grant.

The planned v0.3 primary experiment ends at retrieval and complete-evidence assessment. It does
not claim answer correctness, entailment, faithfulness, or resistance to prompt-based extraction.

## Trust boundaries

```text
identity + query + finite policy environment
                    |
                    v
       authoritative PDP over corpus digest
          | unavailable / malformed / stale
          +-------------------------------> abstain
                    |
             authorized mask
                    |
                    v
       authorized-only exact + HNSW indexes
                    |
         bounded probe -> frozen controller
                    |
                    v
             selected retrieval action
                    |
           fresh request-bound PDP check
                    |
                    v
             emitted document IDs
                    |
 prediction + raw action panel, no labels
                    |
 panel-admission receipt + completion anchor
                    |
 exact sealed-label-byte admission and join
                    |
    exclusive pre-compute attempt receipt
                    |
 detached result receipt + canonical result
```

The controlled exact and HNSW paths copy only authorized vector rows after policy evaluation.
Original document IDs remain available for audit. The development-only global comparator is not
available through `GovernedRetriever`.

The built-in OPA adapter requires HTTPS plus a bearer credential for remote endpoints. A supplied
TLS context must verify certificates and hostnames; the context alone is not treated as proof of a
client identity. Plain HTTP is accepted only for a literal loopback IP, and redirects are rejected.
An injected transport is trusted integration code, not a network-authentication bypass. Each
response must echo a fresh nonce, the pinned policy-bundle revision, and a request digest that binds
subject, action, environment digest, and ordered document-universe digest. Duplicate-key,
nonfinite, malformed, replayed, oversized, unavailable, stale-bundle, or misbound responses deny all
documents.

## Adversaries and failures

- A user asks for a denied document or tests whether it exists.
- A benign query lies closest to denied material.
- A policy revision revokes access while retrieval is running.
- The PDP is unavailable, returns stale state, or replays an earlier decision.
- An embedding migration mixes document and query revisions.
- Approximate search omits a complete authorized evidence route.
- Sealed relevance labels leak into model fitting, action choice, or online prediction.
- Counterfactual action rows are altered or invented after relevance labels become visible.
- A runner swaps a corpus, model, policy bundle, verified-artifact receipt, prediction-completion
  receipt, or result after protocol freeze.
- A scorer computes repeatedly after label release or suppresses a failed primary gate before it
  reaches the controlled result directory.
- Logs retain raw subjects, document content, distances, or generated output.
- A prompt-based attack tries to extract material after retrieval.

Retrieval exposure can become model exposure. Qi et al. recover private RAG material through prompt
injection ([ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/79cafa874121a3435d8a54f454b646b4-Paper-Conference.pdf)).
[GeoEx](https://openreview.net/forum?id=x61zDYEZ91) uses embedding geometry for corpus
reconstruction. [LeakDojo](https://doi.org/10.18653/v1/2026.findings-acl.287) finds that leakage
varies across attacks, models, and datasets. These sources motivate the boundary; the v0.3
experiment does not test those attacks.

## Enforced invariants

1. Policy evaluation precedes query geometry, exact search, and HNSW construction.
2. Unknown subjects, empty grants, invalid environments, corpus mismatch, PDP failure, and replay
   deny by default.
3. Query geometry and learned indexes inspect authorized vectors only. Geometry is limited to LID at
   k=50, LID-CV, relative contrast, and radius expansion; probe latency and work are system
   telemetry.
4. A fresh policy decision is required immediately before result emission. Initial and final
   policy version, mask, request nonce, and request digest must agree without being identical
   decisions.
5. Every emitted ID is permitted by the final mask. `unauthorized_context` is zero on every
   governed action.
6. Audit records bind the trial, policy requests, corpus and component revisions, work counters,
   evidence hashes, and prior record hash without retaining raw content.
7. The online runner receives opaque query and family keys. Relevance, evidence bundles, source
   answers, and label-bearing metadata stay in a separately hashed custodian artifact until an
   external anchor records the exact prediction and typed pre-label action-panel binding. For each
   corpus, the outer manifest separately pins source inputs, canonical online-execution bytes, and
   canonical label bytes; the label artifact binds the online-execution digest.
8. Completed and governed-abstention panel rows are derived from a typed `GovernedResult` matched
   to a self-hashed `AuditRecord`. Returned IDs, latency, entitlement count, action, policy revision,
   authorization records, and search work cannot be replaced by caller scalar fields. Audit records
   cannot be reused; governed counterfactuals share the selected policy revision and one final
   authorization universe.
9. Every panel has a detached admission receipt bound to its exact bytes, manifest, run, execution
   artifact, corpus, primary partition, frozen query-partition audit, trusted audit-chain
   head, and every trial-action cell. Its records bind controller and policy decisions, policy
   request and mask digests, audit position for governed rows, and runner timing for failed rows.
10. Failed rows use one of five registered codes, monotonic start and finish times, and the pinned
    runner identity. Their latency and timing digest are derived from those fields. They have no
    audit claim, returned IDs, or entitlement input. Every `hnsw-low` row must carry its supplied
    pre-outcome feature tuple; admission verifies placement, not how it was computed. Every
    trial-action cell has exactly one admitted outcome, selected failures remain in the panel,
    `abstain` must be governed, and `exact-authorized` must be completed.
11. Post-label ANN recall is derived against the completed `exact-authorized` row in the anchored
    panel. Evidence sufficiency is derived from anchored returned IDs and separately held gold
    bundles. The actual sealed-label custody file enters through the canonical no-follow loader, its
    digest must match the manifest, and each joined label object must equal that admitted payload.
    Raw action, latency, state, feature, and entitlement fields must equal the pre-label panel.
    Low-effort action failure is intent-to-treat: a non-completed `hnsw-low` action or a completed
    action with authorized recall below 0.90. A completed empty result against an empty exact
    reference is a valid governed no-result service outcome. No authorized-universe-size exclusion
    is allowed.
12. Sealed execution accepts a closed externally registered manifest, exact artifact roles, a
    manifest lock, canonical exact artifact-verification receipt, pinned runner identity,
    digest-pinned image, byte-exact local registry record, matching registration receipt, and one
    digest-derived run-receipt path. Production admission rereads every artifact through the
    manifest-bound local map and requires the fresh receipt bytes to equal the admitted verification
    receipt. Immediately before the run-receipt write, a bounded certificate-validated HTTPS fetch
    must return the same registry bytes without a redirect. These checks precede the
    network-disabled sealed execution boundary.
13. The built-in sealed scorer accepts a canonical local `file:` result directory and derives one
    attempt, detached result-receipt, and result path from the manifest digest. It creates the
    attempt receipt with `O_EXCL` before the H1 orientation diagnostic or H2–H3 confirmatory-gate
    computation. The receipt binds the manifest,
    run, joined input, model suite, runner, and intended result. A prior or failed attempt remains
    admitted and cannot be replaced. The detached result receipt binds the attempt digest, result
    digest, manifest, and result URI and is created before the result file.

## Time and revocation semantics

Authorization is asserted at result emission, not for an unlimited future interval. A caller that
caches IDs or content must reauthorize at its own later disclosure boundary. The reference API
cannot prevent a downstream process from retaining a result after revocation.

Likewise, the hash chain and exclusive receipts detect changes only relative to a trusted external
anchor. They do not create write-once storage. The built-in result writer supports a local `file:`
directory only. Its `O_EXCL`, no-follow, owner, permission, and `fsync` checks prevent replacement
through that admitted path, but an administrator with broader filesystem authority can still copy,
hide, or rewrite files outside the package. The protocol therefore requires controlled result-store
retention and custody separate from the online runner. `s3:` and `gs:` stores need a separately
pinned authenticated create-if-absent adapter; the built-in scorer rejects them.

Typed admission and anchoring prove internal agreement and pre-label byte identity. Ordinary Python
objects do not attest which external process, machine, or credential produced them. The exclusive
attempt receipt blocks a second admitted package run; it cannot stop arbitrary Python code,
process-memory inspection, logging, or copying. Runner identity, scorer and online-runner image
provenance, operating system, result-directory custody, audit-head custody, and the independently
administered completion anchor remain part of the trusted computing base.

A technical failure ends confirmatory v0.3. The receipt schema's `reserve` label may support
engineering rehearsals, but it cannot rescue or replace the primary confirmatory attempt. Any later
attempt needs an amended protocol version, a new frozen manifest, and a new external registration.

## Claims outside scope

- The study does not prove that source-system IAM is correct.
- A zero-event upper bound is finite-study evidence, not a universal safety proof.
- Paired-world tests are observational checks, not formal noninterference proofs.
- ANN recall and complete-evidence retrieval do not establish answer correctness.
- Public-corpus ACL generators do not estimate enterprise entitlement prevalence.
- OPA, TLS, the operating system, runner identity, registry and anchor administrators, and
  immutable-store administration remain in the trusted computing base.

Answer evaluation can be registered as a later study using human-validated measures from
[RAGAS](https://doi.org/10.18653/v1/2024.eacl-demo.16),
[ARES](https://doi.org/10.18653/v1/2024.naacl-long.20), and
[RAGChecker](https://proceedings.neurips.cc/paper_files/paper/2024/hash/27245589131d17368cccdfa990cbf16e-Abstract-Datasets_and_Benchmarks_Track.html).
