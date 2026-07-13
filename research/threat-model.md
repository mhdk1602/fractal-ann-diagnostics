# Retrieval authorization threat model

## Protected claim

The system may place a document in a learned reranker, model context, trace, or output only when the
live policy decision for the requesting subject, document, action, and environment is `allow`.

The benchmark treats authorization as a deterministic security decision. A model score, embedding
distance, controller action, cached ACL, or generated answer cannot override it.

## Trust boundaries

```text
identity + query + environment
             |
             v
  live policy decision point  ---- unavailable/stale ----> abstain
             |
       authorized IDs only
             |
             v
 authorized vector universe
             |
   geometry + action controller
             |
      retrieval / reranking
             |
 deterministic evidence gate
             |
      generator or abstain
```

The reference code takes the strict path: it builds HNSW only over the role-authorized vectors.
This is intentionally less storage-efficient than filter-native production indexes. It gives the
tests a clear noninterference boundary.

The `unsafe-unfiltered` implementation exists only as a comparator. It is not exposed through
`GovernedRetriever`.

## Adversaries and failures

- A user intentionally asks for a denied document or infers its membership.
- A benign query is geometrically closest to denied material.
- A cached ACL lags a revocation or embargo.
- The policy engine is unavailable.
- An embedding-model migration makes the active index stale or internally mixed.
- An approximate index omits enough authorized evidence to invite an unsupported answer.
- A prompt-injection attack asks the model to reproduce retrieved material.
- Logs or traces capture denied candidate identifiers or content.

The extraction threat is empirically serious. Qi et al. recover private RAG material through prompt
injection, including high success against customized GPTs
([ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/79cafa874121a3435d8a54f454b646b4-Paper-Conference.pdf)).
[GeoEx](https://openreview.net/forum?id=x61zDYEZ91) uses embedding geometry for corpus
reconstruction, while
[LeakDojo](https://doi.org/10.18653/v1/2026.findings-acl.287) shows that stronger instruction
following and answer faithfulness can increase leakage.

## Invariants

1. Policy evaluation occurs before query-specific geometry or retrieval.
2. Unknown roles, empty grants, stale expected versions, and policy-engine failure deny by default.
3. An authorized index maps local IDs back to globally auditable document IDs.
4. Every returned ID is rechecked against the live mask before a result object is emitted.
5. `unauthorized_context` must equal zero for every governed action.
6. Policy version, action, fallback reason, and evidence IDs enter the audit record.
7. Evidence insufficiency causes wider search, exact search, or abstention. It never relaxes IAM.

## Claims outside scope

- The current pilot does not prove that source-system IAM is correct.
- Zero violations in a finite benchmark is not a universal safety proof.
- ANN recall does not establish factuality, citation entailment, or answer completeness.
- The rule controller is not a calibrated production policy.
- Synthetic ACL prevalence does not estimate enterprise prevalence.

The end-to-end stage will use retrieval and generation measures from
[RAGAS](https://doi.org/10.18653/v1/2024.eacl-demo.16),
[ARES](https://doi.org/10.18653/v1/2024.naacl-long.20), and
[RAGChecker](https://proceedings.neurips.cc/paper_files/paper/2024/hash/27245589131d17368cccdfa990cbf16e-Abstract-Datasets_and_Benchmarks_Track.html).
