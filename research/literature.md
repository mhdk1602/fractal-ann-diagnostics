# Evidence ledger

This ledger records the claim each source is allowed to support. A citation is not treated as a
generic endorsement of the project.

Publication status for the 2026 sources was checked on 2026-07-13. Preprints are identified as
such rather than presented as peer-reviewed results.

## Authorization and governance

| Source | Evidence used here | Boundary |
|---|---|---|
| [Namboothiri, TrustNLP 2026](https://aclanthology.org/2026.trustnlp-main.15/) | Authorization must constrain retrieval before a learned component; retrieve-then-filter can expose denied context. | Small controlled corpus; no production-scale ANN or multiscale geometry. |
| [Permission-Aware RAG, IEEE Access 2025](https://doi.org/10.1109/ACCESS.2025.3628960) | Provider-native IAM validation can avoid flattening heterogeneous policy systems. | Does not study query-local ANN hardness. |
| [NIST SP 800-162](https://doi.org/10.6028/NIST.SP.800-162) | ABAC decisions depend on subject, object, operation, and environment attributes. | Standard, not empirical retrieval evidence. |
| [NIST SP 800-207](https://doi.org/10.6028/NIST.SP.800-207) | Resource access follows explicit authentication and authorization decisions. | Does not prescribe vector-index internals. |
| [NIST AI 600-1](https://doi.org/10.6028/NIST.AI.600-1) | Test retrieval changes, record provenance, monitor deployed behavior, and define intervention criteria. | Risk-management guidance, not a controller-effect estimate. |
| [OWASP LLM08:2025](https://genai.owasp.org/llmrisk/llm082025-vector-and-embedding-weaknesses/) | Multi-tenant leakage, embedding inversion, poisoning, permission-aware storage, and immutable retrieval logs are explicit vector-system risks. | Practitioner security guidance. |

## Geometry and ANN difficulty

| Source | Evidence used here | Boundary |
|---|---|---|
| [Aumüller and Ceccarello, 2021](https://doi.org/10.1016/j.is.2021.101807) | LID, relative contrast, and query expansion distinguish ANN query difficulty. | Does not condition on authorization. |
| [Radovanović et al., 2010](https://jmlr.org/papers/v11/radovanovic10a.html) | Hubness is skew in reverse nearest-neighbor occurrence. | Dataset geometry result, not a governance result. |
| [He, Kumar, and Chang, 2012](https://arxiv.org/abs/1206.6411) | Relative contrast relates data distribution to nearest-neighbor search difficulty. | Predates modern filtered graph indexes. |
| [Elliott and Clark, 2024](https://arxiv.org/abs/2405.17813) | HNSW recall varies with intrinsic dimension and insertion order. | Preprint; not authorization-aware. |
| [Dang, 2026](https://arxiv.org/abs/2604.00102) | Local geometry separates filtered graph-search failure regimes. | Preprint and search-algorithm contribution; it narrows this project's novelty. |

## Filtered and adaptive vector search

| Source | Evidence used here | Boundary |
|---|---|---|
| [Filtered-DiskANN, WWW 2023](https://doi.org/10.1145/3543507.3583552) | Label-aware graphs can provide high-recall filtered retrieval. | Labels are predicates, not live IAM decisions. |
| [ACORN, PACMMOD 2024](https://doi.org/10.1145/3654923) | Predicate-induced HNSW traversal supports arbitrary filters. | Does not test evidence sufficiency or policy drift. |
| [Learned adaptive early termination, SIGMOD 2020](https://doi.org/10.1145/3318464.3380600) | Query-specific early termination can reduce ANN latency at matched accuracy. | No authorization or evidence outcome. |
| [Ada-ef, PACMMOD 2026](https://doi.org/10.1145/3786639) | Per-query `efSearch` can target declarative recall. | Recall objective is not an entitlement invariant. |
| [RACORN-1, 2026](https://arxiv.org/abs/2607.00768) | Adaptive graph and exact fallbacks recover recall under low selectivity and adverse vector-filter correlation. | Preprint; adapts filtered-search execution but does not test live authorization, evidence sufficiency, or a policy-governed action controller. |
| [MoReVec and Global-Local Selectivity, 2026](https://arxiv.org/abs/2602.11443) | Engine execution and vector-filter correlation materially alter filtered recall and latency. | Preprint; no authorization-first controller. |
| [Lim et al., 2026](https://arxiv.org/abs/2606.14193) | Allow rate and vector-attribute correlation can be unstable hardness proxies. | PVLDB 2026 accepted manuscript; supports measured outcome labels rather than proxy-only claims. |
| [FANNBench, 2025](https://github.com/lmccccc/FANNBench) | A shared harness covers ACORN, DiskANN, DSG, FAISS, SeRF, UNIFY, Milvus, and other filtered indexes. | Engineering benchmark; not an evidence-policy study. |

## Extraction and privacy

| Source | Evidence used here | Boundary |
|---|---|---|
| [Qi et al., ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/79cafa874121a3435d8a54f454b646b4-Paper-Conference.pdf) | Prompt injection can extract private RAG data through the model interface. | Attacks generation after retrieval; does not measure filtered ANN. |
| [RAG membership inference, EMNLP 2025](https://doi.org/10.18653/v1/2025.findings-emnlp.438) | Membership probes can have unusually high similarity to a target document. | Attack-specific evidence. |
| [GeoEx, ICML 2026](https://openreview.net/forum?id=x61zDYEZ91) | Proxy-embedding geometry can aid corpus reconstruction. | Geometry is an offensive signal as well as a reliability signal. |
| [LeakDojo, ACL 2026](https://doi.org/10.18653/v1/2026.findings-acl.287) | Leakage varies across attacks, models, and datasets; instruction following can raise risk. | Does not supply an authorization-first ANN controller. |

## Evaluation

| Source | Evidence used here | Boundary |
|---|---|---|
| [RAGAS, EACL 2024](https://doi.org/10.18653/v1/2024.eacl-demo.16) | Context relevance, faithfulness, and answer relevance can be assessed without full reference answers. | Automated evaluators require validation. |
| [ARES, NAACL 2024](https://doi.org/10.18653/v1/2024.naacl-long.20) | A small human-labeled set can anchor evaluator inference with uncertainty. | Evaluation method, not retrieval security evidence. |
| [RAGChecker, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/27245589131d17368cccdfa990cbf16e-Abstract-Datasets_and_Benchmarks_Track.html) | Retrieval- and generation-level diagnostics can be separated. | Does not define IAM correctness. |
| [TREC 2025 RAG Track](https://trec.nist.gov/pubs/trec34/index.html) | Relevance, completeness, attribution verification, and agreement are distinct measures. | Shared-task setting, not enterprise authorization. |

## Protocol, inference, and reproducibility

| Source | Evidence used here | Boundary |
|---|---|---|
| [OSF Registrations guide](https://help.osf.io/article/330-welcome-to-registrations) | Preregistration deposits a time-stamped, read-only study plan before data collection or analysis; a submitted registration cannot be edited. | Platform documentation, not proof that this repository has been registered. |
| [Nosek et al., 2018](https://doi.org/10.1073/pnas.1708274114) | Separating hypothesis generation from tests on unseen observations reduces hindsight bias in confirmatory work. | Methodological argument, not validation of this protocol. |
| [Pineau et al., 2021](https://www.jmlr.org/papers/v22/20-303.html) | Artifact, code, model, and result disclosure improve inspectability of machine-learning reports. | Checklist evidence; it does not make an experiment confirmatory. |
| [Berger and Hsu, 1996](https://doi.org/10.1214/ss/1032280304) | Intersection-union tests require every component null to be rejected for the joint claim. | General statistical result; endpoint-specific validity still depends on the frozen design. |
| [Green and MacLeod, 2016](https://doi.org/10.1111/2041-210X.12504) | Simulation can estimate power for clustered mixed-model designs. | Does not validate this repository's beta-binomial simulator or nuisance assumptions. |

## Research position

The cited work supplies precedents for the principal components. The untested conjunction is
whether multiscale geometry inside a live authorized universe adds held-out predictive value,
whether that prediction changes the cheapest valid action, and whether the controller retains
value through corpus, embedding, and policy drift.

If the geometry ablation fails, the repository becomes a policy-aware retrieval benchmark. That is
an intended result, not a fallback hidden after analysis.
