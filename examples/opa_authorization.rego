package fractal.retrieval

import rego.v1

# Query this rule through POST /v1/data/fractal/retrieval/decision.
# The Python adapter requires OPA decision logging so the HTTP response carries
# a top-level decision_id in addition to this rule's result.

decision := {
    "allowed_document_ids": allowed_document_ids,
    "action": input.action,
    "document_universe_sha256": input.document_universe_sha256,
    "environment_sha256": input.environment_sha256,
    "policy_revision": data.fractal.policy_revision,
    "request_nonce": input.request_nonce,
    "request_sha256": input.request_sha256,
    "subject": input.subject,
}

allowed_document_ids := [document_id |
    some document_id in input.document_ids
    document := data.fractal.documents[document_id]
    input.action == "retrieve"
    input.environment.tenant == document.tenant
    may_read(document)
]

may_read(document) if "*" in document.readers

may_read(document) if input.subject in document.readers
