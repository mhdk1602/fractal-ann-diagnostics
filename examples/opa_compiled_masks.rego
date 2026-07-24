package fractal_auth.retrieval

import rego.v1

# Query this rule through POST /v1/data/fractal_auth/retrieval/mask_decision.
# Decision logging must be enabled so OPA adds the top-level decision_id.
mask_decision := {
    "action": input.action,
    "authorized_count": assignment.authorized_count,
    "catalog_request_sha256": input.catalog_request_sha256,
    "document_count": input.document_count,
    "document_universe_sha256": input.document_universe_sha256,
    "environment_sha256": input.environment_sha256,
    "mask_catalog_sha256": input.mask_catalog_sha256,
    "mask_id": assignment.mask_id,
    "mask_sha256": assignment.mask_sha256,
    "policy_revision": data.fractal.policy_revision,
    "request_nonce": input.request_nonce,
    "request_sha256": input.request_sha256,
    "subject": input.subject,
} if {
    input.action == "retrieve"
    input.document_count == data.fractal.document_count
    input.document_universe_sha256 == data.fractal.document_universe_sha256
    input.mask_catalog_sha256 == data.fractal.mask_catalog_sha256
    input.policy_revision == data.fractal.policy_revision
    assignment := data.fractal.assignments[input.subject][input.environment.policy_state]
}
