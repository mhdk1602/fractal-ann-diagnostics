package fractal.retrieval

import rego.v1

request_input := {
    "action": "retrieve",
    "document_ids": [0, 1, 2],
    "document_universe_sha256": "universe-sha",
    "environment": {"tenant": "research"},
    "environment_sha256": "environment-sha",
    "request_nonce": "nonce-123",
    "request_sha256": "request-sha",
    "subject": "reader",
}

policy_documents := [
    {"readers": ["reader"], "tenant": "research"},
    {"readers": ["other"], "tenant": "research"},
    {"readers": ["*"], "tenant": "research"},
]

test_decision_filters_documents_and_echoes_request_binding if {
    result := decision with input as request_input
        with data.fractal.documents as policy_documents
        with data.fractal.policy_revision as "bundle-7f21"

    result.allowed_document_ids == [0, 2]
    result.policy_revision == "bundle-7f21"
    result.subject == request_input.subject
    result.action == request_input.action
    result.environment_sha256 == request_input.environment_sha256
    result.document_universe_sha256 == request_input.document_universe_sha256
    result.request_nonce == request_input.request_nonce
    result.request_sha256 == request_input.request_sha256
}

test_wrong_tenant_denies_all_documents if {
    different_tenant := object.union(
        request_input,
        {"environment": {"tenant": "another-tenant"}},
    )
    result := decision with input as different_tenant
        with data.fractal.documents as policy_documents
        with data.fractal.policy_revision as "bundle-7f21"

    result.allowed_document_ids == []
}

test_wrong_action_denies_all_documents if {
    wrong_action := object.union(request_input, {"action": "delete"})
    result := decision with input as wrong_action
        with data.fractal.documents as policy_documents
        with data.fractal.policy_revision as "bundle-7f21"

    result.allowed_document_ids == []
}
