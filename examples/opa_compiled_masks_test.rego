package fractal_auth.retrieval

import rego.v1

request_input := {
    "action": "retrieve",
    "catalog_request_sha256": "catalog-request-sha",
    "document_count": 5233329,
    "document_universe_sha256": "universe-sha",
    "environment": {"policy_state": "medium"},
    "environment_sha256": "environment-sha",
    "mask_catalog_sha256": "catalog-sha",
    "policy_revision": "bundle-7f21",
    "request_nonce": "nonce-123",
    "request_sha256": "request-sha",
    "subject": "reader-us",
}

policy_data := {
    "assignments": {
        "reader-us": {
            "medium": {
                "authorized_count": 2616664,
                "mask_id": "reader-us-medium",
                "mask_sha256": "mask-sha",
            },
        },
    },
    "document_count": 5233329,
    "document_universe_sha256": "universe-sha",
    "mask_catalog_sha256": "catalog-sha",
    "policy_revision": "bundle-7f21",
}

test_mask_decision_selects_one_bound_mask if {
    result := mask_decision with input as request_input with data.fractal as policy_data

    result.mask_id == "reader-us-medium"
    result.mask_sha256 == "mask-sha"
    result.authorized_count == 2616664
    result.catalog_request_sha256 == request_input.catalog_request_sha256
    result.request_nonce == request_input.request_nonce
    result.request_sha256 == request_input.request_sha256
}

test_unknown_subject_has_no_decision if {
    unknown := object.union(request_input, {"subject": "unknown"})
    not mask_decision with input as unknown with data.fractal as policy_data
}

test_wrong_catalog_has_no_decision if {
    wrong := object.union(request_input, {"mask_catalog_sha256": "other"})
    not mask_decision with input as wrong with data.fractal as policy_data
}

test_wrong_action_has_no_decision if {
    wrong := object.union(request_input, {"action": "delete"})
    not mask_decision with input as wrong with data.fractal as policy_data
}
