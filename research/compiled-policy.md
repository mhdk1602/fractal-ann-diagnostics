# Compiled authorization masks

The confirmatory runner cannot send a five-million-row document universe to OPA on every
decision. The compiled-mask contract keeps OPA authoritative without turning policy evaluation
into the dominant workload.

OPA receives the subject, action, finite environment, document count, ordered-universe digest,
catalog digest, pinned policy revision, and fresh request bindings. It returns one mask identifier,
mask digest, and authorized count with exact echoes of those bindings. Unknown fields or mismatched
bytes deny the request.

The selected mask is stored locally as `numpy.packbits(..., bitorder="little")`. Its catalog binds:

- the exact document count and ordered document-universe SHA-256;
- the policy-bundle revision;
- each unique mask ID, relative path, byte count, SHA-256, and authorized count; and
- the encoding literal `numpy-packbits-little-v1`.

The local adapter opens the mask through a no-follow, single-link regular-file check. It verifies
the byte count and digest, rejects nonzero padding bits, unpacks exactly `document_count` entries,
checks the authorized count, and exposes an immutable Boolean array to `GovernedRetriever`. A
second authorization decision is still required immediately before result emission.

This design compiles a fixed policy intervention. It does not estimate how enterprise permissions
are distributed. The study must state the chosen mask strata, subject schedule, mutation schedule,
and policy seed before freeze. The catalog, mask files, OPA bundle, and schedule are separate pinned
artifacts so no component can silently redefine another.

The production OPA rule should return this closed result object:

```json
{
  "decision_id": "opa-generated-id",
  "result": {
    "action": "retrieve",
    "authorized_count": 4000000,
    "catalog_request_sha256": "<sha256>",
    "document_count": 5233329,
    "document_universe_sha256": "<sha256>",
    "environment_sha256": "<sha256>",
    "mask_catalog_sha256": "<sha256>",
    "mask_id": "reader-us-medium",
    "mask_sha256": "<sha256>",
    "policy_revision": "<pinned-bundle-revision>",
    "request_nonce": "<fresh-nonce>",
    "request_sha256": "<sha256>",
    "subject": "reader-us"
  }
}
```

The request contains no `document_ids` array. The response contains no `allowed_document_ids`
array. This keeps network payload size independent of corpus cardinality.

## Sealed local service

Production does not accept an OPA URL or rely on a daemon started by the host. C0 contains the
OPA 1.18.2 static binary at `/usr/local/bin/opa` and the compiled module at
`/opt/app/policy/opa_compiled_masks.rego`. The module SHA-256 is
`18f6eb8a7411a7a1415bd2425ad5720f28fcd3b428d9aa2c1e7d73f6e14e356c`. Both files are root-owned
and non-writable to UID 65532. The runtime-attestation plan independently pins the binary bytes and
must name that exact path.

`run_sealed_online_once` writes the corpus attempt marker before it opens `opa-data.json` or the
Rego module. It then reloads the policy receipt and canonical OPA data, checks every assignment
against the three runtime policy groups and the compiled-mask catalog, and writes those exact bytes
to a mode-0600 file in a new mode-0700 `/tmp` directory. OPA receives two positional sources:

```text
/opt/app/policy/opa_compiled_masks.rego
fractal:/tmp/fractal-confirmatory-opa-<random>/opa-data.json
```

OPA 1.18.2 interprets the prefix as a data mount at `data.fractal`. The service binds only
`127.0.0.1:8181`. Readiness requires `GET /health?plugins` plus a valid `POST` to the exact
decision path; the latter proves that the Rego package, data prefix, assignment lookup, closed
result fields, and top-level `decision_id` are all active before timed work begins.

The Python runner owns the child for the entire attempt. It drains stderr without allowing the
retained diagnostic buffer to exceed 64 KiB. Result persistence occurs after graceful termination
or forced termination, stderr closure, and scratch deletion. Any failure in that sequence denies
the attempt and leaves only the one-shot marker.
