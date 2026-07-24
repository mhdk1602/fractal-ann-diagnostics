# GitHub environment-control receipt

The environment verifier is deliberately offline. It does not authenticate, call GitHub, run
`gh`, or alter repository settings. An operator captures five REST response bodies, transfers them
to the controlled C0 workspace, and runs the verifier against those files.

## Admitted control state

The repository must be exactly `mhdk1602/fractal-ann-diagnostics` and its environment listing must
contain exactly these two names:

| Environment | Protection rule | Deployment policy |
|---|---|---|
| `confirmatory-rehearsal` | none | branch `c0-candidate/*` |
| `confirmatory` | one required reviewer: user `mhdk1602`, ID `9646005`; `prevent_self_review=false` | tags `confirmatory-apparatus-c0` and `confirmatory-freeze-c1` |

Both environments must set `protected_branches=false` and `custom_branch_policies=true`. Any extra
environment, rule, reviewer, branch, or tag fails admission. The confirmatory rule is recorded
sole-operator self-approval. It is not independent review or independent custody.

The GitHub environments REST response does not provide admissible evidence of the repository's
administrator-bypass configuration. The receipt therefore fixes
`admin_bypass_rest_attestation` to
`not-attestable-via-github-rest-environments-api`. A UI observation or an operator assertion must
not be promoted into this REST-derived receipt.

## Required response bodies

Retain the JSON bodies returned by these exact API resources:

| CLI input | REST resource |
|---|---|
| `--environments-list` | `/repos/mhdk1602/fractal-ann-diagnostics/environments?per_page=100` |
| `--confirmatory-environment` | `/repos/mhdk1602/fractal-ann-diagnostics/environments/confirmatory` |
| `--confirmatory-deployment-policies` | `/repos/mhdk1602/fractal-ann-diagnostics/environments/confirmatory/deployment-branch-policies?per_page=100` |
| `--rehearsal-environment` | `/repos/mhdk1602/fractal-ann-diagnostics/environments/confirmatory-rehearsal` |
| `--rehearsal-deployment-policies` | `/repos/mhdk1602/fractal-ann-diagnostics/environments/confirmatory-rehearsal/deployment-branch-policies?per_page=100` |

The verifier rejects symbolic links, multiply linked files, duplicate JSON keys, non-finite
numbers, incomplete pagination, cross-repository URLs, and every policy mutation. It hashes the
canonical JSON form of each admitted response and stores all five digests in a closed receipt.

## Verify and read back

```bash
fractal-github-environment-control verify \
  --environments-list /controlled/github/environments.json \
  --confirmatory-environment /controlled/github/confirmatory.json \
  --confirmatory-deployment-policies /controlled/github/confirmatory-policies.json \
  --rehearsal-environment /controlled/github/confirmatory-rehearsal.json \
  --rehearsal-deployment-policies /controlled/github/confirmatory-rehearsal-policies.json \
  --receipt /controlled/github/github-environment-control-receipt.json

fractal-github-environment-control readback \
  --receipt /controlled/github/github-environment-control-receipt.json
```

Publication uses a private same-directory staging file, file synchronization, and an atomic
no-replace link. The loader accepts only the canonical closed JSON object followed by one LF.
The production image workflow captures the five responses with its authenticated GitHub token,
runs this offline verifier, reopens the canonical receipt, and places the exact file in the
checksum-closed image record. The C0 release reopens that authenticated artifact readback and binds
its file digest as `github_environment_control_receipt_file_sha256`. C1 validates that field inside
the closed apparatus-evidence object. The verifier module itself still has no network or credential
authority.
