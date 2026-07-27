# OPENED genesis operator

`open_suite_attempt` is the typed state transition. It is not an operator-facing
entry point: it requires an in-memory verified C1 capability, five guarded
production-closure capabilities, five sets of launcher controls, and a closed
attestation descriptor. Passing those arguments by hand would leave the most
consequential transition dependent on shell assembly.

`operators/opened_genesis.py` closes that gap. It has one production action and
two CLI arguments:

```bash
PYTHONPATH=src python -m operators.opened_genesis \
  --request /controlled/suite/opened-genesis-request.json \
  --request-sha256 '<sha256 of the exact request file>'
```

The command performs real GitHub attestation verification and anonymous Zenodo
readback through `verify_production_protocol_registration`. It then reconstructs
all five closure authorities through the production finalization verifier,
loads every runtime control through its typed loader, and calls
`open_suite_attempt`. It does not read corpus sources, benchmark labels,
ciphertexts, predictions, or outcomes.

## Closed custody request

The canonical request uses schema `fractal-opened-genesis-request-v1`:

```json
{
  "inputs": [
    {
      "file_sha256": "<lowercase SHA-256>",
      "path": "/absolute/canonical/path",
      "role": "<fixed role>"
    }
  ],
  "schema_version": "fractal-opened-genesis-request-v1"
}
```

The file ends with one newline. Rows are unique and sorted by the UTF-8 bytes of
`role`. Each path is absolute, canonical, current-user-owned, singly linked,
and not writable by group or other users. The request contains exactly 58
rows:

- the 27 fixed C1 package files, named
  `c1-package/<fixed-package-filename>`;
- `production-finalization-request`;
- `production-finalization-receipt`;
- `protocol-registration-receipt`;
- `protocol-registry-record`;
- `sealed-run-receipt`;
- `suite-attestation-descriptor`;
- for each of `scifact`, `hotpotqa-fullwiki`, `t2-ragbench`, `bright`, and
  `miracl-transfer`: `preflight-launch-contract`,
  `runtime-preflight-receipt`, `runtime-plan-transition`,
  `registered-plan-instantiation`, and `sealed-launch-contract`.

The 27 package rows must share one real directory and use the filenames fixed
by `PACKAGE_FILE_NAMES`. The finalization request must point to that directory
and to the same request-bound registry record, registration receipt, and sealed
run receipt.

The request cannot contain an execution-artifact map, manifest digest, C0 or C1
commit, suite ID, namespace, closure digest, corpus selection, runtime-plan
digest, or attestation-policy field. The operator derives those values from the
verified C1 package and typed finalization apparatus. Any extra request field,
missing role, reordered row, path reuse, or corpus substitution is fatal.

Construct the request only after all 58 inputs have reached their final paths.
For each row, hash the exact file at that path, serialize the closed object with
sorted keys and compact separators, append one newline, and pin the resulting
request SHA-256 in the ceremony record. Do not regenerate it after C1 readback.

## File admission

The operator opens every path component relative to a directory descriptor with
`O_NOFOLLOW`. Each file is opened with `O_NOFOLLOW`, `O_NONBLOCK`, and
`O_CLOEXEC`; FIFOs, devices, sockets, symbolic links, and hard links fail before
typed parsing. File descriptors remain open for the full operation.

Each admission records device, inode, mode, link count, owner, group, size,
nanosecond modification time, and nanosecond status-change time. Reads use
`pread`, compare pre-read and post-read metadata, and check the request digest.
Before typed verification starts, the operator arms a kernel mutation watch on
every retained file and immediate parent, plus every directory ancestor. The
watch remains armed through `open_suite_attempt`. BSD/macOS hosts use vnode
events through `kqueue`; Linux hosts use `inotify` watches attached to the
retained descriptors. A write, metadata change, link change, move, deletion,
revocation, unmount, ignored watch, or event-queue overflow is fatal.

The operator also reopens each parent from `/` before and after C1 verification,
closure reconstruction, and publication. It checks that the path still reaches
the pinned directory and inode, rereads the retained file descriptor, and
rechecks the digest. The kernel watch makes this a continuous consumption
boundary: renaming an ancestor away, letting a typed loader consume a
substitute, and restoring the original before revalidation still fails.

## Exclusive publication

The finalization receipt derives the only output namespace. The operator
accepts no output argument. Under its private parent it takes a nonblocking
advisory lock and reserves two evidence names:

```text
.opened-genesis-<suite-attempt-id>.lock
.opened-genesis-<suite-attempt-id>.attempted
.opened-genesis-<suite-attempt-id>.quarantine
```

The persistent lock file is an empty, singly linked, current-user-owned mode
`0600` regular file. A newly created lock is explicitly set to `0600` and
synchronized, so an inherited restrictive umask cannot produce an unusable
lock. An existing lock is never repaired. While holding it, the operator proves
that the suite namespace and attempt marker are absent.

After the last request, input, ancestor, and output-parent checks, the operator
creates the attempt marker with `O_EXCL`. It first synchronizes the empty
mode-`0600` inode and parent entry, then writes and synchronizes one canonical
record that binds the suite ID, custody-request SHA-256, and attestation
descriptor file SHA-256. Only then may `open_suite_attempt` run under a private
creation umask. The marker is the no-replay boundary. It remains after success,
checked failure, `SystemExit`, a caught termination signal, or fail-clean
namespace quarantine.

A pre-existing valid marker means the suite ID has already been attempted. An
empty, partial, malformed, linked, symbolic, special, foreign-owned, or
wrong-mode marker is uncertain attempt evidence. Both cases forbid execution.
The operator never repairs or removes either form. Thus a parent-fsync failure
after publication can quarantine the exact partial namespace without making
the suite ID reusable.

A valid genesis has exactly:

```text
suite-attempt-<id>/
├── 000.state.json
├── attestation-descriptor.json
└── online/
```

The two directories are mode `0700`; both files are mode `0600`; `online/` is
empty. The operator reloads `000.state.json` with the public typed loader,
checks the manifest, attempt ID, sequence, predecessor, namespace URI, and
`OPENED` state, then rechecks the pinned state bytes and namespace inode. It
synchronizes both directories and their parent.

If publication fails after the namespace is created, cleanup first requires the
exact request-bound attestation-descriptor digest. After the state primitive
returns, the operator also binds cleanup to that namespace inode. Every known
member and the namespace path are rechecked before quarantine. A substituted
directory or member is preserved.

Cleanup never unlinks a namespace member or removes a directory by pathname.
It uses the host's atomic no-replace rename primitive to move the canonical
entry to the suite-scoped quarantine name, then proves that the moved inode is
the retained namespace descriptor. The complete partial tree remains intact in
quarantine. The operator synchronizes that directory and its parent only after
it has proved the canonical namespace absent.

If an entry replacement lands between the last path check and the atomic
rename, the moved inode will differ. The operator atomically restores that
replacement to the canonical name without overwriting any intervening entry,
verifies the restored identity, and reports the publication as indeterminate.
If restoration or synchronization is uncertain, every entry remains in either
the canonical or quarantine location; none is deleted. An unexpected member,
missing or different provenance file, nonregular known member, hard link,
foreign owner, nonempty `online/`, replaced inode, pre-existing quarantine, or
failed synchronization also produces an explicit indeterminate verdict. No
repair or replay follows it.

`KeyboardInterrupt`, `SystemExit`, `SIGHUP`, and `SIGTERM` pass through the same
provenance-checked cleanup path. The original Python exception is reraised;
the CLI maps a caught termination signal to `128 + signal`. `SIGKILL` and host
power loss cannot run process cleanup, so the attempt marker and any namespace
or quarantine remain evidence for the next operator. CLI failures are one line
and contain no traceback.

The directory modes and advisory lock treat the current Unix identity as the
host custody principal. They do not exclude a malicious concurrent process
running as that same identity, and the lock does not coordinate a second host.
The CLI must run in the main thread to install its signal handlers.

On success, stdout contains one canonical JSON line with the request digest,
manifest digest, suite attempt ID, namespace, state-record digest, and
`state: "OPENED"`.

## Acceptance checks

Before using the emitted state:

1. Preserve the exact custody request and its external SHA-256.
2. Confirm that the success object names the C1-derived namespace.
3. Confirm the namespace inventory and modes above.
4. Reload `000.state.json` through the suite verifier before a provider claim.
5. Preserve and verify the suite-scoped attempt marker.
6. Treat any indeterminate-publication error or invalid attempt marker as an
   incident. Do not delete, repair, or rerun that suite ID.
7. After a failed invocation, preserve any suite-scoped quarantine intact.

The operator, tests, and this document live outside the confirmatory image
context. Adding them cannot change the scientific or release image tree.
