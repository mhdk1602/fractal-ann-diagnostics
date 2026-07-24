from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest
from py_ecc.bls.point_compression import compress_G1, compress_G2
from py_ecc.optimized_bls12_381 import FQ, G2, Z1, iso_map_G1, multiply, optimized_swu_G1

from fractal_ann_diagnostics.drand_beacon import (
    MAX_BEACON_BYTES,
    MAX_CHAIN_INFO_BYTES,
    QUICKNET_CHAIN_HASH,
    QUICKNET_GENESIS_UNIX_SECONDS,
    QUICKNET_GROUP_HASH,
    QUICKNET_NETWORK,
    QUICKNET_PERIOD_SECONDS,
    QUICKNET_PUBLIC_KEY,
    QUICKNET_SCHEME_ID,
    DrandBeaconError,
    DrandHttpResponse,
    QuicknetExecutionBeaconVerifier,
    fetch_quicknet_beacon,
)
from fractal_ann_diagnostics.execution_claim import ExecutionBeaconContract

ROUND_1_SIGNATURE = (
    "b55e7cb2d5c613ee0b2e28d6750aabbb78c39dcc96bd9d38c2c2e12198df955"
    "71de8e8e402a0cc48871c7089a2b3af4b"
)
ROUND_1_RANDOMNESS = "1466a6cd24e327188770752f6134001c64d6efcc590ccc26b721611ad96f165a"
ROUND_1_BYTES = (
    b'{"round":1,"randomness":"1466a6cd24e327188770752f6134001c64d6efcc590ccc26'
    b'b721611ad96f165a","signature":"b55e7cb2d5c613ee0b2e28d6750aabbb78c39dcc96bd9d38'
    b'c2c2e12198df95571de8e8e402a0cc48871c7089a2b3af4b"}'
)
ROUND_2_SIGNATURE = (
    "b6b6a585449b66eb12e875b64fcbab3799861a00e4dbf092d99e969a5eac57dd3"
    "f798acf61e705fe4f093db926626807"
)
ROUND_2_RANDOMNESS = "5782d6987841c654515a0e72b2d1ebb4e741234042c37cb19608ae50d93fb60c"
CHAIN_INFO = {
    "public_key": QUICKNET_PUBLIC_KEY,
    "period": QUICKNET_PERIOD_SECONDS,
    "genesis_time": QUICKNET_GENESIS_UNIX_SECONDS,
    "hash": QUICKNET_CHAIN_HASH,
    "groupHash": QUICKNET_GROUP_HASH,
    "schemeID": QUICKNET_SCHEME_ID,
    "metadata": {"beaconID": "quicknet"},
}
CHAIN_INFO_BYTES = json.dumps(CHAIN_INFO, separators=(",", ":")).encode("ascii")


class FakeApi:
    def __init__(self, responses: dict[str, DrandHttpResponse]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, int]] = []

    def get(self, url: str, *, max_bytes: int) -> DrandHttpResponse:
        self.calls.append((url, max_bytes))
        return self.responses[url]


def _contract() -> ExecutionBeaconContract:
    return ExecutionBeaconContract(
        drand_network=QUICKNET_NETWORK,
        chain_hash=QUICKNET_CHAIN_HASH,
        chain_scheme_id=QUICKNET_SCHEME_ID,
        chain_public_key=QUICKNET_PUBLIC_KEY,
        chain_genesis_unix_seconds=QUICKNET_GENESIS_UNIX_SECONDS,
        chain_period_seconds=QUICKNET_PERIOD_SECONDS,
        execution_round=1,
        label_release_round=101,
        minimum_label_release_safety_rounds=100,
        verification_identity="a" * 64,
    )


def _response(url: str, body: bytes, *, status: int = 200) -> DrandHttpResponse:
    return DrandHttpResponse(
        status=status,
        headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
        body=body,
        final_url=url,
    )


def _api(
    contract: ExecutionBeaconContract | None = None,
    *,
    chain_info: bytes = CHAIN_INFO_BYTES,
    beacon: bytes = ROUND_1_BYTES,
) -> FakeApi:
    admitted = _contract() if contract is None else contract
    info_url = QuicknetExecutionBeaconVerifier.chain_info_url(admitted)
    beacon_url = QuicknetExecutionBeaconVerifier.beacon_url(admitted)
    return FakeApi(
        {
            info_url: _response(info_url, chain_info),
            beacon_url: _response(beacon_url, beacon),
        }
    )


def _beacon_bytes(*, round_number: int, signature: str, randomness: str) -> bytes:
    return json.dumps(
        {"round": round_number, "randomness": randomness, "signature": signature},
        separators=(",", ":"),
    ).encode("ascii")


def test_official_quicknet_round_one_vector_verifies_actual_bls_signature() -> None:
    contract = _contract()
    claims = QuicknetExecutionBeaconVerifier(_api()).verify(
        contract=contract,
        beacon_bytes=ROUND_1_BYTES,
    )

    assert claims.round == 1
    assert claims.signature == ROUND_1_SIGNATURE
    assert claims.randomness == ROUND_1_RANDOMNESS
    assert claims.signature_verified is True
    assert claims.beacon_bytes_sha256 == hashlib.sha256(ROUND_1_BYTES).hexdigest()


def test_fetch_reconstructs_chain_identity_uses_exact_urls_and_returns_exact_bytes() -> None:
    contract = _contract()
    api = _api(contract)

    observed = fetch_quicknet_beacon(contract, api=api)

    assert observed == ROUND_1_BYTES
    assert api.calls == [
        (
            f"{QUICKNET_NETWORK}/{QUICKNET_CHAIN_HASH}/info",
            MAX_CHAIN_INFO_BYTES,
        ),
        (
            f"{QUICKNET_NETWORK}/{QUICKNET_CHAIN_HASH}/public/1",
            MAX_BEACON_BYTES,
        ),
    ]


def test_rejects_wrong_round_before_signature_use() -> None:
    beacon = _beacon_bytes(
        round_number=2,
        signature=ROUND_2_SIGNATURE,
        randomness=ROUND_2_RANDOMNESS,
    )
    with pytest.raises(DrandBeaconError, match="another round"):
        QuicknetExecutionBeaconVerifier(_api()).verify(
            contract=_contract(),
            beacon_bytes=beacon,
        )


def test_rejects_valid_signature_for_another_round() -> None:
    beacon = _beacon_bytes(
        round_number=1,
        signature=ROUND_2_SIGNATURE,
        randomness=ROUND_2_RANDOMNESS,
    )
    with pytest.raises(DrandBeaconError, match="BLS signature is invalid"):
        QuicknetExecutionBeaconVerifier(_api()).verify(
            contract=_contract(),
            beacon_bytes=beacon,
        )


def test_rejects_another_valid_g2_public_key_even_before_pairing() -> None:
    compressed = compress_G2(multiply(G2, 2))
    wrong_key = b"".join(int(value).to_bytes(48, "big") for value in compressed).hex()
    with pytest.raises(DrandBeaconError, match="chain_public_key differs"):
        QuicknetExecutionBeaconVerifier(_api()).verify(
            contract=replace(_contract(), chain_public_key=wrong_key),
            beacon_bytes=ROUND_1_BYTES,
        )


def test_rejects_signature_point_outside_prime_order_subgroup() -> None:
    non_subgroup_point = iso_map_G1(*optimized_swu_G1(FQ(1)))
    signature = int(compress_G1(non_subgroup_point)).to_bytes(48, "big")
    beacon = _beacon_bytes(
        round_number=1,
        signature=signature.hex(),
        randomness=hashlib.sha256(signature).hexdigest(),
    )
    with pytest.raises(DrandBeaconError, match="outside the prime-order subgroup"):
        QuicknetExecutionBeaconVerifier(_api()).verify(
            contract=_contract(),
            beacon_bytes=beacon,
        )


def test_rejects_infinity_noncanonical_and_case_malleability() -> None:
    infinity = int(compress_G1(Z1)).to_bytes(48, "big")
    infinity_beacon = _beacon_bytes(
        round_number=1,
        signature=infinity.hex(),
        randomness=hashlib.sha256(infinity).hexdigest(),
    )
    verifier = QuicknetExecutionBeaconVerifier(_api())
    with pytest.raises(DrandBeaconError, match="point at infinity"):
        verifier.verify(contract=_contract(), beacon_bytes=infinity_beacon)

    uppercase = ROUND_1_BYTES.replace(
        ROUND_1_SIGNATURE.encode(),
        ROUND_1_SIGNATURE.upper().encode(),
    )
    with pytest.raises(DrandBeaconError, match="lowercase hex"):
        verifier.verify(contract=_contract(), beacon_bytes=uppercase)

    duplicate = ROUND_1_BYTES.replace(b'{"round":1,', b'{"round":1,"round":1,')
    with pytest.raises(DrandBeaconError, match="repeats JSON key"):
        verifier.verify(contract=_contract(), beacon_bytes=duplicate)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("hash", "0" * 64, "hash is not canonical"),
        ("public_key", "8" * 192, "hash is not canonical"),
        ("schemeID", "bls-unchained-on-g1", "scheme differs"),
        ("period", 4, "hash is not canonical"),
        ("genesis_time", QUICKNET_GENESIS_UNIX_SECONDS + 1, "hash is not canonical"),
    ),
)
def test_fetch_rejects_substituted_chain_metadata(
    field: str,
    value: object,
    message: str,
) -> None:
    metadata = {**CHAIN_INFO, field: value}
    encoded = json.dumps(metadata, separators=(",", ":")).encode("ascii")
    with pytest.raises(DrandBeaconError, match=message):
        fetch_quicknet_beacon(_contract(), api=_api(chain_info=encoded))


def test_fetch_rejects_redirect_final_url_substitution_and_oversize() -> None:
    contract = _contract()
    info_url = QuicknetExecutionBeaconVerifier.chain_info_url(contract)
    beacon_url = QuicknetExecutionBeaconVerifier.beacon_url(contract)
    redirect = FakeApi(
        {
            info_url: _response(info_url, CHAIN_INFO_BYTES, status=302),
            beacon_url: _response(beacon_url, ROUND_1_BYTES),
        }
    )
    with pytest.raises(DrandBeaconError, match="redirect"):
        fetch_quicknet_beacon(contract, api=redirect)

    changed_url = f"{QUICKNET_NETWORK}/{QUICKNET_CHAIN_HASH}/public/2"
    substitution = FakeApi(
        {
            info_url: _response(changed_url, CHAIN_INFO_BYTES),
            beacon_url: _response(beacon_url, ROUND_1_BYTES),
        }
    )
    with pytest.raises(DrandBeaconError, match="URL changed"):
        fetch_quicknet_beacon(contract, api=substitution)

    oversized = FakeApi(
        {
            info_url: _response(info_url, b"x" * (MAX_CHAIN_INFO_BYTES + 1)),
            beacon_url: _response(beacon_url, ROUND_1_BYTES),
        }
    )
    with pytest.raises(DrandBeaconError, match="byte bound"):
        fetch_quicknet_beacon(contract, api=oversized)

    truncated = FakeApi(
        {
            info_url: DrandHttpResponse(
                status=200,
                headers={"Content-Type": "application/json", "Content-Length": "999"},
                body=CHAIN_INFO_BYTES,
                final_url=info_url,
            ),
            beacon_url: _response(beacon_url, ROUND_1_BYTES),
        }
    )
    with pytest.raises(DrandBeaconError, match="differs from Content-Length"):
        fetch_quicknet_beacon(contract, api=truncated)


def test_rejects_non_exact_contract_endpoint_and_extra_beacon_fields() -> None:
    with pytest.raises(DrandBeaconError, match="drand_network differs"):
        QuicknetExecutionBeaconVerifier(_api()).verify(
            contract=replace(_contract(), drand_network="https://api2.drand.sh"),
            beacon_bytes=ROUND_1_BYTES,
        )
    extra = ROUND_1_BYTES[:-1] + b',"previous_signature":"00"}'
    with pytest.raises(DrandBeaconError, match="exact schema"):
        QuicknetExecutionBeaconVerifier(_api()).verify(
            contract=_contract(),
            beacon_bytes=extra,
        )
