"""Realistic GitHub release-attestation fixtures shared by boundary tests."""

import base64
from urllib.parse import quote

from kapso.cross_run.canonical import canonical_json_bytes


def release_attestation(repository, tag, commit_sha, asset_digests):
    package_uri = f"pkg:github/{repository}@{quote(tag, safe='')}"
    subjects = [
        {"uri": package_uri, "digest": {"sha1": commit_sha}},
        *[
            {
                "name": name,
                "digest": {"sha256": digest.removeprefix("sha256:")},
            }
            for name, digest in sorted(asset_digests.items())
        ],
    ]
    statement = {
        "_type": "https://in-toto.io/Statement/v1",
        "predicateType": "https://in-toto.io/attestation/release/v0.2",
        "subject": subjects,
        "predicate": {
            "purl": package_uri,
            "repository": repository,
            "tag": tag,
        },
    }
    return {
        "attestation": {
            "initiator": "",
            "bundle": {
                "dsseEnvelope": {
                    "payload": base64.b64encode(canonical_json_bytes(statement)).decode(
                        "ascii"
                    ),
                    "payloadType": "application/vnd.in-toto+json",
                    "signatures": [{"sig": "fixture"}],
                },
                "verificationMaterial": {"fixture": True},
            },
        },
        "verificationResult": {"statement": statement},
    }
