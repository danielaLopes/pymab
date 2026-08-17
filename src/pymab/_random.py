"""Versioned deterministic random-stream derivation."""

from __future__ import annotations

import hashlib
import json

import numpy as np

RNG_SCHEME_VERSION = "pymab-v2-blake2b-seedsequence-v1"


def stable_seed(master_seed: int, *parts: object) -> int:
    """Derive a stable unsigned seed from a master seed and stream labels."""

    payload = json.dumps([master_seed, *parts], separators=(",", ":")).encode()
    digest = hashlib.blake2b(payload, digest_size=16, person=b"pymab-v2").digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def generator(master_seed: int, *parts: object) -> np.random.Generator:
    """Create a NumPy generator for a stable named random stream."""

    return np.random.default_rng(
        np.random.SeedSequence(stable_seed(master_seed, *parts))
    )


__all__ = ["RNG_SCHEME_VERSION", "generator", "stable_seed"]
