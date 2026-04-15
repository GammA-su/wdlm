"""Deterministic random helpers."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Sequence, TypeVar


T = TypeVar("T")


def derive_seed(seed: int, *parts: object) -> int:
    """Derive a stable integer seed from a base seed and additional labels."""

    material = "|".join([str(seed), *[str(part) for part in parts]]).encode("utf-8")
    digest = hashlib.sha256(material).hexdigest()
    return int(digest[:16], 16)


@dataclass
class SeededRNG:
    """Small wrapper around ``random.Random`` with deterministic substreams."""

    seed: int
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def derive(self, *parts: object) -> "SeededRNG":
        """Create a child RNG from the current seed."""

        return SeededRNG(derive_seed(self.seed, *parts))

    def choice(self, items: Sequence[T]) -> T:
        """Return a deterministic choice from a non-empty sequence."""

        if not items:
            raise ValueError("Cannot choose from an empty sequence.")
        return items[self._rng.randrange(len(items))]

    def randint(self, lower: int, upper: int) -> int:
        """Return a deterministic integer in the inclusive range."""

        return self._rng.randint(lower, upper)

    def shuffle(self, items: Sequence[T]) -> list[T]:
        """Return a shuffled copy of the input sequence."""

        values = list(items)
        self._rng.shuffle(values)
        return values

    def coin_flip(self) -> bool:
        """Return a deterministic boolean."""

        return bool(self._rng.getrandbits(1))
