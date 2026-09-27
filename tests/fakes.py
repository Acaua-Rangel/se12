"""In-memory fakes for every application port, used by tests instead of real
adapters (constitution P-013: "every port has ... one in-memory fake used by
the tests"). This module is under tests/, so it is exempt from the
calisthenics rules that govern src/lws.
"""

from __future__ import annotations

import typing

from lws.application.ports.codec.weight_source import SourceTensor


class FakeWeightSource:
    """A WeightSource backed by a fixed, in-memory list of tensors."""

    def __init__(self, items: typing.Sequence[SourceTensor]) -> None:
        self._items = list(items)

    def tensors(self) -> typing.Iterator[SourceTensor]:
        return iter(self._items)
