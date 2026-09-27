"""Use case: pack a whole model into SE12, shard by shard (US-003).

Eligibility (Q-002, decided here and only here): a tensor is SE12-encoded
when it is BF16 AND 2-D. Everything else — a different dtype, or a BF16
tensor that is not 2-D (norm scales, biases — ASM-002) — is copied through
unchanged (AC-007). The tied embedding / LM head is an ordinary 2-D BF16
weight, so it is eligible like any other (Q-002, answered).
"""

from __future__ import annotations

import dataclasses

from lws.application.ports.codec.pack_source import PackSource, ShardOfTensors
from lws.application.ports.codec.packed_store import PackedEntries, PackedEntry, PackedShard, PackedStore, PassThroughTensor, Se12PackedTensor
from lws.application.ports.codec.weight_source import EligibleTensor, IneligibleSourceTensor, SourceTensor
from lws.domain.se12.codec import encode_tensor
from lws.domain.se12.summary import ByteCount, ByteSizes, add_sizes, se12_tensor_byte_size, zero_sizes
from lws.domain.se12.tile import MatrixShape
from lws.domain.weights import ElementCount, RawTensor, RawTensorPayload, ShapedWeights, bf16_weights_as_raw_bytes


@dataclasses.dataclass(frozen=True)
class PackedTensorCounts:
    encoded: ElementCount
    passed_through: ElementCount


@dataclasses.dataclass(frozen=True)
class PackSummary:
    """The eligible-tensor byte totals (AC-006) and how many tensors of each kind were packed."""

    eligible_sizes: ByteSizes
    tensor_counts: PackedTensorCounts


@dataclasses.dataclass(frozen=True)
class _PackedItemResult:
    entry: PackedEntry
    eligible_sizes: ByteSizes | None  # None for a pass-through entry (AC-006 covers eligible tensors only)


@dataclasses.dataclass(frozen=True)
class _ShardPackReport:
    eligible_sizes: ByteSizes
    tensor_counts: PackedTensorCounts


@dataclasses.dataclass(frozen=True)
class _ShardPackResult:
    shard: PackedShard
    report: _ShardPackReport


def pack_model(source: PackSource, store: PackedStore) -> PackSummary:
    running_sizes = zero_sizes()
    running_encoded = 0
    running_passed_through = 0
    for shard in source.shards():
        shard_result = _pack_shard(shard)
        store.write_shard(shard_result.shard)
        report = shard_result.report
        counts = report.tensor_counts
        encoded_count = counts.encoded
        passed_through_count = counts.passed_through
        running_sizes = add_sizes(running_sizes, report.eligible_sizes)
        running_encoded += encoded_count.value
        running_passed_through += passed_through_count.value
    counts = PackedTensorCounts(encoded=ElementCount(running_encoded), passed_through=ElementCount(running_passed_through))
    return PackSummary(eligible_sizes=running_sizes, tensor_counts=counts)


def _pack_shard(shard: ShardOfTensors) -> _ShardPackResult:
    results = tuple(_pack_one(item) for item in shard.tensors)
    entries = tuple(_entry_of(result) for result in results)
    eligible_sizes = _sum_eligible_sizes(results)
    counts = _count_by_kind(results)
    packed_shard = PackedShard(shard_name=shard.shard_name, entries=PackedEntries(items=entries))
    report = _ShardPackReport(eligible_sizes=eligible_sizes, tensor_counts=counts)
    return _ShardPackResult(shard=packed_shard, report=report)


def _entry_of(result: _PackedItemResult) -> PackedEntry:
    return result.entry


def _sum_eligible_sizes(results: tuple[_PackedItemResult, ...]) -> ByteSizes:
    total = zero_sizes()
    for result in results:
        total = _add_if_eligible(total, result)
    return total


def _add_if_eligible(total: ByteSizes, result: _PackedItemResult) -> ByteSizes:
    sizes = result.eligible_sizes
    if sizes is None:
        return total
    return add_sizes(total, sizes)


def _count_by_kind(results: tuple[_PackedItemResult, ...]) -> PackedTensorCounts:
    encoded = sum(1 for result in results if result.eligible_sizes is not None)
    passed_through = len(results) - encoded
    return PackedTensorCounts(encoded=ElementCount(encoded), passed_through=ElementCount(passed_through))


def _pack_one(item: SourceTensor) -> _PackedItemResult:
    if isinstance(item, EligibleTensor):
        return _pack_eligible(item)
    return _pack_ineligible(item)


def _pack_eligible(item: EligibleTensor) -> _PackedItemResult:
    name = item.name
    shaped = item.tensor
    shape = shaped.shape
    dims = shape.dims
    if len(dims) == 2:
        return _pack_as_se12(name, shaped, dims)
    return _pack_as_passthrough_bf16(name, shaped)


def _pack_as_se12(name: str, shaped: ShapedWeights, dims: tuple[int, ...]) -> _PackedItemResult:
    weights = shaped.weights
    matrix_shape = MatrixShape(rows=dims[0], cols=dims[1])
    encoded = encode_tensor(weights, matrix_shape)
    entry = Se12PackedTensor(name=name, encoded=encoded)
    original_bytes = _bf16_byte_size(dims)
    packed_bytes = se12_tensor_byte_size(encoded)
    sizes = ByteSizes(original=ByteCount(original_bytes), packed=packed_bytes)
    return _PackedItemResult(entry=entry, eligible_sizes=sizes)


def _pack_as_passthrough_bf16(name: str, shaped: ShapedWeights) -> _PackedItemResult:
    weights = shaped.weights
    shape = shaped.shape
    raw_bytes = bf16_weights_as_raw_bytes(weights)
    payload = RawTensorPayload(bytes_value=raw_bytes, shape=shape)
    raw_tensor = RawTensor(dtype_name="torch.bfloat16", payload=payload)
    entry = PassThroughTensor(name=name, raw=raw_tensor)
    return _PackedItemResult(entry=entry, eligible_sizes=None)


def _pack_ineligible(item: IneligibleSourceTensor) -> _PackedItemResult:
    name = item.name
    raw_tensor = item.tensor
    entry = PassThroughTensor(name=name, raw=raw_tensor)
    return _PackedItemResult(entry=entry, eligible_sizes=None)


def _bf16_byte_size(dims: tuple[int, ...]) -> int:
    total_elements = dims[0] * dims[1]
    bytes_per_element = 2
    return total_elements * bytes_per_element
