"""Default launch configuration for the GEMV kernels (fused-decode-gemv
design.md, "Launch configuration"): a heuristic from the queried SM count
and the problem shape, never a GPU name (P-012).

The grid size a GEMV needs is fixed by the problem (one warp per output row
per batch element — gemv.py), so the only real lever here is
`threads_per_block`: fewer threads per block means more, smaller blocks for
the SAME total warp count, which matters when the problem is small relative
to the GPU's SM count (e.g. batch 1 on a GPU with many SMs).
"""

from __future__ import annotations

import dataclasses

from lws.domain.device.properties import SmCount

WARP_SIZE = 32
THREADS_PER_BLOCK_CANDIDATES = (256, 128, 64, 32)
MIN_THREADS_PER_BLOCK = 32
TARGET_BLOCKS_PER_SM = 4


@dataclasses.dataclass(frozen=True)
class GemvShape:
    output_rows: int
    batch_size: int


@dataclasses.dataclass(frozen=True)
class ThreadsPerBlock:
    value: int


@dataclasses.dataclass(frozen=True)
class BlockCount:
    value: int


@dataclasses.dataclass(frozen=True)
class LaunchConfiguration:
    threads_per_block: ThreadsPerBlock
    blocks: BlockCount


def default_launch_configuration(sm_count: SmCount, shape: GemvShape) -> LaunchConfiguration:
    total_warps_needed = _total_warps_needed(shape)
    target_blocks = _target_block_count(sm_count)
    threads_per_block = _choose_threads_per_block(total_warps_needed, target_blocks)
    blocks = _block_count(total_warps_needed, threads_per_block)
    return LaunchConfiguration(threads_per_block=threads_per_block, blocks=blocks)


def _total_warps_needed(shape: GemvShape) -> int:
    rows = shape.output_rows
    batch = shape.batch_size
    return rows * batch


def _target_block_count(sm_count: SmCount) -> int:
    count = sm_count.value
    return count * TARGET_BLOCKS_PER_SM


def _choose_threads_per_block(total_warps_needed: int, target_blocks: int) -> ThreadsPerBlock:
    fitting = tuple(candidate for candidate in THREADS_PER_BLOCK_CANDIDATES if _meets_target(candidate, total_warps_needed, target_blocks))
    if not fitting:
        return ThreadsPerBlock(MIN_THREADS_PER_BLOCK)
    return ThreadsPerBlock(max(fitting))


def _meets_target(candidate: int, total_warps_needed: int, target_blocks: int) -> bool:
    warps_per_block = candidate // WARP_SIZE
    blocks = _ceil_div(total_warps_needed, warps_per_block)
    return blocks >= target_blocks


def _block_count(total_warps_needed: int, threads_per_block: ThreadsPerBlock) -> BlockCount:
    threads = threads_per_block.value
    warps_per_block = threads // WARP_SIZE
    blocks = _ceil_div(total_warps_needed, warps_per_block)
    return BlockCount(blocks)


def _ceil_div(numerator: int, denominator: int) -> int:
    negated = -numerator // denominator
    return -negated
