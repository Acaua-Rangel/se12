# Design: SE12 — Split-Exponent, 12 bits per weight

> feature: weight-codec

## BF16 anatomy

```
bit 15   14 ........ 7   6 ........ 0
 sign     exponent(8)     mantissa(7)
```

Sign and mantissa behave like random bits (≈8 bits of entropy). The exponent
distribution of trained weights is very concentrated (a few adjacent values
dominate), so it carries ≈2.5–3 bits. SE12 stores:

| Stream | Bits/weight | Content |
|---|---|---|
| `sm` | 8 | `(sign << 7) \| mantissa`, raw |
| `ec` | 4 | exponent code: 0..14 → index into the tensor's 15-entry codebook; 15 → ESCAPE |
| `esc` | amortized | per-tile slots holding the full 8-bit exponent of escaped weights, in order of appearance |

## Tiles (random access)

The matrix `W[N_out, K]` is cut in tiles of `TILE_N × TILE_K` (default
16 × 256 = 4096 weights; tunable, recorded in metadata). Each tile has a fixed
escape budget `E` (default 16 slots = 128 bits → 0.03 bit/weight). Because
every tile has the same byte size, the address of tile `(i, j)` is pure
arithmetic — no offset table, no prefix sum across tiles.

If a tile has more than `E` escapes, it is a **fallback tile**: its 4096
weights are stored raw in a side region, and a per-tensor bitmap marks it. The
decoder checks one bit per tile. Target: fallback tiles < 0.1% of tiles.

## Decode (what the GPU will do per weight)

```
code  = (ec_byte >> shift) & 0xF
rank  = popcount(escape_mask_of_tile & ((1 << lane) - 1))   # only when code == 15
exp   = code == 15 ? esc[tile][rank] : codebook[code]
bits  = ((sm & 0x80) << 8) | (exp << 7) | (sm & 0x7F)
```

Shifts, masks, one 16-entry lookup (lives in registers/shared memory) and a
rare popcount — no data-dependent loops, no variable-length parsing.

## Size

12 bits + 0.03 (escapes) + codebook/bitmap (negligible) ≈ 12.03 bits →
ratio ≈ 0.752. The theoretical upper bound for bandwidth-bound decode speed is
therefore ≈ 16 / 12.03 ≈ 1.33×, before decode overhead. Entropy coders reach
≈ 11 bits (ratio 0.69) but need variable-length parsing — out of scope.

## Container

One `.safetensors` file per original shard. For each eligible tensor `name`:
`name.se12.sm` (uint8), `name.se12.ec` (uint8, 2 codes per byte),
`name.se12.esc` (uint8), `name.se12.codebook` (uint8[15]),
`name.se12.fallback_bitmap` (uint8), `name.se12.fallback_raw` (int16).
File-level metadata (safetensors `__metadata__`): `format=SE12`,
`format_version=1`, `tile_n`, `tile_k`, `escape_budget`, original shape and
dtype per tensor.

The format is GPU-agnostic (principle P-010): packing runs on CPU, is
deterministic (same input → same bytes), and stores nothing about the GPU that
will read it. Tile shape is part of the format and is the same on every GPU;
per-GPU speed tuning only changes launch parameters, never the layout.

## External evidence: Split12 (closest public experiment)

Source: Brian Bell, `brianbell-x/weight-compression` — method folder
`Split12/`, serving record `Split12/inference/README.md`, writeup
https://brianbell-x.github.io/weight-compression/Split12/ (read 2026-09-27).
Numbers below are the author's; the inference record says its raw artifacts
and kernels were lost and the entries are rebuilt from research ledgers, so
they are documentation, not reproducible results.

**Format — the same bet as SE12, split at a byte boundary.** The high byte
(sign + 7 exponent bits) comes from a codebook + indices + an escape stream;
the low byte (last exponent bit + 7 mantissa bits) is kept raw. Measured
12.005 bits/weight (−24.967%), all 59,509 BF16 tensors of `zai-org/GLM-5.2`
(753B) round-tripped bit for bit. A "K15" layout — the 9-bit sign+exponent
symbol as a 4-bit code into a 15-entry table, escapes for the rest, 7-bit
mantissa raw — is priced at 11.173 bits/weight, but as an accounting estimate,
never decoded at that scale. This independently supports ASM-001 on a very
different model; K15 is a candidate for a later `format_version`, not for v1.

**Kernel-level evidence (relevant to fused-decode-gemv):**

- A40, dense 12-bit prototype decoding in registers: 0.733× the BF16 GEMV
  time — **without** the escape correction, which was "validated separately
  but not fused into or included in that timing". It shows the bandwidth
  saving can survive the common-path decode on one GPU; it is NOT a lossless
  GO in the sense of AC-014.
- B300, scalar (CUDA-core) batch-tiled kernels: nothing crossed 1.0× BF16;
  the author calls them "issue/latency-bound on per-element integer
  reconstruction" and says this confirms "the earlier H100 verdict". Consistent
  with this plan's expectation that HBM datacenter GPUs are the hardest
  (ASM-005) and that the verdict must be per GPU (P-011).
- B300, tensor cores: decoding straight into `mma.sync` fragment registers
  reached 0.80× BF16 on one MoE shape; dense shapes stayed 1.21–1.86× slower,
  with a ~10–12 µs launch floor — the reason for stacked mode and CUDA graphs
  in fused-decode-gemv, and the path left out of scope there.

**End-to-end evidence (relevant to model-integration):**

- GLM-4-9B: +20.1% tok/s on an RTX A6000 and +3.4% on an RTX 6000 Ada at
  batch 1 (development harness).
- Full GLM-5.2 on 8× B300: +27% against a reference harness, then −64.6%
  (73.6 vs 208 tok/s) against a matched production SGLang setup — the earlier
  baseline "was a harness artifact". The kernel win did not survive a strong
  baseline, which is why this plan measures against the fastest lossless
  baseline on each GPU (P-009) and runs the e2e benchmark separately (AC-018).
- Generated tokens diverged from BF16 after a 26/32-token matching prefix
  from reduction order alone (weights exact) — the risk AC-017 and ASM-013
  measure.
