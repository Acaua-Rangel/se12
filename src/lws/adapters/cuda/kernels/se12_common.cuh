// Shared SE12 decode logic (weight-codec design.md, "Decode";
// lws.domain.se12.codec is the CPU oracle this must match bit for bit,
// AC-009). Included by both the decode-only kernel (se12_decode.cu, T-004)
// and the fused decode+GEMV kernel (gemv_se12.cu, T-005), so the two paths
// never drift apart on what "decode" means.
//
// UNVERIFIED ON REAL HARDWARE (fused-decode-gemv spec.md: proof runs on
// Kaggle or another real GPU). This environment has no NVIDIA driver — not
// even NVRTC loads here (no libnvrtc.so) — so nothing under
// src/lws/adapters/cuda/kernels/ has been compiled or run. Written carefully
// against weight-codec design.md and the exact byte layout
// lws.domain.se12.codec.encode_tensor produces (verified there, on CPU, with
// 200+ random fuzz trials plus forced-fallback and float-edge-case tests),
// but it must be treated as a first draft until a `gpu`-marked test run
// confirms every byte-order and bit-order choice below on real hardware.
//
// Portable across every supported architecture (sm_60 and up): no
// __CUDA_ARCH__-guarded path, no tensor-core or async-copy instructions
// (principle P-008 — those are for the fused kernel's optional fast path,
// gated behind #if __CUDA_ARCH__ >= 800, never here).
#pragma once

#define LWS_TILE_ROWS 16
#define LWS_TILE_COLS 256
#define LWS_TILE_SIZE (LWS_TILE_ROWS * LWS_TILE_COLS)
#define LWS_ESCAPE_BUDGET 16
#define LWS_ESCAPE_CODE 15
#define LWS_CODEBOOK_SIZE 15

// Rebuilds one exact BF16 bit pattern from its exponent byte and its `sm`
// byte = (sign << 7) | mantissa (weight-codec design.md's decode formula):
//   bits = ((sm & 0x80) << 8) | (exp << 7) | (sm & 0x7F)
// Matches lws.domain.weights.join_bit_pattern exactly (verified there).
__device__ __forceinline__ unsigned short lws_bits_from_parts(unsigned char sm, unsigned char exponent) {
    unsigned int sign_bit = (static_cast<unsigned int>(sm) & 0x80u) << 8;
    unsigned int mantissa_bits = static_cast<unsigned int>(sm) & 0x7Fu;
    unsigned int exponent_bits = static_cast<unsigned int>(exponent) << 7;
    return static_cast<unsigned short>(sign_bit | exponent_bits | mantissa_bits);
}

// Looks up one weight's 4-bit exponent code from a tile's packed `ec` bytes
// (lws.domain.se12.codec._pack_ec: 2 codes per byte, even flat index in the
// low nibble, odd in the high nibble).
__device__ __forceinline__ unsigned char lws_ec_code(const unsigned char* tile_ec, int flat_index) {
    unsigned char byte = tile_ec[flat_index >> 1];
    unsigned char low_nibble = byte & 0x0Fu;
    unsigned char high_nibble = (byte >> 4) & 0x0Fu;
    bool is_even = (flat_index & 1) == 0;
    return is_even ? low_nibble : high_nibble;
}

// True when tile `tile_index` (of `tile_count` total, in row-major tile
// order) is a fallback tile (lws.domain.se12.codec._pack_bitmap /
// _unpack_bitmap: numpy.packbits' default bit order — element i of a
// byte-group of 8 is bit (7 - i % 8), MSB first, empirically confirmed
// against numpy before writing this).
__device__ __forceinline__ bool lws_is_fallback_tile(const unsigned char* fallback_bitmap, int tile_index) {
    int byte_index = tile_index >> 3;
    int bit_position = 7 - (tile_index & 7);
    unsigned char byte = fallback_bitmap[byte_index];
    return ((byte >> bit_position) & 1u) != 0;
}

// Decodes every weight of ONE non-fallback tile, sequentially, into `out`
// (LWS_TILE_SIZE uint16 bit patterns, tile-local row-major order — matching
// lws.domain.se12.codec._to_tiles' flatten convention exactly: local row
// outer, local column inner). One thread per tile: this kernel's job is
// correctness (US-004: "a correctness oracle ... and a fallback path"), not
// peak throughput — the fused kernel (T-005) is where per-weight
// parallelism and the performance engineering happen, reusing this same
// escape-rank scheme but amortized over a whole GEMV pass per tile.
//
// The caller must not call this for a fallback tile (its `esc` slots may
// not hold the escapes for every weight of the tile — read from
// fallback_raw directly instead, via the tile's fallback_offsets entry).
__device__ void lws_decode_tile(
    const unsigned char* tile_sm,
    const unsigned char* tile_ec,
    const unsigned char* tile_esc,
    const unsigned char* codebook,
    unsigned short* out
) {
    int escape_rank = 0;
    for (int flat_index = 0; flat_index < LWS_TILE_SIZE; ++flat_index) {
        unsigned char code = lws_ec_code(tile_ec, flat_index);
        unsigned char exponent;
        // codebook has only LWS_CODEBOOK_SIZE (15) entries — code 15 (escape)
        // must never index into it, so the branches read from disjoint
        // sources rather than computing codebook[code] unconditionally.
        if (code == LWS_ESCAPE_CODE) {
            exponent = tile_esc[escape_rank];
            escape_rank += 1;
        } else {
            exponent = codebook[code];
        }
        out[flat_index] = lws_bits_from_parts(tile_sm[flat_index], exponent);
    }
}
