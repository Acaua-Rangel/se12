// Fused SE12 decode + GEMV kernel (US-005, AC-010, AC-011): computes
// y = x . W^T reading ONLY the SE12 streams — the full BF16 weight matrix
// never crosses the memory bus, never touches global memory as a decoded
// intermediate (AC-010's "never writes decoded weights to global memory").
//
// Same skeleton as gemv_raw_bf16.cu (principle P-009: one warp per (output
// row, batch element), lock-step stream over K, warp-shuffle reduction) —
// only the per-weight load step changes from "read 2 bytes" to "read sm +
// ec (+ rare esc/fallback_raw) and reconstruct". Escape rank is computed
// per warp with __ballot_sync + __popc against a host-precomputed
// per-(output row, k-tile) starting offset (gemv.py) — this design was
// verified in Python (against the CPU reference codec, lws.domain.se12.codec)
// before being written here, see the fused-decode-gemv task history; the
// CUDA translation itself is UNVERIFIED ON REAL HARDWARE (se12_common.cuh's
// note applies here too).
//
// KNOWN LIMITATION (first draft): assumes K is an exact multiple of
// LWS_TILE_COLS (256) — true for every Linear shape of the reference model
// (weight-codec Q-001) but not the general case AC-011 also asks for
// (out-of-bounds-safe reads when K is not a multiple of the tile size).
// That generalization is follow-up work, not implemented here.
#include "se12_common.cuh"

#ifndef LWS_EXTRA_DECODE_OPS
#define LWS_EXTRA_DECODE_OPS 0
#endif

// performance-model AC-027's controlled experiment: LWS_EXTRA_DECODE_OPS
// dependent integer ops per weight that leave `value` unchanged. Each
// iteration XORs the same data-dependent mask into the value in an
// `asm volatile` block — real PTX `xor.b32` instructions NVRTC cannot prove
// are dead and remove (unlike plain C, where `x ^ mask ^ mask` folds to `x`
// at compile time and the whole computation can vanish). XOR-ing the SAME
// mask an EVEN number of times exactly cancels; an ODD count would corrupt
// the value once more — AC-027's sweep only ever uses even k (0, 2, 4, 8,
// 12, 16, 24, 32), so this is never a correctness risk in practice. Used
// ONLY by the benchmark's decode-cost sweep — LWS_EXTRA_DECODE_OPS is 0 (no
// extra work, the loop below does not execute) in every other build,
// including the runtime's own kernel compiles.
__device__ __forceinline__ unsigned short lws_extra_decode_ops(unsigned short value) {
    unsigned int register_value = static_cast<unsigned int>(value);
    unsigned int mask = register_value * 2654435761u; // Knuth multiplicative hash: data-dependent, not a compile-time constant
#pragma unroll
    for (int i = 0; i < LWS_EXTRA_DECODE_OPS; ++i) {
        asm volatile("xor.b32 %0, %0, %1;" : "+r"(register_value) : "r"(mask));
    }
    return static_cast<unsigned short>(register_value);
}

extern "C" __global__ void lws_gemv_se12(
    const unsigned char* sm,               // [tile_count, LWS_TILE_SIZE]
    const unsigned char* ec,               // [tile_count, LWS_TILE_SIZE/2]
    const unsigned char* esc,              // [tile_count, LWS_ESCAPE_BUDGET]
    const unsigned char* codebook,         // [LWS_CODEBOOK_SIZE], tensor-wide
    const unsigned char* fallback_bitmap,  // packed bits, one per tile
    const int* fallback_offsets,           // [tile_count]
    const short* fallback_raw,             // [num_fallback_tiles, LWS_TILE_SIZE]
    const int* row_escape_offset,          // [N, K/LWS_TILE_COLS]: escapes in EARLIER rows of this row's tile
    const float* activations,              // [B, K] FP32
    float* output,                         // [B, N] FP32
    int N,
    int K,
    int B
) {
    int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_thread / warpSize;
    int lane = global_thread % warpSize;

    int total_warps_needed = N * B;
    if (warp_id >= total_warps_needed) {
        return;
    }

    int n = warp_id % N;
    int b = warp_id / N;

    int k_tile_count = K / LWS_TILE_COLS;
    int tile_row = n / LWS_TILE_ROWS;
    int local_row = n % LWS_TILE_ROWS;
    int n_tile_cols = k_tile_count;

    const float* x = activations + static_cast<long long>(b) * K;
    float accumulator = 0.0f;

    for (int tile_col = 0; tile_col < k_tile_count; ++tile_col) {
        int tile_index = tile_row * n_tile_cols + tile_col;
        int row_offset_in_tile = local_row * LWS_TILE_COLS;
        int k_base = tile_col * LWS_TILE_COLS;

        if (lws_is_fallback_tile(fallback_bitmap, tile_index)) {
            int fallback_index = fallback_offsets[tile_index];
            const short* tile_raw = fallback_raw + static_cast<long long>(fallback_index) * LWS_TILE_SIZE;
            for (int local_c = lane; local_c < LWS_TILE_COLS; local_c += warpSize) {
                unsigned short bits = static_cast<unsigned short>(tile_raw[row_offset_in_tile + local_c]);
                bits = lws_extra_decode_ops(bits);
                unsigned int widened = static_cast<unsigned int>(bits) << 16;
                float w = __uint_as_float(widened);
                accumulator += w * x[k_base + local_c];
            }
            continue;
        }

        const unsigned char* tile_sm = sm + static_cast<long long>(tile_index) * LWS_TILE_SIZE;
        const unsigned char* tile_ec = ec + static_cast<long long>(tile_index) * (LWS_TILE_SIZE / 2);
        const unsigned char* tile_esc = esc + static_cast<long long>(tile_index) * LWS_ESCAPE_BUDGET;
        int rank = row_escape_offset[static_cast<long long>(n) * k_tile_count + tile_col];

        for (int base = 0; base < LWS_TILE_COLS; base += warpSize) {
            int local_c = base + lane;
            int flat_index = row_offset_in_tile + local_c;
            unsigned char code = lws_ec_code(tile_ec, flat_index);
            bool is_escape = (code == LWS_ESCAPE_CODE);
            unsigned int ballot = __ballot_sync(0xFFFFFFFFu, is_escape);
            unsigned int lane_mask_below = (1u << lane) - 1u;
            int rank_in_group = __popc(ballot & lane_mask_below);

            unsigned char exponent;
            if (is_escape) {
                exponent = tile_esc[rank + rank_in_group];
            } else {
                exponent = codebook[code];
            }

            unsigned char sm_byte = tile_sm[flat_index];
            unsigned short bits = lws_bits_from_parts(sm_byte, exponent);
            bits = lws_extra_decode_ops(bits);
            unsigned int widened = static_cast<unsigned int>(bits) << 16;
            float w = __uint_as_float(widened);
            accumulator += w * x[k_base + local_c];

            rank += __popc(ballot);
        }
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        accumulator += __shfl_down_sync(0xFFFFFFFFu, accumulator, offset);
    }

    if (lane == 0) {
        output[static_cast<long long>(b) * N + n] = accumulator;
    }
}
