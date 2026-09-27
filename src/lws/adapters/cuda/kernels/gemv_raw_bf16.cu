// Raw-BF16 baseline GEMV kernel (US-005; the baseline principle P-009
// compares the fused kernel against). Reads full 16-bit weights, upcasts
// them exactly to FP32 (`bits << 16` — lossless, design.md), FP32 FMA.
//
// One warp decodes one (output row n, batch element b): the 32 lanes stream
// K in lock-step (coalesced, consecutive addresses per lane), each lane
// accumulates its own partial dot product, then a warp-shuffle tree reduces
// it to lane 0, which writes the result. This is EXACTLY the skeleton
// gemv_se12.cu uses too (P-009) — only the per-weight load+decode step
// differs; the grid, the loop structure and the reduction are identical, so
// the same launch configuration gives bit-identical outputs on both
// kernels for a given configuration (AC-032).
//
// UNVERIFIED ON REAL HARDWARE — see se12_common.cuh's note; this file
// shares that caveat.
#include "se12_common.cuh"

extern "C" __global__ void lws_gemv_raw_bf16(
    const unsigned short* weights,   // [N, K] BF16 bit pattern, row-major
    const float* activations,        // [B, K] FP32 (already upcast if the caller's input was BF16)
    float* output,                   // [B, N] FP32
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

    const unsigned short* row = weights + static_cast<long long>(n) * K;
    const float* x = activations + static_cast<long long>(b) * K;

    float accumulator = 0.0f;
    for (int k = lane; k < K; k += warpSize) {
        unsigned short bits = row[k];
        unsigned int widened = static_cast<unsigned int>(bits) << 16;
        float w = __uint_as_float(widened);
        accumulator += w * x[k];
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        accumulator += __shfl_down_sync(0xFFFFFFFFu, accumulator, offset);
    }

    if (lane == 0) {
        output[static_cast<long long>(b) * N + n] = accumulator;
    }
}
