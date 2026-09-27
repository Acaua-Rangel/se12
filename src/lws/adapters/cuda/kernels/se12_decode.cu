// Decode-only kernel: rebuilds a whole SE12 tensor's exact BF16 bit pattern
// in VRAM (US-004, AC-009). See se12_common.cuh for the shared decode logic
// and the "UNVERIFIED ON REAL HARDWARE" note — the same caveat applies here.
#include "se12_common.cuh"

// `fallback_offsets[tile]` is the tile's 0-indexed rank among fallback
// tiles (cumsum(fallback_mask) - 1, computed once on the host from the
// unpacked bitmap — a global prefix sum over tiles, not per-weight, so it
// is cheap and belongs on the host, not in this kernel; fused-decode-gemv
// design.md calls this out explicitly). Its value is unused (don't-care)
// for a non-fallback tile.
extern "C" __global__ void lws_se12_decode(
    const unsigned char* sm,
    const unsigned char* ec,
    const unsigned char* esc,
    const unsigned char* codebook,
    const unsigned char* fallback_bitmap,
    const int* fallback_offsets,
    const short* fallback_raw,
    unsigned short* out,
    int tile_count
) {
    int tile_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (tile_index >= tile_count) {
        return;
    }

    unsigned short* tile_out = out + static_cast<long long>(tile_index) * LWS_TILE_SIZE;

    if (lws_is_fallback_tile(fallback_bitmap, tile_index)) {
        int fallback_index = fallback_offsets[tile_index];
        const short* tile_raw = fallback_raw + static_cast<long long>(fallback_index) * LWS_TILE_SIZE;
        for (int i = 0; i < LWS_TILE_SIZE; ++i) {
            tile_out[i] = static_cast<unsigned short>(tile_raw[i]);
        }
        return;
    }

    const unsigned char* tile_sm = sm + static_cast<long long>(tile_index) * LWS_TILE_SIZE;
    const unsigned char* tile_ec = ec + static_cast<long long>(tile_index) * (LWS_TILE_SIZE / 2);
    const unsigned char* tile_esc = esc + static_cast<long long>(tile_index) * LWS_ESCAPE_BUDGET;
    lws_decode_tile(tile_sm, tile_ec, tile_esc, codebook, tile_out);
}
