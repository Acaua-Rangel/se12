// INT32/FP32 instruction-throughput microkernels for the performance model's
// calibration (US-015, AC-026). Each thread does a long, register-resident
// dependency chain of FMAs/integer ops on data it already holds — no global
// memory traffic once the operands are loaded — so the measured time is
// purely instruction-issue-bound, giving sustained throughput.
//
// UNVERIFIED ON REAL HARDWARE — see se12_common.cuh's note; this file
// shares that caveat.

#ifndef LWS_CALIBRATION_ITERATIONS
#define LWS_CALIBRATION_ITERATIONS 4096
#endif

extern "C" __global__ void lws_calibrate_fp32_throughput(const float* seeds, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    float accumulator = seeds[i];
    float multiplier = 1.0000001f;
#pragma unroll 8
    for (int step = 0; step < LWS_CALIBRATION_ITERATIONS; ++step) {
        accumulator = accumulator * multiplier + seeds[i];
    }
    out[i] = accumulator;
}

extern "C" __global__ void lws_calibrate_int32_throughput(const int* seeds, int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    int accumulator = seeds[i];
#pragma unroll 8
    for (int step = 0; step < LWS_CALIBRATION_ITERATIONS; ++step) {
        accumulator = (accumulator * 2654435761) ^ seeds[i];
    }
    out[i] = accumulator;
}
