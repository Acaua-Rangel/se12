# Design: streaming lossless-compressed weights

> feature: weight-offload

## Why the effect compounds

Let `T` be the model's BF16 weight bytes, `r ≈ 0.752` the SE12 ratio and `R`
the VRAM left for weights. Both formats fill `R` with resident layers and
stream the rest every token:

```
streamed_bf16 = T − R
streamed_se12 = r·T − R            (0 when r·T ≤ R: the model fits)
ratio         = (r·T − R) / (T − R)
```

Gemma 2 9B (`T ≈ 18.5 GB`, `r·T ≈ 13.9 GB`):

| R (weights resident) | BF16 streamed | SE12 streamed | ratio | at ~12 GB/s: BF16 → SE12 per token |
|---|---|---|---|---|
| 4 GB | 14.5 GB | 9.9 GB | 0.68 | ~1.21 s → ~0.83 s |
| 6 GB | 12.5 GB | 7.9 GB | 0.63 | ~1.04 s → ~0.66 s |
| 10 GB | 8.5 GB | 3.9 GB | 0.46 | ~0.71 s → ~0.33 s |
| 14 GB | 4.5 GB | 0 (fits) | 0 | ~0.38 s → compute-bound |

The ratio falls below the format's 0.75 as `R` grows — the paper's headline
curve. The decode cost is irrelevant here: at ~12 GB/s the GPU has ~27× (T4)
to ~61× (P100) more time per streamed byte than when reading VRAM — hundreds of
instructions per byte.

## Pipeline

- Placement (pure domain): walk layers in forward order, keep them resident
  until the budget (minus KV cache, activations and two staging buffers) is
  used; the rest are streamed. Same order for both formats, so the only
  difference is bytes.
- Host side: streamed layers live in pinned memory, one contiguous block per
  layer (fixed-rate SE12 makes each block's size known up front).
- Device side: two staging buffers sized for the largest streamed layer; while
  layer i computes from buffer A, layer i+1 copies into buffer B on a
  dedicated CUDA stream; events order copy → compute.
- Compute on a staged layer uses exactly the same kernels as resident layers
  (raw BF16 kernel or SE12 fused/exact path), so AC-029's bit-identity follows
  from AC-032/AC-033.
- Overlap efficiency = 1 − (exposed copy time ÷ total copy time), from CUDA
  events on both streams.

## Code layout (P-013 / P-014)

| Concern | Layer | Where |
|---|---|---|
| placement rule, streamed-bytes prediction | domain | `lws.domain.offload` |
| load and benchmark use cases, `WeightStreamer` port | application | `lws.application.runtime`, `lws.application.ports.runtime` |
| pinned buffers, copy stream, events | adapter | `lws.adapters.cuda.weight_streamer` |
| hooking streamed layers into the transformers model | adapter | `lws.adapters.transformers.offloaded_layers` |
| llama.cpp reference run (subprocess) | adapter | `lws.adapters.external.llama_cpp_runner` |
| `lws.offload.load`, `python -m lws.bench.offload` | entrypoint | `lws.offload`, `lws.bench.offload` |
