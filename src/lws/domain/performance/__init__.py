"""The roofline speed model (US-015): predicts the fused ÷ baseline latency
ratio from measured hardware facts — bandwidth, instruction throughput,
decode cost per weight — never from simulation. Every term is measured; one
parameter (overlap efficiency) is fitted, and only on the raw kernel, so the
fused prediction is out-of-sample (design.md)."""
