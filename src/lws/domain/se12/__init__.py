"""SE12: split-exponent, ~12 bits/weight, fixed-rate lossless BF16 codec.

See weight-codec design.md for the format. This package is pure domain code:
numpy integer bit manipulation only, no torch, no I/O.
"""
