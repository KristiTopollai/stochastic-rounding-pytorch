# Changelog

## 0.1.0

- BF16 stochastic and nearest-even casts, with eager PyTorch, compiled, and Triton implementations.
- Fused Triton AdamW and SGDM with FP32 update arithmetic and BF16 state storage; FP32 states and nearest rounding are also supported.
- Checkpoints preserve state precision, AdamW steps, and rounding counters.
- Synthetic optimizer-step and fixed-input benchmarks compare eager, compiled, and Triton implementations.
- H200 validation: 299 tests passed, with two multi-GPU tests skipped. Three-repeat timing results and raw samples are included in the repository.

For 16,777,216 parameters in one tensor, Triton SR optimizer steps took 145.90 µs
for AdamW and 122.94 µs for SGDM: 2.61× and 1.20× faster than compiled SR.
Measured overhead versus matched Triton NR was 0.27% and 0.25%. These timings
exclude gradients and compilation and do not measure training throughput.
