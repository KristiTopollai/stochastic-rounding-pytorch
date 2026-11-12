# Changelog

## Unreleased

- Optimizer RNG streams use the group seed, global optimizer step, parameter
  position, state index, and flat global element index. Checkpoints now store
  `sr_step`. Legacy checkpoints preserve formats and state bytes and start the
  new stream at step zero.
- The fused backend hook `_step_parameter` now takes `seed` before `offset`.
  Functional optimizer updates interpret `seed` as the update key and `offset`
  as the flat shard start. DTensor execution remains unsupported.
- Per-state `StateFormat` storage, group overrides, shared optimizer bookkeeping,
  and legacy checkpoint migration across eager, compiled, and Triton backends.
- Saturating E4M3/E5M2 FP8 and IEEE FP16 nearest-even/stochastic casts; software
  SR uses 24-bit probabilities while preserving the BF16 stream.
- Independent moment callbacks in fused AdamW and a pre-launch sm_89 FP8 check.
- `AdamWReferenceRN` and `SGDMReferenceRN` accept `state_format=`; a
  `_forced_rounding` class policy keeps every state, group override and loaded
  checkpoint on nearest rounding.
- Exhaustive encoding/boundary tests, mixed-state resume tests, frozen baseline
  byte regressions, and Linux Triton/Inductor checks. Library execution and
  same-environment GPU baseline regressions were validated on H200; the complete
  suite including the FP8 validator fix passed on L40S.
- Direct FP32-to-FP8 nearest conversions avoid intermediate FP16 precision loss;
  FP8 state validation checks FP32 arithmetic separately from exact storage bytes.
- The per-state AdamW example keeps its second moment in BF16 and documents the
  underflow risks of unscaled FP8, cast policies, and extension interfaces.

## 0.1.0

- BF16 stochastic and nearest-even casts, with eager PyTorch, compiled, and Triton implementations.
- Fused Triton AdamW and SGDM with FP32 update arithmetic and BF16 state storage; FP32 states and nearest rounding are also supported.
- Checkpoints preserve state precision, AdamW steps, and rounding counters.
- Synthetic optimizer-step and fixed-input benchmarks compare eager, compiled, and Triton implementations.
