"""Benchmark complete in-place AdamW state updates."""

from benchmarks.bench_states import main

if __name__ == "__main__":
    main(optimizer="adamw")
