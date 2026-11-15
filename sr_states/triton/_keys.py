"""The on-device counterpart to the optimizer key derivation in sr_states._hash."""

from ._common import tl, triton

RNG_WORD_ARGS = ("seed_lo", "seed_hi", "step_lo", "step_hi")


def split_words(value):
    # Use the signed representation so every word has the same Triton i32 ABI.
    # The kernel reinterprets it as uint32 before hashing.
    def signed(word):
        return ((word + (1 << 31)) & 0xFFFFFFFF) - (1 << 31)

    return signed(value), signed(value >> 32)


if triton is not None:
    from .sr_cast import _u32_hash

    @triton.jit
    def _update_key(seed_lo, seed_hi, step_lo, step_hi, param_index):
        key = _u32_hash(seed_lo.to(tl.uint32) ^ 0x9E3779B9)
        key = _u32_hash(key ^ seed_hi.to(tl.uint32))
        key = _u32_hash(key ^ step_lo.to(tl.uint32))
        key = _u32_hash(key ^ step_hi.to(tl.uint32))
        return _u32_hash(key ^ param_index.to(tl.uint32))

    @triton.jit
    def _state_key(key, state_index):
        return _u32_hash(key ^ tl.cast(state_index, tl.uint32))
