import pytest

from sr_states._hash import state_key, update_key


@pytest.mark.parametrize(
    "seed,step,index,expected",
    [
        (0, 0, 0, 0x0EAA7511),
        (0, 1, 0, 0x7D2B5EA8),
        (42, 17, 9, 0xFA76534D),
        (0x12345678ABCDEFFE, 0xFEDCBA9876543210, 0xA5A5A5A5, 0xF820C188),
        (-1, 1 << 32, 1, 0x03D350C7),
    ],
)
def test_update_key_known_answers(seed, step, index, expected):
    assert update_key(seed, step, index) == expected


@pytest.mark.parametrize(
    "key,index,expected",
    [
        (0, 0, 0),
        (0, 1, 0x688990C0),
        (1, 0, 0x688990C0),
        (0x89ABCDEF, 1, 0x575E21BD),
        (0xFFFFFFFF, 2, 0x5681FFD9),
    ],
)
def test_state_key_known_answers(key, index, expected):
    assert state_key(key, index) == expected
