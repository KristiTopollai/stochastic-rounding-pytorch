from sr_states.triton import is_triton_available, sr_cast_bf16


def test_optional_triton_api_imports_without_triton_installed():
    assert callable(sr_cast_bf16)
    assert isinstance(is_triton_available(), bool)
