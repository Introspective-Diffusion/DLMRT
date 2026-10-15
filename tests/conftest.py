import pytest


@pytest.fixture(scope="session", autouse=True)
def deep_gemm_as_the_models_use_it():
    """DeepGEMM with dlmrt's patched SM90 BF16 GEMM, as every model sets it up, for tests that call it directly."""
    try:
        import deep_gemm  # noqa: F401
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        from dlmrt.ops.extensions import init_deep_gemm

        init_deep_gemm()
