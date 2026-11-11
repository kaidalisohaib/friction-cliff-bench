import pytest
import warp as wp

wp.config.quiet = True

try:  # the CPU CI job installs no torch (`uv sync --no-group rl`): skip the learned-method test modules there
    import torch  # noqa: F401
except ImportError:
    collect_ignore = ["test_adaptive.py", "test_ppo.py", "test_rl_env.py"]


def has_cuda() -> bool:
    try:
        return wp.get_cuda_device_count() > 0
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    if has_cuda():
        return
    skip = pytest.mark.skip(reason="no CUDA device")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
