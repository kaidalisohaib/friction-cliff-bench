import pytest
import warp as wp

wp.config.quiet = True


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
